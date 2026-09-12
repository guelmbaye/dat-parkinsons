"""Generateur de fantomes DaT-SPECT synthetiques.

But : executer et valider toute la chaine (pretraitement -> features -> modele
-> soumission) sans les donnees reelles du challenge. Trois "centres" simules,
avec des matrices, des resolutions, des orientations et des niveaux de comptage
differents -- exactement le type d'heterogeneite decrit par l'organisateur.

Usage :
    python tools/make_phantom.py --out data_phantom --n 240
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage as ndi

# (matrice, spacing_mm, comptages relatifs, FWHM mm)
CENTERS = [
    ("A", (128, 128, 128), 2.46, 1.0, 8.0),
    ("B", (142, 142, 112), 3.895, 0.55, 11.0),
    ("C", (96, 96, 64), 4.42, 2.2, 9.5),
]
AXCODES = [("R", "A", "S"), ("L", "P", "S"), ("L", "A", "S")]


def _ellipsoid(grid_mm, center, radii, angle=0.0):
    x, y, z = grid_mm
    dx, dy, dz = x - center[0], y - center[1], z - center[2]
    ca, sa = np.cos(angle), np.sin(angle)
    rx, ry = ca * dx + sa * dy, -sa * dx + ca * dy
    return ((rx / radii[0]) ** 2 + (ry / radii[1]) ** 2 + (dz / radii[2]) ** 2) <= 1.0


def make_volume(rng, shape, spacing, counts, fwhm, pathologic):
    nx, ny, nz = shape
    ax = (np.arange(nx) - nx / 2) * spacing
    ay = (np.arange(ny) - ny / 2) * spacing
    az = (np.arange(nz) - nz / 2) * spacing
    grid = np.meshgrid(ax, ay, az, indexing="ij")

    tilt = np.radians(rng.uniform(-12, 12))
    shift = rng.uniform(-8, 8, size=3)
    grid = [g - s for g, s in zip(grid, shift)]

    head = _ellipsoid(grid, (0, 0, 0), (78, 96, 78), tilt)
    vol = np.where(head, 1.0, 0.02)                     # fond non specifique

    # Position des striata (comma) : caude anterieur + putamen posterieur.
    for side in (-1.0, 1.0):
        if pathologic:
            # Atteinte posterieure predominante, asymetrique.
            severity = rng.uniform(0.35, 0.9)
            asym = rng.uniform(0.0, 0.55) * (1.0 if side > 0 else -1.0)
            put_gain = np.clip(1.0 - severity - asym, 0.02, 1.0)
            cau_gain = np.clip(1.0 - 0.45 * severity - 0.5 * asym, 0.15, 1.0)
        else:
            put_gain = rng.uniform(0.88, 1.0)
            cau_gain = rng.uniform(0.9, 1.0)

        peak = rng.uniform(5.5, 9.0)
        cx = side * 13.0
        for gain, cy, radii in ((cau_gain, 10.0, (6.0, 8.0, 7.0)),
                                (put_gain, -6.0, (5.5, 11.0, 6.5))):
            blob = _ellipsoid(grid, (cx, cy, 2.0), radii, tilt)
            vol = np.where(blob, np.maximum(vol, 1.0 + (peak - 1.0) * gain), vol)

    vol = ndi.gaussian_filter(vol, sigma=fwhm / 2.355 / spacing)
    lam = np.maximum(vol * counts * 900.0, 0.0)
    vol = rng.poisson(lam).astype(np.float32)
    return np.clip(vol, 0, 65535).astype(np.uint16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_phantom")
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prevalence", type=float, default=0.55)
    args = p.parse_args()

    out = Path(args.out)
    (out / "niftis").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for i in range(args.n):
        ci = int(rng.integers(len(CENTERS)))
        name, shape, spacing, counts, fwhm = CENTERS[ci]
        y = int(rng.random() < args.prevalence)
        vol = make_volume(rng, shape, spacing, counts * rng.uniform(0.7, 1.4),
                          fwhm, bool(y))

        affine = np.diag([spacing, spacing, spacing, 1.0])
        img = nib.Nifti1Image(vol, affine)
        img = img.as_reoriented(nib.orientations.ornt_transform(
            nib.orientations.axcodes2ornt(("R", "A", "S")),
            nib.orientations.axcodes2ornt(AXCODES[ci])))
        img.header.set_zooms((spacing,) * 3)
        uid = f"ph{i:05d}"
        nib.save(img, out / "niftis" / f"{uid}.nii.gz")
        rows.append({"uid": uid, "is_pathologic": float(y), "center_true": name})

    df = pd.DataFrame(rows)
    df[["uid", "is_pathologic"]].to_csv(out / "train_labels.csv", index=False)
    df.to_csv(out / "_truth_with_center.csv", index=False)
    print(f"{len(df)} fantomes ecrits dans {out} "
          f"(prevalence {df.is_pathologic.mean():.2f})")


if __name__ == "__main__":
    main()
