"""Descripteurs semi-quantitatifs striataux calcules sur le volume pretraite.

Ce sont, volontairement, les memes grandeurs que celles utilisees en routine
clinique pour lire un DaTscan : intensite de fixation rapportee au fond,
rapport putamen/caude (gradient antero-posterieur), etendue du noyau, et
asymetrie droite/gauche. Elles alimentent un modele a arbres qui :

* fournit une reference forte, calibree et interpretable des le premier jour ;
* se combine tres bien avec un CNN (erreurs peu correlees) ;
* reste stable d'un centre a l'autre car tout est exprime en ratios.

Point cle : le cote atteint est arbitraire d'un patient a l'autre. Toutes les
grandeurs bilaterales sont donc reduites en min / max / moyenne / asymetrie, ce
qui les rend invariantes par echange gauche-droite. Le modele n'a aucune raison
d'apprendre "l'hemisphere gauche est special".
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .imaging import TARGET_SPACING, crop_shape

# Boite de recherche striatale, en mm par rapport au centre du volume recadre.
SIDE_X_MM = (5.0, 42.0)     # distance a la ligne mediane
SIDE_Y_MM = (-42.0, 42.0)   # postero-anterieur
SIDE_Z_MM = (-14.0, 14.0)   # inferieur-superieur
EPS = 1e-6


def _slice_mm(lo_mm, hi_mm, center, n, spacing):
    lo = int(np.clip(round(center + lo_mm / spacing), 0, n))
    hi = int(np.clip(round(center + hi_mm / spacing), 0, n))
    return slice(min(lo, hi), max(lo, hi) + 1)


def _side_features(box: np.ndarray, spacing: float) -> dict:
    """Descripteurs d'un hemistriatum a partir de sa boite de recherche.

    ``box`` est en unites "fond = 1.0" et oriente [X vers l'exterieur,
    Y vers l'avant, Z vers le haut].
    """
    vox_ml = (spacing ** 3) / 1000.0
    f: dict = {}
    flat = box.ravel()
    if flat.size == 0 or not np.isfinite(flat).any():
        return {k: 0.0 for k in _SIDE_KEYS}

    peak = float(np.percentile(flat, 99.5))
    ntop = max(1, int(0.002 * flat.size))
    top = np.sort(flat)[-ntop:]
    f["peak"] = peak
    f["top_mean"] = float(top.mean())
    f["specific_peak"] = max(peak - 1.0, 0.0)          # SBR au pic

    # VOI adaptative a mi-hauteur au-dessus du fond : c'est la definition
    # classique d'un contour isocontour 50 %, insensible au gain du scanner.
    thr = 1.0 + 0.5 * max(peak - 1.0, EPS)
    voi = box >= thr
    if voi.sum() >= 5:
        voi = _largest_cc(voi)
    n_voi = int(voi.sum())
    f["voi_ml"] = n_voi * vox_ml

    if n_voi < 5:
        for k in _SIDE_KEYS:
            f.setdefault(k, 0.0)
        return f

    vals = box[voi]
    f["voi_mean"] = float(vals.mean())
    f["voi_specific_mean"] = float(np.maximum(vals - 1.0, 0.0).mean())
    f["specific_sum_ml"] = float(np.maximum(vals - 1.0, 0.0).sum() * vox_ml)

    idx = np.argwhere(voi).astype(np.float64)
    w = np.maximum(box[voi] - 1.0, EPS)
    centroid = (idx * w[:, None]).sum(0) / w.sum()
    for i, ax in enumerate("xyz"):
        f[f"extent_{ax}_mm"] = float((idx[:, i].max() - idx[:, i].min() + 1) * spacing)
        f[f"centroid_{ax}_mm"] = float((centroid[i] - box.shape[i] / 2.0) * spacing)
    f["elongation_yx"] = f["extent_y_mm"] / (f["extent_x_mm"] + EPS)

    # Gradient antero-posterieur : signature de l'atteinte putaminale.
    y = idx[:, 1]
    y_lo, y_hi = y.min(), y.max()
    span = max(y_hi - y_lo, EPS)
    post = vals[y <= y_lo + 0.4 * span]       # putamen (posterieur)
    ant = vals[y >= y_hi - 0.4 * span]        # caude (anterieur)
    p_mean = float(np.maximum(post - 1.0, 0.0).mean()) if post.size else 0.0
    a_mean = float(np.maximum(ant - 1.0, 0.0).mean()) if ant.size else 0.0
    f["post_specific"] = p_mean
    f["ant_specific"] = a_mean
    f["post_ant_ratio"] = p_mean / (a_mean + EPS)

    # Pente normalisee de l'intensite le long de l'axe antero-posterieur.
    if np.ptp(y) > 1:
        slope = np.polyfit(y * spacing, vals, 1)[0]
        f["ap_slope"] = float(slope / (f["voi_specific_mean"] + EPS))
    else:
        f["ap_slope"] = 0.0

    # Volumes au-dessus de plusieurs seuils de SBR : profil de severite.
    for t in (1.5, 2.0, 3.0, 4.0, 5.0):
        f[f"vol_gt{t:g}_ml"] = float((box >= t).sum() * vox_ml)

    for k in _SIDE_KEYS:
        f.setdefault(k, 0.0)
    return f


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


_SIDE_KEYS = [
    "peak", "top_mean", "specific_peak", "voi_ml", "voi_mean",
    "voi_specific_mean", "specific_sum_ml",
    "extent_x_mm", "extent_y_mm", "extent_z_mm",
    "centroid_x_mm", "centroid_y_mm", "centroid_z_mm", "elongation_yx",
    "post_specific", "ant_specific", "post_ant_ratio", "ap_slope",
    "vol_gt1.5_ml", "vol_gt2_ml", "vol_gt3_ml", "vol_gt4_ml", "vol_gt5_ml",
]


def extract(vol: np.ndarray, spacing: float = TARGET_SPACING) -> dict:
    """Descripteurs complets d'un volume pretraite.

    Retourne un dictionnaire plat de flottants. Les cles sont stables : l'ordre
    des colonnes est fige au moment de l'entrainement et rejoue a l'inference.
    """
    shape = vol.shape
    cx, cy, cz = (s / 2.0 for s in shape)
    sy = _slice_mm(*SIDE_Y_MM, cy, shape[1], spacing)
    sz = _slice_mm(*SIDE_Z_MM, cz, shape[2], spacing)

    # x > centre = cote droit du patient (convention RAS apres canonicalisation).
    right = vol[_slice_mm(*SIDE_X_MM, cx, shape[0], spacing), sy, sz]
    left = vol[_slice_mm(-SIDE_X_MM[1], -SIDE_X_MM[0], cx, shape[0], spacing), sy, sz]
    left = left[::-1]  # miroir : les deux cotes vus dans le meme repere

    fr = _side_features(right, spacing)
    fl = _side_features(left, spacing)

    out: dict = {}
    for k in _SIDE_KEYS:
        a, b = float(fl[k]), float(fr[k])
        lo, hi = (a, b) if a <= b else (b, a)
        out[f"{k}__min"] = lo
        out[f"{k}__max"] = hi
        out[f"{k}__mean"] = 0.5 * (a + b)
        out[f"{k}__asym"] = (hi - lo) / (abs(hi) + abs(lo) + EPS)

    # Descripteurs globaux du volume recadre.
    v = vol.ravel()
    vox_ml = (spacing ** 3) / 1000.0
    out["glob_peak"] = float(np.percentile(v, 99.9))
    out["glob_p99"] = float(np.percentile(v, 99.0))
    out["glob_p95"] = float(np.percentile(v, 95.0))
    out["glob_mean"] = float(v.mean())
    out["glob_std"] = float(v.std())
    for t in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        out[f"glob_vol_gt{t:g}_ml"] = float((v >= t).sum() * vox_ml)
    out["glob_specific_sum_ml"] = float(np.maximum(v - 1.0, 0.0).sum() * vox_ml)

    # Symetrie globale du volume recadre : proxy direct de l'asymetrie de
    # fixation, independant de la segmentation.
    mirror = vol[::-1]
    num = float((vol * mirror).sum())
    den = float(np.sqrt((vol ** 2).sum() * (mirror ** 2).sum()) + EPS)
    out["glob_mirror_corr"] = num / den
    diff = np.abs(vol - mirror)
    out["glob_mirror_l1"] = float(diff.sum() / (vol.sum() + EPS))

    # Profil antero-posterieur de la bande striatale (forme "virgule" -> "point").
    band = vol[:, :, _slice_mm(-10.0, 10.0, shape[2] / 2.0, shape[2], spacing)]
    prof = np.maximum(band.mean(axis=(0, 2)) - 1.0, 0.0)
    if prof.sum() > 0:
        yy = np.arange(prof.size) * spacing
        m = float((prof * yy).sum() / prof.sum())
        out["prof_y_mean_mm"] = m - shape[1] * spacing / 2.0
        out["prof_y_std_mm"] = float(np.sqrt((prof * (yy - m) ** 2).sum() / prof.sum()))
        out["prof_y_skew"] = float(
            (prof * (yy - m) ** 3).sum() / prof.sum()
            / (out["prof_y_std_mm"] ** 3 + EPS))
    else:
        out["prof_y_mean_mm"] = out["prof_y_std_mm"] = out["prof_y_skew"] = 0.0

    return out


def feature_names(spacing: float = TARGET_SPACING) -> list[str]:
    """Liste ordonnee des colonnes, derivee d'un volume factice."""
    return sorted(extract(np.ones(crop_shape(spacing), dtype=np.float32),
                          spacing).keys())
