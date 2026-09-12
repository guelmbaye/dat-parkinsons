"""Preprocessing DaT-SPECT : canonicalisation -> normalisation -> recadrage striatal.

Ce module est LE point de contact avec les images brutes. Il est utilise a
l'identique a l'entrainement et a l'inference (aucune divergence possible).

Choix de conception
-------------------
* Aucun atlas externe n'est requis : tout est derive de l'image elle-meme.
  Le conteneur d'evaluation n'a pas d'acces reseau, et les templates SPECT
  publics sont mal licencies -> on reste auto-suffisant.
* Toutes les etapes sont deterministes (pas de RNG) et vectorisees numpy/scipy,
  ~0.3-1.5 s par examen sur un coeur -> quelques minutes sur 24 vCPU.
* L'intensite est ramenee en unites "fond non specifique = 1.0", ce qui est
  l'equivalent robuste du SBR clinique et supprime l'essentiel de l'effet
  scanner/centre.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Parametres geometriques (mm). Modifier ici change train ET inference.
# ---------------------------------------------------------------------------
TARGET_SPACING = 2.0          # resolution isotrope de travail
CROP_MM = (128.0, 128.0, 80.0)  # taille de la fenetre (X=G-D, Y=P-A, Z=I-S)
MAX_YAW_DEG = 25.0            # correction de rotation axiale bornee
MIDLINE_SEARCH_MM = 24.0      # +/- amplitude de recherche du plan sagittal median
MASK_DS = 2                   # sous-echantillonnage pour la morphologie du masque


def crop_shape(spacing: float = TARGET_SPACING) -> tuple[int, int, int]:
    return tuple(int(round(c / spacing)) for c in CROP_MM)  # (64, 64, 40)


@dataclass
class ScanQC:
    """Traces de controle qualite, ecrites dans le manifeste."""
    orig_shape: tuple = (0, 0, 0)
    orig_spacing: tuple = (0.0, 0.0, 0.0)
    background: float = 0.0        # counts bruts du fond non specifique
    peak_ratio: float = 0.0        # p99.9 / fond  (~ contraste global)
    mask_voxels: int = 0
    yaw_deg: float = 0.0
    midline_shift_mm: float = 0.0
    slab_z_frac: float = 0.0       # position relative du plan striatal
    n_frames: int = 1
    ok: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("orig_shape", "orig_spacing"):
            v = d.pop(k)
            for i, ax in enumerate("xyz"):
                d[f"{k}_{ax}"] = v[i]
        return d


# ---------------------------------------------------------------------------
# 1. Lecture
# ---------------------------------------------------------------------------
def load_canonical(path) -> tuple[np.ndarray, np.ndarray, int]:
    """Charge un NIfTI et le ramene en orientation RAS+.

    Retourne (volume float32 [X=droite, Y=avant, Z=haut], spacing_mm, n_frames).
    La reorientation est indispensable : les centres n'exportent pas tous leurs
    reconstructions dans la meme convention, et un modele entraine sur une
    orientation ne generalise pas a l'autre.
    """
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj)
    n_frames = 1
    if data.ndim > 3:
        n_frames = int(np.prod(data.shape[3:]))
        # Un seul volume reconstruit est attendu ; si plusieurs, on moyenne.
        data = data.reshape(data.shape[:3] + (-1,)).mean(axis=3)
    vol = np.ascontiguousarray(data, dtype=np.float32)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    zooms = np.where(np.isfinite(zooms) & (zooms > 1e-3), zooms, 1.0)
    return vol, zooms, n_frames


def resample_iso(vol: np.ndarray, zooms: np.ndarray,
                 spacing: float = TARGET_SPACING) -> np.ndarray:
    """Reechantillonnage trilineaire vers une grille isotrope."""
    factors = zooms / float(spacing)
    if np.allclose(factors, 1.0, atol=1e-3):
        return vol
    return ndi.zoom(vol, factors, order=1, prefilter=False, mode="nearest")


# ---------------------------------------------------------------------------
# 2. Masque tete + normalisation d'intensite
# ---------------------------------------------------------------------------
def head_mask(vol: np.ndarray, ds: int = MASK_DS) -> np.ndarray:
    """Masque binaire de la tete, robuste au contraste variable.

    L'ecretage a p99 avant seuillage est essentiel : sans lui, le striatum (tres
    chaud) tire le seuil d'Otsu vers le haut et le masque se reduit aux noyaux
    gris. La morphologie tourne sur une grille sous-echantillonnee (~8x moins de
    voxels) : la precision au demi-centimetre suffit ici.
    """
    small = vol[::ds, ::ds, ::ds] if ds > 1 else vol
    sm = ndi.gaussian_filter(small, sigma=2.0 / ds)
    hi = float(np.percentile(sm, 99.0))
    if not np.isfinite(hi) or hi <= 0:
        return np.zeros(vol.shape, dtype=bool)
    clipped = np.minimum(sm, hi)
    m = clipped > max(_otsu(clipped), 0.05 * hi)
    m = ndi.binary_closing(m, structure=np.ones((3, 3, 3)))
    m = _largest_component(m)
    if m.any():
        m = ndi.binary_fill_holes(m)
    if ds > 1:
        m = np.repeat(np.repeat(np.repeat(m, ds, 0), ds, 1), ds, 2)
        m = m[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
        pad = [(0, max(0, vol.shape[i] - m.shape[i])) for i in range(3)]
        if any(p[1] for p in pad):
            m = np.pad(m, pad, mode="edge")
    return m


def _otsu(vol: np.ndarray, nbins: int = 256) -> float:
    v = vol.ravel()
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    tot = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = (tot - m0) / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
    var_between = np.nan_to_num(var_between)
    return float(centers[int(np.argmax(var_between))])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def background_level(vol: np.ndarray, mask: np.ndarray) -> float:
    """Estime le fond non specifique (equivalent VOI occipitale, sans atlas).

    Moyenne tronquee p25-p85 des voxels intra-tete : exclut le bord du masque
    (partiel volume, bas) et le striatum (top ~3 %, haut).
    """
    vals = vol[mask]
    if vals.size < 100:
        vals = vol[vol > 0]
    if vals.size < 10:
        return 1.0
    lo, hi = np.percentile(vals, [25.0, 85.0])
    sel = vals[(vals >= lo) & (vals <= hi)]
    bg = float(sel.mean()) if sel.size else float(np.median(vals))
    return bg if bg > 1e-6 else 1.0


# ---------------------------------------------------------------------------
# 3. Recalage leger : plan striatal, lacet, ligne mediane
# ---------------------------------------------------------------------------
def striatal_slab_z(vol: np.ndarray, mask: np.ndarray) -> int:
    """Coupe axiale de captation maximale ~ plan des noyaux gris centraux."""
    prof = (vol * mask).sum(axis=(0, 1))
    if prof.sum() <= 0:
        return vol.shape[2] // 2
    prof = ndi.uniform_filter1d(prof, size=3, mode="nearest")
    return int(np.argmax(prof))


def _rot2d(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)],
                     [np.sin(a), np.cos(a)]], dtype=np.float64)


def estimate_yaw(mask2d: np.ndarray) -> float:
    """Lacet estime par l'axe principal de l'ellipse cranienne (vue axiale).

    Le grand axe d'une tete en coupe axiale est l'axe antero-posterieur : on
    mesure son ecart a +Y et on le borne, pour ne jamais degrader une
    acquisition deja bien positionnee.
    """
    idx = np.argwhere(mask2d)
    if idx.shape[0] < 50:
        return 0.0
    idx = idx - idx.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(idx.T))
    v = evecs[:, int(np.argmax(evals))]
    if v[1] < 0:
        v = -v
    ang = float(np.degrees(np.arctan2(v[0], v[1])))
    if not np.isfinite(ang):
        return 0.0
    return float(np.clip(ang, -MAX_YAW_DEG, MAX_YAW_DEG))


def find_midline(profile: np.ndarray, spacing: float = TARGET_SPACING) -> float:
    """Plan sagittal median par maximisation de la symetrie gauche/droite.

    Nettement plus stable que le barycentre d'intensite : en cas de denervation
    unilaterale severe le barycentre se decale vers le cote sain et decentre
    tout le recadrage.
    """
    prof = np.asarray(profile, dtype=np.float64)
    n = prof.size
    if prof.sum() <= 0:
        return (n - 1) / 2.0
    x = np.arange(n, dtype=np.float64)
    c0 = float(np.average(x, weights=prof))
    rad = MIDLINE_SEARCH_MM / spacing
    cands = np.arange(max(1.0, c0 - rad), min(n - 2.0, c0 + rad) + 0.25, 0.25)
    best_c, best_s = c0, -np.inf
    norm_p = float(np.linalg.norm(prof))
    for c in cands:
        mirrored = np.interp(2.0 * c - x, x, prof, left=0.0, right=0.0)
        denom = norm_p * float(np.linalg.norm(mirrored))
        score = float(prof @ mirrored / denom) if denom > 0 else -np.inf
        if score > best_s:
            best_s, best_c = score, float(c)
    return best_c


# ---------------------------------------------------------------------------
# 4. Chaine complete
# ---------------------------------------------------------------------------
def preprocess(path, spacing: float = TARGET_SPACING) -> tuple[np.ndarray, ScanQC]:
    """Retourne (volume normalise de forme crop_shape(), QC).

    Le volume rendu est en unites "fond = 1.0" : une valeur de 6.0 signifie une
    captation six fois superieure au fond non specifique. Aucune exception n'est
    propagee : un examen illisible ressort en volume nul avec ``qc.ok = False``,
    ce qui evite qu'un seul fichier fasse echouer toute une soumission.
    """
    qc = ScanQC()
    out_shape = crop_shape(spacing)
    try:
        raw, zooms, n_frames = load_canonical(path)
        qc.orig_shape = tuple(int(v) for v in raw.shape)
        qc.orig_spacing = tuple(round(float(z), 4) for z in zooms)
        qc.n_frames = n_frames

        vol = resample_iso(raw, zooms, spacing)
        vol = np.maximum(np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        mask = head_mask(vol)
        qc.mask_voxels = int(mask.sum())
        if qc.mask_voxels < 500:
            mask = vol > np.percentile(vol, 70.0)
            qc.note = "masque de secours"

        bg = background_level(vol, mask)
        qc.background = bg
        vol = vol / bg
        qc.peak_ratio = float(np.percentile(vol[mask], 99.9)) if mask.any() else 0.0

        half_z = max(1, int(round(10.0 / spacing)))
        z0 = striatal_slab_z(vol, mask)
        qc.slab_z_frac = float(z0 / max(1, vol.shape[2] - 1))
        z1, z2 = max(0, z0 - half_z), min(vol.shape[2], z0 + half_z + 1)

        # Projection axiale du plan striatal : sert au lacet, a la ligne mediane
        # et au centrage antero-posterieur, pour un cout negligeable.
        slab = (vol * mask)[:, :, z1:z2].sum(axis=2)
        slab_mask = mask[:, :, z1:z2].any(axis=2)

        yaw = estimate_yaw(slab_mask)
        qc.yaw_deg = yaw
        # affine_transform applique output[o] = input[mat @ o + offset] : la
        # matrice va de la sortie VERS l'entree, donc redresser d'un lacet +yaw
        # demande la rotation inverse. (Signe verifie par un test de symetrie.)
        rot = _rot2d(-yaw)
        c2 = np.array(ndi.center_of_mass(slab_mask), dtype=np.float64) \
            if slab_mask.any() else np.array([(vol.shape[0] - 1) / 2.0,
                                              (vol.shape[1] - 1) / 2.0])

        if abs(yaw) >= 0.5:
            slab_r = ndi.affine_transform(slab, rot, offset=c2 - rot @ c2,
                                          order=1, mode="constant", cval=0.0)
        else:
            slab_r = slab

        cx = find_midline(slab_r.sum(axis=1), spacing)
        qc.midline_shift_mm = float((cx - (vol.shape[0] - 1) / 2.0) * spacing)
        yprof = slab_r.sum(axis=0)
        cy = float(np.average(np.arange(yprof.size), weights=yprof)) \
            if yprof.sum() > 0 else (vol.shape[1] - 1) / 2.0

        # Lacet + translation + recadrage en UNE seule interpolation :
        # rapide, et sans accumulation de flou.
        mat = np.eye(3)
        mat[:2, :2] = rot
        center = np.array([c2[0], c2[1], 0.0])
        shift = np.array([cx - out_shape[0] / 2.0,
                          cy - out_shape[1] / 2.0,
                          z0 - out_shape[2] / 2.0])
        offset = mat @ (shift - center) + center
        out = ndi.affine_transform(vol, mat, offset=offset,
                                   output_shape=out_shape, order=1,
                                   mode="constant", cval=0.0)
        return np.clip(out, 0.0, 20.0).astype(np.float32), qc

    except Exception as exc:  # aucune exception ne doit remonter a l'inference
        qc.ok = False
        qc.note = f"{type(exc).__name__}: {exc}"[:200]
        return np.zeros(out_shape, dtype=np.float32), qc
