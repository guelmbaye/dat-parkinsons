"""Schema de validation et metriques.

Le classement final se joue sur un jeu de test prive dont la composition par
centre est inconnue. Une validation croisee aleatoire surestime donc la
performance : deux examens du meme scanner se retrouvent des deux cotes du pli
et le modele peut s'appuyer sur des indices propres au centre.

On construit ici un identifiant de centre approche a partir de la signature
d'acquisition (matrice + resolution + orientation), puis on valide en
StratifiedGroupKFold. Le score obtenu repond a la vraie question : "que vaut ce
modele sur un scanner qu'il n'a jamais vu ?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SIG_COLS = ["orig_shape_x", "orig_shape_y", "orig_shape_z",
            "orig_spacing_x", "orig_spacing_y", "orig_spacing_z"]


def acquisition_signature(df: pd.DataFrame, min_size: int = 8) -> pd.Series:
    """Pseudo-centre : signature matrice + resolution, arrondie au 1/100 mm.

    Les groupes trop petits sont fusionnes dans "autre" pour eviter des plis
    degeneres. Cette colonne sert UNIQUEMENT au decoupage de validation ; elle
    n'est jamais donnee au modele (ce serait le meilleur moyen d'apprendre la
    prevalence par centre, qui ne se transporte pas).
    """
    parts = []
    for c in SIG_COLS:
        v = df[c] if c in df else pd.Series(0, index=df.index)
        parts.append(np.round(v.astype(float), 2).astype(str))
    sig = parts[0]
    for p in parts[1:]:
        sig = sig + "|" + p
    counts = sig.value_counts()
    small = set(counts[counts < min_size].index)
    return sig.where(~sig.isin(small), "autre").rename("pseudo_center")


def make_folds(y: np.ndarray, groups: np.ndarray | None, n_splits: int = 5,
               seed: int = 2026) -> np.ndarray:
    """Identifiants de pli. Grouped si ``groups`` est fourni, stratifie sinon."""
    folds = np.full(len(y), -1, dtype=int)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)
        it = splitter.split(np.zeros(len(y)), y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        it = splitter.split(np.zeros(len(y)), y)
    for k, (_, va) in enumerate(it):
        folds[va] = k
    return folds


def report(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> dict:
    """Metriques du challenge : log loss (classement) et AUROC (indicatif)."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y, dtype=int)
    out = {"logloss": float(log_loss(y, p, labels=[0, 1])), "n": int(len(y))}
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    prior = float(y.mean())
    base = np.clip(np.full_like(p, prior), eps, 1 - eps)
    out["logloss_baseline"] = float(log_loss(y, base, labels=[0, 1]))
    out["gain_vs_baseline"] = out["logloss_baseline"] - out["logloss"]
    out["brier"] = float(np.mean((p - y) ** 2))
    return out


def per_group_report(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Log loss par pseudo-centre : revele les centres ou le modele derape."""
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            continue
        r = report(y[m], p[m])
        r["group"] = g
        rows.append(r)
    return pd.DataFrame(rows).sort_values("logloss", ascending=False) if rows \
        else pd.DataFrame()
