"""Etape 2 : modeles sur descripteurs semi-quantitatifs (reference forte).

    python scripts/02_train_tab.py --cache cache --out models/tab

Produit ``gbm.pkl``, ``linear.pkl``, ``oof_gbm.csv``, ``oof_linear.csv``.
A lancer en premier : ces modeles tournent en quelques secondes, se calibrent
tres bien et fournissent une soumission valide et competitive avant meme que le
premier CNN ait fini d'apprendre.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.features import feature_names                                # noqa: E402
from datscan.models_tab import (gbm_importance, train_gbm_cv,             # noqa: E402
                                train_linear_cv)
from datscan.cv import make_folds, per_group_report, report               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--group-cv", type=int, default=1)
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(Path(args.cache) / "manifest.csv")
    man = man[man["is_pathologic"].notna()].reset_index(drop=True)

    cols = [c for c in feature_names() if c in man.columns]
    X = man[cols].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = man["is_pathologic"].to_numpy(dtype=int)
    groups = man["pseudo_center"].to_numpy() if args.group_cv else None
    folds = make_folds(y, groups, args.folds, args.seed)

    print(f"{len(y)} examens | {len(cols)} descripteurs | "
          f"prevalence {y.mean():.3f} | plis "
          f"{'groupes par pseudo-centre' if args.group_cv else 'aleatoires'}")

    print("\nLightGBM")
    oof_g, models_g, info_g = train_gbm_cv(X, y, folds, seed=args.seed)
    rg = report(y, oof_g)
    print(f"  hors-pli : logloss {rg['logloss']:.4f} | auroc {rg['auroc']:.4f} "
          f"| reference constante {rg['logloss_baseline']:.4f}")

    print("\nRegression logistique")
    oof_l, models_l, info_l = train_linear_cv(X, y, folds, C=args.C, seed=args.seed)
    rl = report(y, oof_l)
    print(f"  hors-pli : logloss {rl['logloss']:.4f} | auroc {rl['auroc']:.4f}")

    if groups is not None:
        pg = per_group_report(y, oof_g, groups)
        if len(pg):
            print("\nLightGBM par pseudo-centre (pires en tete) :")
            print(pg[["group", "n", "logloss", "auroc"]].head(8)
                  .round(4).to_string(index=False))

    print("\nDescripteurs les plus utilises :")
    print(gbm_importance(models_g, cols, top=15).round(1).to_string(index=False))

    with (out / "gbm.pkl").open("wb") as f:
        pickle.dump({"models": [m.model_to_string() for m in models_g],
                     "columns": cols, "info": info_g}, f)
    with (out / "linear.pkl").open("wb") as f:
        pickle.dump({"models": models_l, "columns": cols, "info": info_l}, f)

    for name, oof in (("gbm", oof_g), ("linear", oof_l)):
        pd.DataFrame({"uid": man["uid"], "fold": folds, "y": y,
                      "pred": oof}).to_csv(out / f"oof_{name}.csv", index=False)
    print(f"\necrit dans {out}")


if __name__ == "__main__":
    main()
