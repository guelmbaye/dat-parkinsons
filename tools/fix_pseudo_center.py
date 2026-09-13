"""Recalcule la colonne pseudo_center d'un manifeste deja construit.

    python tools/fix_pseudo_center.py --cache cache

Evite de refaire tout le pretraitement pour une simple correction de
regroupement : les colonnes source sont deja dans manifest.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.cv import acquisition_signature  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
args = ap.parse_args()

path = Path(args.cache) / "manifest.csv"
man = pd.read_csv(path)
before = man["pseudo_center"].nunique() if "pseudo_center" in man else 0
man["pseudo_center"] = acquisition_signature(man).to_numpy()
man.to_csv(path, index=False)

print(f"pseudo-centres : {before} -> {man['pseudo_center'].nunique()}\n")
tab = man.groupby("pseudo_center")["is_pathologic"].agg(["size", "mean"])
print(tab.sort_values("size", ascending=False).round(3).to_string())
print(f"\nprevalence globale : {man['is_pathologic'].mean():.3f}")
