"""Etape 1 : pretraite tous les examens et met en cache volumes + descripteurs.

Le cache est calcule UNE fois puis relu par tous les entrainements : c'est ce
qui permet d'iterer sur les modeles en minutes plutot qu'en heures.

    python scripts/01_preprocess.py --data data --out cache --workers 24

Produit :
    cache/volumes.npy    (N, 64, 64, 40) float16, unites "fond = 1.0"
    cache/manifest.csv   uid, label, QC, pseudo-centre, 110 descripteurs
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.imaging import preprocess, crop_shape          # noqa: E402
from datscan.features import extract                        # noqa: E402
from datscan.cv import acquisition_signature                # noqa: E402


def _one(args):
    uid, path = args
    vol, qc = preprocess(path)
    row = {"uid": uid, **qc.as_dict()}
    row.update(extract(vol))
    return uid, vol.astype(np.float16), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dossier contenant niftis/")
    ap.add_argument("--out", required=True, help="dossier de cache")
    ap.add_argument("--labels", default=None,
                    help="CSV de labels (defaut : <data>/train_labels.csv si present)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted((data / "niftis").glob("*.nii.gz"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"aucun .nii.gz dans {data / 'niftis'}")
    items = [(p.name[:-len(".nii.gz")], p) for p in paths]
    print(f"{len(items)} examens, {args.workers} processus")

    shape = crop_shape()
    vols = np.zeros((len(items),) + shape, dtype=np.float16)
    rows = [None] * len(items)
    order = {uid: i for i, (uid, _) in enumerate(items)}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, (uid, vol, row) in enumerate(ex.map(_one, items, chunksize=4), 1):
            i = order[uid]
            vols[i] = vol
            rows[i] = row
            if n % 200 == 0 or n == len(items):
                el = time.time() - t0
                print(f"  {n}/{len(items)}  {el:.0f}s  "
                      f"({el / n:.2f}s/examen, reste {el / n * (len(items) - n):.0f}s)",
                      flush=True)

    man = pd.DataFrame(rows)
    labels_path = Path(args.labels) if args.labels else data / "train_labels.csv"
    if labels_path.exists():
        lab = pd.read_csv(labels_path)
        man = man.merge(lab[["uid", "is_pathologic"]], on="uid", how="left")
    man["pseudo_center"] = acquisition_signature(man).to_numpy()

    np.save(out / "volumes.npy", vols)
    man.to_csv(out / "manifest.csv", index=False)

    print(f"\ncache ecrit dans {out}")
    print(f"  volumes.npy   {vols.shape} float16  "
          f"({vols.nbytes / 1e6:.0f} Mo)")
    print(f"  manifest.csv  {man.shape[0]} lignes x {man.shape[1]} colonnes")
    bad = man[~man["ok"].astype(bool)]
    if len(bad):
        print(f"  ATTENTION {len(bad)} examens en echec :")
        print(bad[["uid", "note"]].head(10).to_string(index=False))
    print("\npseudo-centres detectes :")
    print(man["pseudo_center"].value_counts().to_string())
    if "is_pathologic" in man:
        print(f"\nprevalence globale : {man['is_pathologic'].mean():.3f}")
        print(man.groupby("pseudo_center")["is_pathologic"]
              .agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
