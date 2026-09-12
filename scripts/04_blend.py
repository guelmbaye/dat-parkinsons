"""Etape 4 : fusion des modeles et calibration finale.

    python scripts/04_blend.py --out models/blend \\
        --oof gbm=models/tab/oof_gbm.csv \\
        --oof linear=models/tab/oof_linear.csv \\
        --oof resnet3d=models/resnet3d/oof.csv

Tout est ajuste sur les predictions HORS-PLI. Ajuster la calibration sur des
predictions in-fold donnerait une log loss flatteuse en validation et un
resultat degrade sur le test prive : c'est l'erreur la plus courante sur une
competition notee en log loss.

Produit ``blend.json``, relu tel quel par ``main.py`` a l'inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.blend import (LogitBlender, PlattScaler, shrink_to_prior,  # noqa: E402
                           to_logit)
from datscan.cv import per_group_report, report                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", action="append", required=True,
                    metavar="NOM=CHEMIN",
                    help="peut etre repete : --oof gbm=models/tab/oof_gbm.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None,
                    help="cache/manifest.csv, pour le detail par pseudo-centre")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="retrecissement vers la prevalence (0.05-0.15 = assurance)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, names = {}, []
    for spec in args.oof:
        if "=" not in spec:
            raise SystemExit(f"format attendu NOM=CHEMIN, recu : {spec}")
        name, path = spec.split("=", 1)
        df = pd.read_csv(path)[["uid", "y", "pred"]].rename(columns={"pred": name})
        frames[name] = df
        names.append(name)

    merged = frames[names[0]]
    for n in names[1:]:
        merged = merged.merge(frames[n].drop(columns=["y"]), on="uid", how="inner")
    if merged.empty:
        raise SystemExit("aucun uid commun entre les fichiers hors-pli")

    y = merged["y"].to_numpy(dtype=int)
    print(f"{len(merged)} examens communs | prevalence {y.mean():.3f}\n")

    print(f"{'modele':16} {'logloss':>9} {'auroc':>8} {'brier':>8}")
    for n in names:
        r = report(y, merged[n].to_numpy())
        print(f"{n:16} {r['logloss']:9.4f} {r['auroc']:8.4f} {r['brier']:8.4f}")
    base = report(y, np.full(len(y), y.mean()))
    print(f"{'(constante)':16} {base['logloss']:9.4f}")

    logits = np.column_stack([to_logit(merged[n].to_numpy()) for n in names])
    blender = LogitBlender().fit(logits, y, names)
    p_blend = blender.predict(logits)
    print(f"\nfusion : {blender.describe()}")
    rb = report(y, p_blend)
    print(f"fusion  logloss {rb['logloss']:.4f} | auroc {rb['auroc']:.4f}")

    platt = PlattScaler().fit(blender.predict(logits, apply_clip=False), y)
    p_final = platt.transform(blender.predict(logits, apply_clip=False))
    rp = report(y, p_final)
    print(f"platt a={platt.a:.3f} b={platt.b:+.3f} ecretage={platt.clip_eps:.4f}")
    print(f"final   logloss {rp['logloss']:.4f} | auroc {rp['auroc']:.4f} | "
          f"brier {rp['brier']:.4f}")

    prior = float(y.mean())

    # La fusion et le Platt ajoutent k+3 parametres ajustes sur ces memes
    # points hors-pli : le score ci-dessus est donc legerement optimiste. On le
    # reestime en refaisant fusion + calibration en validation croisee.
    from sklearn.model_selection import StratifiedKFold
    nested = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(logits, y):
        b2 = LogitBlender().fit(logits[tr], y[tr], names)
        p2 = PlattScaler().fit(b2.predict(logits[tr], apply_clip=False), y[tr])
        nested[va] = p2.transform(b2.predict(logits[va], apply_clip=False))
    rn = report(y, nested)
    print(f"estimation croisee de la calibration : logloss {rn['logloss']:.4f} "
          f"(c'est ce chiffre qu'il faut comparer d'une iteration a l'autre)")

    print("\nsensibilite au retrecissement vers la prevalence "
          "(assurance contre un decalage de prevalence) :")
    for a in (0.0, 0.05, 0.10, 0.15, 0.25):
        rs = report(y, np.clip(shrink_to_prior(p_final, prior, a),
                               platt.clip_eps, 1 - platt.clip_eps))
        print(f"  alpha={a:.2f}  logloss {rs['logloss']:.4f}")

    if args.manifest:
        man = pd.read_csv(args.manifest)[["uid", "pseudo_center"]]
        g = merged.merge(man, on="uid", how="left")["pseudo_center"].to_numpy()
        pg = per_group_report(y, p_final, g)
        if len(pg):
            print("\nfinal par pseudo-centre (pires en tete) :")
            print(pg[["group", "n", "logloss", "auroc"]].head(8)
                  .round(4).to_string(index=False))

    recipe = {
        "models": names,
        "weights": blender.weights.tolist(),
        "bias": blender.bias,
        "platt_a": platt.a,
        "platt_b": platt.b,
        "clip_eps": platt.clip_eps,
        "shrink": float(args.shrink),
        "prior": prior,
        "oof_logloss": rp["logloss"],
        "oof_logloss_nested": rn["logloss"],
        "oof_auroc": rp["auroc"],
    }
    (out / "blend.json").write_text(json.dumps(recipe, indent=2))
    merged.assign(blend=p_final).to_csv(out / "oof_blend.csv", index=False)
    print(f"\necrit dans {out / 'blend.json'}")


if __name__ == "__main__":
    main()
