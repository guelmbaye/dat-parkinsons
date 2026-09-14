"""Etape 5 : assemblage du ``submission.zip``.

    python scripts/05_pack.py --tab models/tab --blend models/blend \\
        --cnn resnet3d=models/resnet3d --out dist

Copie le module ``datscan`` (le meme code de pretraitement qu'a
l'entrainement), les poids et la recette de fusion, puis produit une archive
avec ``main.py`` A LA RACINE — c'est la seule contrainte stricte du format de
soumission, et l'erreur classique est de zipper le dossier parent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEEP = ["__init__.py", "imaging.py", "features.py", "models_cnn.py", "blend.py"]


def _verify_cnns(stage: Path) -> None:
    """Charge reellement chaque checkpoint dans le modele reconstruit.

    Sans ce controle, une incoherence d'architecture entre entrainement et
    inference ne se revele que dans le conteneur d'evaluation : main.py
    degrade en silence sur les modeles restants et sort en code 0. On croit
    avoir soumis son meilleur modele, et on a soumis l'ancien.
    """
    root = stage / "assets" / "cnn"
    if not root.is_dir():
        return
    sys.path.insert(0, str(stage))
    try:
        import torch
        from datscan.models_cnn import build_model
        from main import _model_kwargs
    except Exception as exc:
        print(f"  VERIFICATION IMPOSSIBLE ({type(exc).__name__}) — "
              f"empaqueter sur une machine avec torch")
        return
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        cfg = json.loads((d / "config.json").read_text())
        for fp in sorted(d.glob("fold*.pt")):
            state = torch.load(fp, map_location="cpu")
            model = build_model(cfg.get("model", "resnet3d"),
                                **_model_kwargs(cfg, state))
            model.load_state_dict(state)          # leve si incoherent
        print(f"  verifie : {d.name} ({len(list(d.glob('fold*.pt')))} plis "
              f"se chargent)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", default=None, help="dossier des modeles tabulaires")
    ap.add_argument("--blend", required=True, help="dossier contenant blend.json")
    ap.add_argument("--cnn", action="append", default=[], metavar="NOM=DOSSIER")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()

    out = Path(args.out)
    stage = out / "submission_src"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "assets").mkdir(parents=True)

    shutil.copy(REPO / "submission_src" / "main.py", stage / "main.py")
    pkg = stage / "datscan"
    pkg.mkdir()
    for name in KEEP:
        shutil.copy(REPO / "src" / "datscan" / name, pkg / name)

    shutil.copy(Path(args.blend) / "blend.json", stage / "assets" / "blend.json")
    if args.tab:
        for name in ("gbm.pkl", "linear.pkl"):
            src = Path(args.tab) / name
            if src.exists():
                shutil.copy(src, stage / "assets" / name)

    for spec in args.cnn:
        name, path = spec.split("=", 1)
        dst = stage / "assets" / "cnn" / name
        dst.mkdir(parents=True)
        src = Path(path)
        for fp in sorted(src.glob("fold*.pt")):
            shutil.copy(fp, dst / fp.name)
        shutil.copy(src / "config.json", dst / "config.json")

    _verify_cnns(stage)

    zip_path = out / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for fp in sorted(stage.rglob("*")):
            if fp.is_file():
                z.write(fp, fp.relative_to(stage))

    names = zipfile.ZipFile(zip_path).namelist()
    assert "main.py" in names, "main.py doit etre a la racine de l'archive"
    recipe = json.loads((stage / "assets" / "blend.json").read_text())
    size = zip_path.stat().st_size / 1e6
    print(f"{zip_path}  ({size:.1f} Mo, {len(names)} fichiers)")
    print(f"  modeles de la recette : {recipe.get('models')}")
    print(f"  log loss hors-pli (croisee) : "
          f"{recipe.get('oof_logloss_nested', recipe.get('oof_logloss')):.4f}")
    if size > 900:
        print("  ATTENTION archive volumineuse : verifier la limite d'upload")


if __name__ == "__main__":
    main()
