"""Point d'entree d'inference — DaT Parkinson's Challenge (SFMN / DrivenData).

Execute dans le conteneur d'evaluation : lit ``/code_execution/data/niftis/``,
ecrit ``submission.csv`` a la racine du repertoire de travail.

Trois principes gouvernent ce fichier :

1. **Une soumission sort toujours.** Chaque etage est protege. Si le CNN tombe,
   on rend les modeles sur descripteurs ; si tout tombe, on rend la prevalence
   d'entrainement. Une soumission mediocre vaut infiniment mieux qu'un job en
   erreur, surtout a quelques jours de la cloture.
2. **Le budget temps est surveille.** La limite est de 3 h. Les etages couteux
   sont abandonnes avant de la depasser, jamais apres.
3. **Le pretraitement est importe du meme module qu'a l'entrainement.** Aucune
   reimplementation, donc aucune divergence train/test possible.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from datscan.imaging import preprocess, crop_shape  # noqa: E402
from datscan.features import extract, feature_names  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "/code_execution/data"))
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT = DATA_DIR / "submission_format.csv"
OUT_PATH = Path("submission.csv")
ASSETS = ROOT / "assets"

START = time.time()
BUDGET_S = float(os.environ.get("BUDGET_S", 3 * 3600 * 0.80))
IS_SMOKE = bool(os.environ.get("IS_SMOKE_TEST"))


def log(msg: str) -> None:
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def remaining() -> float:
    return BUDGET_S - (time.time() - START)


# ---------------------------------------------------------------------------
def _one(uid: str):
    vol, qc = preprocess(NIFTI_DIR / f"{uid}.nii.gz")
    return uid, vol.astype(np.float16), extract(vol), qc.ok


def run_preprocessing(uids: list[str], workers: int):
    shape = crop_shape()
    vols = np.zeros((len(uids),) + shape, dtype=np.float16)
    feats: list[dict] = [None] * len(uids)  # type: ignore[list-item]
    pos = {u: i for i, u in enumerate(uids)}
    n_fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (uid, vol, f, ok) in enumerate(ex.map(_one, uids, chunksize=4), 1):
            i = pos[uid]
            vols[i] = vol
            feats[i] = f
            n_fail += (not ok)
            if n % 250 == 0:
                el = time.time() - t0
                log(f"  pretraitement {n}/{len(uids)} "
                    f"({el / n:.2f}s/examen, reste ~{el / n * (len(uids) - n):.0f}s)")
    log(f"pretraitement termine en {time.time() - t0:.0f}s"
        f"{' — ATTENTION : des fichiers ont echoue' if n_fail else ''}")
    return vols, pd.DataFrame(feats), n_fail


# ---------------------------------------------------------------------------
def predict_tabular(X: pd.DataFrame) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    try:
        import lightgbm as lgb
        with (ASSETS / "gbm.pkl").open("rb") as f:
            bundle = pickle.load(f)
        Xg = X.reindex(columns=bundle["columns"], fill_value=0.0)
        Xv = Xg.to_numpy(dtype=np.float64)
        boosters = [lgb.Booster(model_str=s) for s in bundle["models"]]
        preds["gbm"] = np.mean([b.predict(Xv) for b in boosters], axis=0)
        log(f"gbm : {len(boosters)} plis")
    except Exception as exc:
        log(f"gbm indisponible ({type(exc).__name__}: {exc})")
    try:
        with (ASSETS / "linear.pkl").open("rb") as f:
            bundle = pickle.load(f)
        Xl = X.reindex(columns=bundle["columns"], fill_value=0.0)
        preds["linear"] = np.mean([m.predict_proba(Xl)[:, 1]
                                   for m in bundle["models"]], axis=0)
        log(f"linear : {len(bundle['models'])} plis")
    except Exception as exc:
        log(f"linear indisponible ({type(exc).__name__}: {exc})")
    return preds


def predict_cnns(vols: np.ndarray) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    cnn_root = ASSETS / "cnn"
    if not cnn_root.is_dir():
        return preds
    try:
        import torch
        from datscan.models_cnn import build_model, predict_logits, resolve_amp
    except Exception as exc:
        log(f"torch indisponible ({type(exc).__name__}: {exc})")
        return preds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on, amp_dtype = resolve_amp(device, "auto")
    log(f"cnn sur {device} (precision mixte : "
        f"{str(amp_dtype).replace('torch.', '') if amp_on else 'desactivee'})")
    for d in sorted(p for p in cnn_root.iterdir() if p.is_dir()):
        folds = sorted(d.glob("fold*.pt"))
        if not folds:
            continue
        # Marge : on n'engage un modele que si le temps restant le permet.
        need = 60.0 + 0.02 * len(vols) * len(folds)
        if remaining() < need:
            log(f"{d.name} ignore : {remaining():.0f}s restantes < {need:.0f}s requises")
            continue
        try:
            cfg = json.loads((d / "config.json").read_text())
            kw = {} if cfg.get("model") == "resnet3d" else \
                {"backbone": cfg.get("backbone", "convnext_tiny"), "pretrained": False}
            acc, t0 = [], time.time()
            for fp in folds:
                model = build_model(cfg.get("model", "resnet3d"), **kw).to(device)
                model.load_state_dict(torch.load(fp, map_location=device))
                model.eval()
                bs = int(cfg.get("eval_batch_size", 32))
                z = np.zeros(len(vols), dtype=np.float64)
                with torch.no_grad():
                    for s in range(0, len(vols), bs):
                        x = torch.from_numpy(
                            np.asarray(vols[s:s + bs], dtype=np.float32)).to(device)
                        with torch.autocast("cuda", dtype=amp_dtype,
                                            enabled=amp_on):
                            out = predict_logits(model, x, tta_flip=True)
                        z[s:s + x.shape[0]] = out.float().cpu().numpy()
                acc.append(z)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            zm = np.mean(acc, axis=0)
            preds[d.name] = 1.0 / (1.0 + np.exp(-np.clip(zm, -40, 40)))
            log(f"{d.name} : {len(folds)} plis en {time.time() - t0:.0f}s")
        except Exception as exc:
            log(f"{d.name} echoue ({type(exc).__name__}: {exc})")
    return preds


# ---------------------------------------------------------------------------
def combine(preds: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Applique la recette de fusion, en se limitant aux modeles disponibles."""
    try:
        recipe = json.loads((ASSETS / "blend.json").read_text())
    except Exception:
        recipe = {}
    prior = float(recipe.get("prior", 0.5))
    if not preds:
        log("aucun modele disponible : sortie a la prevalence d'entrainement")
        return np.full(n, prior)

    names = [m for m in recipe.get("models", []) if m in preds]
    weights = [w for m, w in zip(recipe.get("models", []),
                                 recipe.get("weights", [])) if m in preds]
    if not names:  # recette absente ou modeles tous inconnus
        names, weights = list(preds), [1.0] * len(preds)
        log("recette inapplicable : moyenne simple des modeles disponibles")
    missing = set(recipe.get("models", [])) - set(names)
    if missing:
        log(f"modeles manquants, poids renormalises : {sorted(missing)}")

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1.0 / len(names))
    eps = 1e-6
    z = np.zeros(n)
    for name, wi in zip(names, w):
        p = np.clip(np.asarray(preds[name], dtype=np.float64), eps, 1 - eps)
        z += wi * np.log(p / (1 - p))
    z += float(recipe.get("bias", 0.0))
    z = float(recipe.get("platt_a", 1.0)) * z + float(recipe.get("platt_b", 0.0))
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    alpha = float(recipe.get("shrink", 0.0))
    if alpha > 0:
        z0 = np.log(max(prior, eps) / max(1 - prior, eps))
        p = 1.0 / (1.0 + np.exp(-np.clip((1 - alpha) * np.log(
            np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
            + alpha * z0, -40, 40)))

    clip = float(recipe.get("clip_eps", 1e-3))
    return np.clip(p, clip, 1.0 - clip)


def main() -> None:
    sub = pd.read_csv(SUBMISSION_FORMAT)
    uids = sub["uid"].astype(str).tolist()
    log(f"{len(uids)} examens a predire | budget {BUDGET_S / 60:.0f} min"
        f"{' | smoke test' if IS_SMOKE else ''}")

    workers = max(1, min(int(os.environ.get("N_WORKERS", 0)) or (os.cpu_count() or 4),
                         os.cpu_count() or 4))
    preds: dict[str, np.ndarray] = {}
    try:
        vols, feats, _ = run_preprocessing(uids, workers)
        X = feats.reindex(columns=feature_names(), fill_value=0.0) \
                 .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        preds.update(predict_tabular(X))
        preds.update(predict_cnns(vols))
    except Exception as exc:
        log(f"ECHEC du pipeline principal ({type(exc).__name__}: {exc})")

    sub["is_pathologic"] = combine(preds, len(uids))
    sub[["uid", "is_pathologic"]].to_csv(OUT_PATH, index=False)
    # NE JAMAIS journaliser de statistique derivee des predictions ou des
    # images de test (moyenne, min, max, distribution...). Le reglement
    # l'interdit explicitement et en fait un motif de disqualification. Seuls
    # l'avancement et les diagnostics du code sont autorises.
    log(f"submission.csv ecrit : {len(sub)} lignes | "
        f"modeles utilises : {sorted(preds) or 'aucun'}")


if __name__ == "__main__":
    main()
