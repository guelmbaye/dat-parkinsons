"""Etape 3 : entrainement des reseaux convolutifs en validation croisee.

    python scripts/03_train_cnn.py --cache cache --out models/resnet3d \\
        --model resnet3d --epochs 60 --folds 5

Produit, dans ``--out`` :
    fold{k}.pt      poids EMA du pli k
    oof.csv         uid, probabilite hors-pli
    curve.csv       log loss de validation moyenne par epoque
    config.json     tout ce qu'il faut pour rejouer l'inference

Choix : pas de selection d'epoque sur le pli de validation. On entraine un
nombre d'epoques fixe avec un cosinus et on garde les poids EMA finaux. Les
predictions hors-pli sont ainsi strictement non biaisees, ce qui compte
enormement ici puisque la calibration sera ajustee dessus. La colonne
``curve.csv`` sert a regler ``--epochs`` une fois pour toutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.models_cnn import (AugConfig, EMA, augment, build_model,  # noqa: E402
                                predict_logits, resolve_amp)
from datscan.cv import make_folds, report, per_group_report              # noqa: E402


def batches(idx: np.ndarray, bs: int, shuffle: bool, rng=None):
    order = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


@torch.no_grad()
def evaluate(model, vols, idx, device, bs, amp, tta=True) -> np.ndarray:
    enabled, dtype = amp
    out = np.zeros(len(idx), dtype=np.float64)
    for s in range(0, len(idx), bs):
        sl = idx[s:s + bs]
        x = torch.from_numpy(np.asarray(vols[sl], dtype=np.float32)).to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=enabled):
            z = predict_logits(model, x, tta_flip=tta)
        out[s:s + len(sl)] = z.float().cpu().numpy()
    return out


def train_fold(vols, y, tr, va, args, device, amp, log) -> tuple[np.ndarray, dict, list]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.model == "resnet3d":
        kw = {"widths": tuple(int(w) for w in args.widths.split(",")),
              "drop": args.dropout, "head_drop": args.head_dropout}
    else:
        kw = {"backbone": args.backbone, "pretrained": bool(args.pretrained),
              "head_drop": args.head_dropout}
    model = build_model(args.model, **kw).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = max(1, int(np.ceil(len(tr) / args.batch_size))) * args.epochs
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    ema = EMA(model, decay=args.ema)
    enabled, dtype = amp
    scaler = torch.amp.GradScaler("cuda", enabled=(enabled and dtype is torch.float16))
    cfg = AugConfig(rot_deg=args.rot_deg, scale=args.scale,
                    shift_vox=args.shift_vox, noise=args.noise)
    rng = np.random.default_rng(args.seed)

    curve = []
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for sl in batches(tr, args.batch_size, True, rng):
            x = torch.from_numpy(np.asarray(vols[sl], dtype=np.float32)).to(device)
            t = torch.from_numpy(y[sl].astype(np.float32)).to(device)
            x = augment(x, cfg)
            with torch.autocast("cuda", dtype=dtype, enabled=enabled):
                loss = F.binary_cross_entropy_with_logits(model(x), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += float(loss) * len(sl)
            n += len(sl)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict(model))
            z = evaluate(model, vols, va, device, args.eval_batch_size, amp)
            model.load_state_dict(snapshot)
            r = report(y[va], 1 / (1 + np.exp(-z)))
            curve.append({"epoch": ep + 1, "train_loss": tot / max(n, 1),
                          "val_logloss": r["logloss"], "val_auroc": r["auroc"]})
            log(f"    ep {ep + 1:3d}/{args.epochs}  train {tot / max(n, 1):.4f}  "
                f"val {r['logloss']:.4f}  auroc {r['auroc']:.4f}")

    model.load_state_dict(ema.state_dict(model))
    z = evaluate(model, vols, va, device, args.eval_batch_size, amp)
    return z, {k: v.detach().cpu() for k, v in model.state_dict().items()}, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="resnet3d", choices=["resnet3d", "proj2d"])
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--rot-deg", type=float, default=8.0)
    ap.add_argument("--scale", type=float, default=0.08)
    ap.add_argument("--shift-vox", type=float, default=3.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--group-cv", type=int, default=1,
                    help="1 = plis par pseudo-centre (recommande)")
    ap.add_argument("--channels-last", type=int, default=0)
    ap.add_argument("--widths", default="24,48,96,160",
                    help="largeurs du ResNet3D ; sur un petit jeu, essayer 16,32,64,96")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head-dropout", type=float, default=0.3)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"],
                    help="auto = bf16 si le GPU le supporte, fp16 sinon (T4)")
    ap.add_argument("--resume", type=int, default=1,
                    help="reprend les plis deja termines (utile sur Colab)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    man = pd.read_csv(cache / "manifest.csv")
    man = man[man["is_pathologic"].notna()].reset_index(drop=True)
    vols = np.load(cache / "volumes.npy", mmap_mode="r")[man.index.to_numpy()]
    vols = np.ascontiguousarray(vols)
    y = man["is_pathologic"].to_numpy(dtype=np.float32)
    groups = man["pseudo_center"].to_numpy() if args.group_cv else None
    folds = make_folds(y.astype(int), groups, args.folds, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = resolve_amp(device, args.amp)
    log = print
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    log(f"{len(y)} examens | prevalence {y.mean():.3f} | {args.folds} plis "
        f"({'groupes' if args.group_cv else 'aleatoires'})")
    log(f"{gpu} | precision mixte : "
        f"{str(amp[1]).replace('torch.', '') if amp[0] else 'desactivee'}")

    oof = np.zeros(len(y), dtype=np.float64)
    curves = []
    t0 = time.time()
    for k in range(args.folds):
        tr = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        done = out / f"fold{k}.pt"
        cached = out / f"oof_fold{k}.csv"
        if args.resume and done.exists() and cached.exists():
            oof[va] = pd.read_csv(cached)["pred"].to_numpy()
            log(f"  pli {k}: deja entraine, repris depuis {done.name}")
            continue
        log(f"  pli {k}: train {len(tr)} / val {len(va)}")
        z, state, curve = train_fold(vols, y, tr, va, args, device, amp, log)
        oof[va] = 1 / (1 + np.exp(-z))
        torch.save(state, done)
        pd.DataFrame({"uid": man["uid"].to_numpy()[va],
                      "pred": oof[va]}).to_csv(cached, index=False)
        for c in curve:
            c["fold"] = k
        curves += curve

    r = report(y.astype(int), oof)
    log(f"\nhors-pli : logloss {r['logloss']:.4f} | auroc {r['auroc']:.4f} | "
        f"reference constante {r['logloss_baseline']:.4f} "
        f"({time.time() - t0:.0f}s)")
    if groups is not None:
        pg = per_group_report(y.astype(int), oof, groups)
        if len(pg):
            log("\npar pseudo-centre (pires en tete) :")
            log(pg[["group", "n", "logloss", "auroc"]].head(8)
                .round(4).to_string(index=False))

    pd.DataFrame({"uid": man["uid"], "fold": folds, "y": y,
                  "pred": oof}).to_csv(out / "oof.csv", index=False)
    cur = pd.DataFrame(curves)
    if len(cur):
        cur.to_csv(out / "curve.csv", index=False)
        log("\nlog loss de validation moyenne par epoque "
            "(pour regler --epochs) :")
        log(cur.groupby("epoch")["val_logloss"].mean().round(4).to_string())
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
