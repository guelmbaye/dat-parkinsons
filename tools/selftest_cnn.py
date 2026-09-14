"""Auto-test du code reseau. A lancer UNE fois sur la machine GPU, avant tout
entrainement long :

    python tools/selftest_cnn.py

Ce fichier existe parce que le code CNN n'a pas pu etre execute sur la machine
ou il a ete ecrit (pas de GPU, torch non installable). Il verifie les points ou
une erreur est silencieuse et couteuse : convention d'axes de ``affine_grid``,
formes de sortie, effet reel de l'augmentation, capacite a sur-apprendre un
mini-lot, aller-retour EMA. Trente secondes ici valent mieux que six heures
d'entrainement sur un bug d'axe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.imaging import crop_shape                                    # noqa: E402
from datscan.models_cnn import (AugConfig, EMA, augment, build_model,     # noqa: E402
                                predict_logits, to_input, to_projections)

OK, KO = "  OK  ", " ECHEC"
fails = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global fails
    fails += (not cond)
    print(f"[{OK if cond else KO}] {name}{(' - ' + detail) if detail else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = automatique (2 sur CPU, 8 sur GPU)")
    ap.add_argument("--skip-proj2d", action="store_true",
                    help="saute l'encodeur 2D pre-entraine (RAM limitee)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = crop_shape()
    # Le gros du cout memoire vient du premier etage, a pleine resolution :
    # environ 110 Mo d'activations par exemple. Sur une machine a 1 Go libre,
    # un lot de 8 suffit a declencher l'OOM killer.
    bs = args.batch or (8 if dev.type == "cuda" else 2)
    print(f"torch {torch.__version__} | {dev} | volume {shape} | lot {bs}\n")

    # -- 1. Mise en forme -----------------------------------------------------
    v = torch.rand(bs, *shape, device=dev) * 6.0
    check("to_input", tuple(to_input(v).shape) == (bs, 1) + shape,
          str(tuple(to_input(v).shape)))
    proj = to_projections(v, 224)
    check("to_projections", tuple(proj.shape) == (bs, 3, 224, 224),
          str(tuple(proj.shape)))

    # -- 2. Axes de l'augmentation -------------------------------------------
    # Barre orientee selon +Y, decalee en Z. Une rotation axiale de 90 degres
    # doit la faire basculer sur l'axe X en laissant le profil Z intact.
    # Si la rotation s'appliquait au mauvais plan (piege classique de
    # affine_grid en 5D, dont les colonnes sont ordonnees W,H,D), c'est le
    # profil Z qui bougerait.
    nx, ny, nz = shape
    bar = torch.ones(1, *shape, device=dev)
    bar[0, nx // 2 - 1:nx // 2 + 1, ny // 4:3 * ny // 4, nz // 2] = 9.0
    cfg = AugConfig(flip_lr=0.0, rot_deg=0.0, scale=0.0, shift_vox=0.0,
                    intensity_scale=0.0, gamma=0.0, noise=0.0, cutout_p=0.0)
    same = augment(bar.clone(), cfg)
    check("augment neutre = identite", torch.allclose(same, bar, atol=1e-4),
          f"ecart max {float((same - bar).abs().max()):.4f}")

    theta = torch.zeros(1, 3, 4, device=dev)
    theta[0, 0, 1] = -1.0   # X <- -Y
    theta[0, 1, 0] = 1.0    # Y <-  X
    theta[0, 2, 2] = 1.0
    vv = bar.permute(0, 3, 2, 1).unsqueeze(1)
    grid = F.affine_grid(theta, list(vv.shape), align_corners=False)
    rot = F.grid_sample(vv, grid, align_corners=False).squeeze(1) \
           .permute(0, 3, 2, 1)
    zp_before = (bar[0] - 1).clamp(min=0).sum(dim=(0, 1))
    zp_after = (rot[0] - 1).clamp(min=0).sum(dim=(0, 1))
    xe_before = float((bar[0] - 1).clamp(min=0).sum(dim=(1, 2)).nonzero().numel())
    xe_after = float((rot[0] - 1).clamp(min=0).sum(dim=(1, 2)).nonzero().numel())
    check("rotation dans le plan axial (profil Z inchange)",
          bool(torch.allclose(zp_before / zp_before.sum(),
                              zp_after / zp_after.sum(), atol=0.02)),
          "sinon la rotation touche le mauvais plan")
    check("rotation 90 deg : la barre change d'axe", xe_after > xe_before * 3,
          f"etendue X {xe_before:.0f} -> {xe_after:.0f}")

    aug = augment(bar.repeat(bs, 1, 1, 1), AugConfig())
    check("augment complet : formes conservees", aug.shape == (bs,) + shape)
    check("augment complet : pas de NaN", bool(torch.isfinite(aug).all()))
    check("augment complet : effet non nul",
          float((aug[0] - bar[0]).abs().max()) > 1e-3)

    # -- 3. Passe avant / arriere --------------------------------------------
    archs = [("resnet3d", {})]
    if not args.skip_proj2d:
        archs.append(("proj2d", {"backbone": "convnext_tiny", "pretrained": False}))
    for name, kw in archs:
        try:
            model = build_model(name, **kw).to(dev)
            n_par = sum(p.numel() for p in model.parameters()) / 1e6
            x = torch.rand(2, *shape, device=dev) * 6.0
            out = model(x)
            check(f"{name} : sortie (B,)", tuple(out.shape) == (2,),
                  f"{n_par:.1f} M parametres")
            loss = F.binary_cross_entropy_with_logits(
                out, torch.tensor([0.0, 1.0], device=dev))
            loss.backward()
            g = max(float(p.grad.abs().max()) for p in model.parameters()
                    if p.grad is not None)
            check(f"{name} : gradients non nuls et finis",
                  np.isfinite(g) and g > 0, f"|grad| max {g:.2e}")
            check(f"{name} : TTA symetrique",
                  tuple(predict_logits(model, x).shape) == (2,))
            del model, out, loss
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            check(f"{name} : construction", False, f"{type(exc).__name__}: {exc}")

    # -- 4. Capacite a sur-apprendre un mini-lot -----------------------------
    torch.manual_seed(0)
    model = build_model("resnet3d").to(dev)
    n_ov = max(2, bs)
    x = torch.rand(n_ov, *shape, device=dev)
    y = torch.tensor([0.0, 1.0] * (n_ov // 2), device=dev)
    x[y > 0.5] += 3.0
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    n_iter = 40 if dev.type == "cuda" else 25
    first = last = None
    for it in range(n_iter):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        first = last if it == 0 else first
    check("resnet3d sur-apprend un mini-lot", last < 0.5 * first,
          f"{first:.3f} -> {last:.3f}")

    # -- 5. EMA ---------------------------------------------------------------
    ema = EMA(model, decay=0.9)
    ema.update(model)
    sd = ema.state_dict(model)
    check("EMA : cles completes", set(sd) == set(model.state_dict()))
    model.load_state_dict(sd)
    check("EMA : rechargement", True)

    print(f"\n{'TOUT EST VERT' if fails == 0 else str(fails) + ' VERIFICATION(S) EN ECHEC'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
