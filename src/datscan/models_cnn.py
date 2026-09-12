"""Reseaux de neurones sur les volumes pretraites.

Deux architectures complementaires :

* ``ResNet3D`` — convolutions 3D entrainees de zero. Voit la forme complete du
  striatum (le passage "virgule" -> "point" est un phenomene 3D).
* ``Proj2D`` — projections d'intensite maximale par sous-dalles axiales,
  envoyees dans un encodeur 2D pre-entraine ImageNet. C'est la vue que lit le
  medecin, et le transfert ImageNet compense un effectif limite.

Points de methode importants
----------------------------
1. Pas de normalisation par examen : le volume est deja en unites
   "fond = 1.0" et le NIVEAU de fixation est precisement le signal
   diagnostique. Un z-score par image le detruirait.
2. La symetrie gauche-droite est exploitee comme augmentation ET au moment du
   test : le cote atteint est arbitraire, le modele doit etre insensible a
   l'echange.
3. Selection d'epoque commune a tous les plis (et non par pli), sinon les
   predictions hors-pli sont optimistes et la calibration derape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mise en forme des entrees
# ---------------------------------------------------------------------------
def to_input(vol: torch.Tensor) -> torch.Tensor:
    """(B, X, Y, Z) en unites fond=1.0 -> (B, 1, X, Y, Z) centre autour de 0."""
    x = vol.clamp(0.0, 12.0)
    return ((x - 1.0) / 3.0).unsqueeze(1)


def to_projections(vol: torch.Tensor, size: int = 224) -> torch.Tensor:
    """(B, X, Y, Z) -> (B, 3, size, size) : trois sous-dalles axiales en MIP.

    Sous-dalles inferieure / mediane / superieure : on conserve l'information
    de niveau de coupe que perdrait une projection unique, tout en obtenant une
    image RGB directement exploitable par un encodeur ImageNet.
    """
    z = vol.shape[-1]
    cuts = (0, z // 3, 2 * z // 3, z)
    chans = [vol[..., cuts[i]:cuts[i + 1]].amax(dim=-1) for i in range(3)]
    x = torch.stack(chans, dim=1).clamp(0.0, 12.0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - 1.0) / 3.0


# ---------------------------------------------------------------------------
# Augmentation, executee sur GPU par lot
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    flip_lr: float = 0.5
    rot_deg: float = 8.0
    scale: float = 0.08
    shift_vox: float = 3.0
    intensity_scale: float = 0.12
    gamma: float = 0.15
    noise: float = 0.05
    cutout_p: float = 0.25
    cutout_vox: int = 10


def augment(vol: torch.Tensor, cfg: AugConfig) -> torch.Tensor:
    """Augmentation geometrique + photometrique sur un lot (B, X, Y, Z).

    Attention a la convention de ``affine_grid`` : en 5D, les colonnes de theta
    sont ordonnees (W, H, D), soit l'INVERSE de l'ordre des axes du tenseur. On
    permute donc en (B, Z, Y, X) avant l'echantillonnage pour que la rotation
    s'applique bien dans le plan axial X-Y, puis on revient a (B, X, Y, Z).
    """
    b = vol.shape[0]
    dev = vol.device
    nx, ny, nz = vol.shape[1], vol.shape[2], vol.shape[3]

    def rnd(n=b):
        return torch.rand(n, device=dev)

    if cfg.flip_lr > 0:
        m = rnd() < cfg.flip_lr
        if bool(m.any()):
            vol = torch.where(m.view(-1, 1, 1, 1), vol.flip(1), vol)

    if cfg.rot_deg > 0 or cfg.scale > 0 or cfg.shift_vox > 0:
        ang = (rnd() * 2 - 1) * math.radians(cfg.rot_deg)
        inv_s = 1.0 / (1.0 + (rnd() * 2 - 1) * cfg.scale)
        ca, sa = torch.cos(ang) * inv_s, torch.sin(ang) * inv_s
        theta = torch.zeros(b, 3, 4, device=dev, dtype=vol.dtype)
        theta[:, 0, 0], theta[:, 0, 1] = ca, -sa      # colonne 0 <-> axe X
        theta[:, 1, 0], theta[:, 1, 1] = sa, ca       # colonne 1 <-> axe Y
        theta[:, 2, 2] = inv_s                        # colonne 2 <-> axe Z
        for i, n in enumerate((nx, ny, nz)):
            theta[:, i, 3] = (rnd() * 2 - 1) * (2.0 * cfg.shift_vox / n)
        v = vol.permute(0, 3, 2, 1).unsqueeze(1)      # (B, 1, Z, Y, X)
        grid = F.affine_grid(theta, list(v.shape), align_corners=False)
        v = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False)
        vol = v.squeeze(1).permute(0, 3, 2, 1).contiguous()

    spec = (vol - 1.0).clamp(min=0.0)
    if cfg.gamma > 0:
        g = 1.0 + (rnd() * 2 - 1) * cfg.gamma
        spec = spec.clamp(min=1e-4).pow(g.view(-1, 1, 1, 1))
    if cfg.intensity_scale > 0:
        s = 1.0 + (rnd() * 2 - 1) * cfg.intensity_scale
        spec = spec * s.view(-1, 1, 1, 1)
    vol = 1.0 + spec

    if cfg.noise > 0:
        vol = vol + torch.randn_like(vol) * (rnd().view(-1, 1, 1, 1) * cfg.noise)

    if cfg.cutout_p > 0 and cfg.cutout_vox > 0:
        k = cfg.cutout_vox
        # Tirages sur CPU (numpy) : evite une synchronisation GPU par exemple.
        hits = np.random.rand(b) < cfg.cutout_p
        for i in np.nonzero(hits)[0]:
            ox, oy, oz = (np.random.randint(0, max(1, n - k))
                          for n in (nx, ny, nz))
            vol[int(i), ox:ox + k, oy:oy + k, oz:oz + k] = 1.0

    return vol.clamp(0.0, 20.0)


# ---------------------------------------------------------------------------
# Architecture 3D
# ---------------------------------------------------------------------------
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c), c)


class Block3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _gn(cout)
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False),
                                   _gn(cout)))

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.drop(self.n2(self.c2(h)))
        return F.silu(h + self.skip(x))


class ResNet3D(nn.Module):
    """ResNet 3D compact (~4 M parametres), normalisation par groupes.

    GroupNorm plutot que BatchNorm : les lots sont petits en 3D et la
    BatchNorm couple les exemples, ce qui la rend instable ici.
    """

    def __init__(self, widths=(24, 48, 96, 160), drop: float = 0.1,
                 head_drop: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, 1, 1, bias=False),
                                  _gn(widths[0]), nn.SiLU())
        stages, cin = [], widths[0]
        for i, w in enumerate(widths):
            stages.append(Block3D(cin, w, stride=1 if i == 0 else 2, drop=drop))
            stages.append(Block3D(w, w, stride=1, drop=drop))
            cin = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear(cin * 2, 1))

    def forward(self, vol):
        h = self.stages(self.stem(to_input(vol)))
        pooled = torch.cat([h.mean(dim=(2, 3, 4)),
                            h.amax(dim=(2, 3, 4))], dim=1)
        return self.head(pooled).squeeze(1)


# ---------------------------------------------------------------------------
# Architecture 2.5D sur encodeur pre-entraine
# ---------------------------------------------------------------------------
class Proj2D(nn.Module):
    """Projections axiales -> encodeur timm pre-entraine ImageNet.

    ``pretrained`` doit valoir False dans le conteneur d'evaluation (aucun
    acces reseau) : les poids arrivent via le state_dict embarque dans le zip.
    """

    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = False,
                 size: int = 224, head_drop: float = 0.3):
        super().__init__()
        import timm
        self.size = size
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                         num_classes=0, in_chans=3)
        nf = self.encoder.num_features
        self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear(nf, 1))

    def forward(self, vol):
        return self.head(self.encoder(to_projections(vol, self.size))).squeeze(1)


def build_model(name: str, **kw) -> nn.Module:
    if name == "resnet3d":
        return ResNet3D(**kw)
    if name == "proj2d":
        return Proj2D(**kw)
    raise ValueError(f"modele inconnu : {name}")


def resolve_amp(device, choice: str = "auto") -> tuple[bool, "torch.dtype"]:
    """Choisit le type de precision mixte selon le GPU disponible.

    Important pour Colab : les T4 (Turing) n'ont pas de bfloat16 materiel. Y
    forcer bf16 donne une emulation lente, voire une erreur. Les A100, L4 et
    plus recents preferent bf16, qui evite le sur/sous-debordement du fp16 et
    se passe de GradScaler.
    """
    if device.type != "cuda" or choice == "off":
        return False, torch.float32
    if choice == "bf16":
        return True, torch.bfloat16
    if choice == "fp16":
        return True, torch.float16
    supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return True, (torch.bfloat16 if supported else torch.float16)


# ---------------------------------------------------------------------------
# Moyenne mobile exponentielle des poids
# ---------------------------------------------------------------------------
class EMA:
    """Lisse les poids sur les dernieres iterations : gain systematique en
    log loss, pour un cout nul a l'inference."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def state_dict(self, model: nn.Module) -> dict:
        out = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            out[k] = v.to(out[k].dtype)
        return out


# ---------------------------------------------------------------------------
# Inference avec augmentation au test
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model: nn.Module, vols: torch.Tensor, tta_flip: bool = True
                   ) -> torch.Tensor:
    """Logit moyen sur l'image et son miroir gauche-droite."""
    model.eval()
    out = model(vols)
    if tta_flip:
        out = 0.5 * (out + model(vols.flip(1)))
    return out
