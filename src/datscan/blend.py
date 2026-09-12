"""Fusion des modeles et calibration des probabilites.

La metrique du challenge est la log loss : le classement recompense autant la
calibration que le pouvoir discriminant. Un modele qui separe parfaitement mais
sort des 0.99 quand il devrait sortir des 0.85 perd des places.

Trois etages, tous ajustes sur les predictions hors-pli (jamais sur le train
in-fold, sinon la calibration est optimiste et se retourne sur le test) :

1. fusion lineaire dans l'espace des logits, poids contraints positifs ;
2. mise a l'echelle de Platt (temperature + biais), qui corrige la
   sur-confiance typique des CNN ;
3. ecretage symetrique, dont la borne est choisie sur la meme base hors-pli.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

EPS = 1e-7


def to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def to_prob(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -40, 40)))


def _softplus(x):
    return np.logaddexp(0.0, x)


@dataclass
class LogitBlender:
    """Fusion a poids positifs + biais, optimisee directement en log loss."""
    l2: float = 1e-3
    names: list = field(default_factory=list)
    raw_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0
    clip_eps: float = 1e-4

    @property
    def weights(self) -> np.ndarray:
        w = _softplus(self.raw_w)
        s = w.sum()
        return w / s if s > 0 else np.full_like(w, 1.0 / max(len(w), 1))

    def fit(self, logits: np.ndarray, y: np.ndarray,
            names: list | None = None) -> "LogitBlender":
        """``logits`` : (n, k) logits hors-pli, une colonne par modele."""
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, k = z.shape
        self.names = list(names) if names else [f"m{i}" for i in range(k)]

        def obj(theta):
            raw, b = theta[:k], theta[k]
            w = _softplus(raw)
            s = w.sum()
            if s <= 0:
                return 1e9, np.zeros_like(theta)
            wn = w / s
            f = z @ wn + b
            p = to_prob(f)
            loss = -np.mean(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS))
            loss += self.l2 * float(np.sum(raw ** 2))
            g_f = (p - y) / n
            g_wn = z.T @ g_f
            # d(w/s)/d(w_j) puis d(w)/d(raw) = sigmoid(raw)
            g_w = (g_wn - float(g_wn @ wn)) / s
            grad_raw = g_w * to_prob(raw) + 2 * self.l2 * raw
            return loss, np.concatenate([grad_raw, [float(g_f.sum())]])

        theta0 = np.concatenate([np.zeros(k), [0.0]])
        res = minimize(obj, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.raw_w, self.bias = res.x[:k], float(res.x[k])
        self.clip_eps = _best_clip(self.predict(z, apply_clip=False), y)
        return self

    def predict(self, logits: np.ndarray, apply_clip: bool = True) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        p = to_prob(z @ self.weights + self.bias)
        return np.clip(p, self.clip_eps, 1 - self.clip_eps) if apply_clip else p

    def describe(self) -> str:
        parts = [f"{n}={w:.3f}" for n, w in zip(self.names, self.weights)]
        return (f"poids [{', '.join(parts)}]  biais={self.bias:+.3f}  "
                f"ecretage={self.clip_eps:.4f}")


@dataclass
class PlattScaler:
    """p' = sigmoid(a * logit(p) + b). Corrige la sur-confiance d'un modele."""
    a: float = 1.0
    b: float = 0.0
    clip_eps: float = 1e-4

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattScaler":
        z = to_logit(p)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        def obj(theta):
            a, b = theta
            f = a * z + b
            q = to_prob(f)
            loss = -np.mean(y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS))
            g = (q - y) / n
            return loss, np.array([float(g @ z), float(g.sum())])

        res = minimize(obj, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                       bounds=[(0.05, 5.0), (-5.0, 5.0)])
        self.a, self.b = float(res.x[0]), float(res.x[1])
        self.clip_eps = _best_clip(self.transform(p, apply_clip=False), y)
        return self

    def transform(self, p: np.ndarray, apply_clip: bool = True) -> np.ndarray:
        q = to_prob(self.a * to_logit(p) + self.b)
        return np.clip(q, self.clip_eps, 1 - self.clip_eps) if apply_clip else q


def _best_clip(p: np.ndarray, y: np.ndarray, tol: float = 0.01) -> float:
    """Borne d'ecretage, choisie prudemment sur les predictions hors-pli.

    Une seule prediction tres confiante et fausse coute jusqu'a 14 nats : sur
    1000 examens de test, cela ajoute 0.014 a la log loss moyenne, soit
    largement de quoi perdre des places. L'optimum empirique n'est donc pas le
    bon choix, parce qu'il est estime sur un echantillon fini et que le test
    prive contiendra des cas plus durs que la validation.

    Deux garde-fous : un plancher lie a l'effectif (regle de type Laplace,
    ~1/4n) et, a performance quasi egale (``tol`` en relatif), on retient
    l'ecretage le PLUS large. C'est une assurance qui coute quelques millièmes
    en validation et evite une catastrophe sur le test.
    """
    grid = np.array([1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 6e-3, 1e-2, 2e-2, 3e-2, 5e-2])
    y = np.asarray(y, dtype=int)
    floor = max(1e-4, 1.0 / (4.0 * max(len(y), 1)))
    grid = grid[grid >= floor]
    if grid.size == 0:
        return floor
    losses = np.array([log_loss(y, np.clip(p, e, 1 - e), labels=[0, 1])
                       for e in grid])
    best = float(losses.min())
    ok = grid[losses <= best * (1.0 + tol)]
    return float(ok.max())


def shrink_to_prior(p: np.ndarray, prior: float, alpha: float) -> np.ndarray:
    """Retrecissement vers la prevalence d'entrainement, dans l'espace logit.

    Assurance contre un decalage de prevalence entre train et test prive.
    ``alpha=0`` ne change rien ; 0.05-0.15 coute quelques millièmes de log loss
    en validation et protege d'une derive de plusieurs centiemes.
    """
    if alpha <= 0:
        return p
    z = to_logit(p)
    z0 = to_logit(np.array([prior]))[0]
    return to_prob((1 - alpha) * z + alpha * z0)
