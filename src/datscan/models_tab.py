"""Modeles sur descripteurs semi-quantitatifs : gradient boosting et logistique.

Deux familles volontairement differentes :

* LightGBM capte les seuils et les interactions (p. ex. "SBR bas ET asymetrie
  forte"), ce que fait implicitement un lecteur experimente ;
* la regression logistique regularisee est lineaire, tres bien calibree et
  beaucoup plus stable si le test prive contient un scanner inconnu.

Les deux fournissent des predictions hors-pli honnetes : le nombre d'arbres est
fige AVANT le calcul des scores definitifs (voir ``train_gbm_cv``), sinon
l'arret precoce lit le pli de validation et la log loss hors-pli est optimiste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.02,
    num_leaves=15,
    min_data_in_leaf=25,
    feature_fraction=0.55,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l1=0.5,
    lambda_l2=5.0,
    max_depth=5,
    verbose=-1,
    num_threads=0,
)
MAX_ROUNDS = 4000


def train_gbm_cv(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                 params: dict | None = None, seed: int = 2026,
                 log=print) -> tuple[np.ndarray, list, dict]:
    """Retourne (predictions hors-pli, modeles par pli, info).

    Deux passes : la premiere cherche le nombre d'arbres par arret precoce, la
    seconde reentraine chaque pli avec un nombre fige (mediane des passes 1).
    Les scores hors-pli de la seconde passe sont donc non biaises.
    """
    import lightgbm as lgb

    p = dict(GBM_PARAMS)
    if params:
        p.update(params)
    p["seed"] = seed
    Xv = X.to_numpy(dtype=np.float64)
    y = np.asarray(y, dtype=int)
    ks = sorted(np.unique(folds[folds >= 0]))

    best_iters = []
    for k in ks:
        tr, va = folds != k, folds == k
        ds_tr = lgb.Dataset(Xv[tr], label=y[tr])
        ds_va = lgb.Dataset(Xv[va], label=y[va], reference=ds_tr)
        booster = lgb.train(p, ds_tr, num_boost_round=MAX_ROUNDS,
                            valid_sets=[ds_va],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
        best_iters.append(booster.best_iteration or 300)
    n_rounds = int(np.median(best_iters))
    log(f"  arbres : par pli {best_iters} -> fige a {n_rounds}")

    oof = np.zeros(len(y), dtype=np.float64)
    models = []
    for k in ks:
        tr, va = folds != k, folds == k
        booster = lgb.train(p, lgb.Dataset(Xv[tr], label=y[tr]),
                            num_boost_round=n_rounds)
        oof[va] = booster.predict(Xv[va])
        models.append(booster)
    return oof, models, {"n_rounds": n_rounds, "columns": list(X.columns)}


def predict_gbm(models: list, X: pd.DataFrame) -> np.ndarray:
    Xv = X.to_numpy(dtype=np.float64)
    return np.mean([m.predict(Xv) for m in models], axis=0)


def train_linear_cv(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                    C: float = 0.05, seed: int = 2026,
                    log=print) -> tuple[np.ndarray, list, dict]:
    """Logistique L2 sur descripteurs rangs-normalises (robuste aux valeurs extremes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import QuantileTransformer

    y = np.asarray(y, dtype=int)
    ks = sorted(np.unique(folds[folds >= 0]))
    oof = np.zeros(len(y), dtype=np.float64)
    models = []
    for k in ks:
        tr, va = folds != k, folds == k
        pipe = Pipeline([
            ("qt", QuantileTransformer(output_distribution="normal",
                                       n_quantiles=min(500, int(tr.sum())),
                                       random_state=seed)),
            ("lr", LogisticRegression(C=C, max_iter=3000, solver="lbfgs")),
        ])
        pipe.fit(X[tr], y[tr])
        oof[va] = pipe.predict_proba(X[va])[:, 1]
        models.append(pipe)
    log(f"  logistique : C={C}, {len(models)} plis")
    return oof, models, {"C": C, "columns": list(X.columns)}


def predict_linear(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def gbm_importance(models: list, columns: list, top: int = 20) -> pd.DataFrame:
    gains = np.mean([m.feature_importance("gain") for m in models], axis=0)
    return (pd.DataFrame({"feature": columns, "gain": gains})
            .sort_values("gain", ascending=False).head(top).reset_index(drop=True))
