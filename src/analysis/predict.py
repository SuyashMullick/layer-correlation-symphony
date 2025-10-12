# src/analysis/predict.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    a = a[m]
    b = b[m]
    va = a - a.mean()
    vb = b - b.mean()
    denom = np.sqrt((va * va).sum()) * np.sqrt((vb * vb).sum())
    return float((va * vb).sum() / denom) if denom > 0 else float("nan")


@dataclass
class FitResult:
    y_test: np.ndarray
    y_pred: np.ndarray
    metrics: Dict[str, float]
    model_name: str
    coefficients: Optional[np.ndarray] = None       # for linear models
    importances: Optional[np.ndarray] = None        # for RF


def fit_model_with_options(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str = "ridge",               # 'ridge' | 'lasso' | 'rf'
    alpha: float = 1.0,                 # ridge/lasso only
    standardize: bool = True,           # ridge/lasso only
    test_size: float = 0.2,
    random_state: int = 42,
    feature_names: Optional[List[str]] = None,
    rf_params: Optional[Dict] = None,   # n_estimators, max_depth, min_samples_leaf, max_features, ...
) -> Dict:
    """
    Train a regression model and return predictions + diagnostics.
    Works with linear (ridge/lasso) and tree (random forest) models.
    """
    if rf_params is None:
        rf_params = {}

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = model.lower().strip()
    linear = model in {"ridge", "lasso"}

    if linear:
        base = Ridge(alpha=alpha, random_state=random_state) if model == "ridge" \
            else Lasso(alpha=alpha, random_state=random_state, max_iter=20000)
        if standardize:
            est = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("model", base)])
        else:
            est = base
    elif model == "rf":
        # sensible defaults for noisy ecological rasters
        defaults = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
            max_features=1.0,  # all features (scikit-learn 1.4+ default)
        )
        defaults.update(rf_params)
        est = RandomForestRegressor(**defaults)
    else:
        raise ValueError("Unknown --model. Use 'ridge', 'lasso', or 'rf'.")

    # fit + predict
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)

    # metrics
    resid = y_test - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    # R² from estimator (uses the same test split)
    r2 = float(getattr(est, "score")(X_test, y_test))
    pearson_r = _pearson_corr(y_test, y_pred)

    out = FitResult(
        y_test=y_test,
        y_pred=y_pred,
        metrics={"r2": r2, "rmse": rmse, "mae": mae, "pearson_r": pearson_r},
        model_name=model,
    )

    # coefficients or importances
    if linear:
        coef: Optional[np.ndarray] = None
        if isinstance(est, Pipeline):
            coef = est.named_steps["model"].coef_
        else:
            coef = est.coef_
        out.coefficients = np.asarray(coef, dtype="float64")
    else:  # rf
        out.importances = np.asarray(est.feature_importances_, dtype="float64")

    # return as a plain dict to keep your CLI code simple
    return {
        "y_test": out.y_test,
        "y_pred": out.y_pred,
        "metrics": out.metrics,
        "model_name": out.model_name,
        "coefficients": out.coefficients,
        "importances": out.importances,
    }
