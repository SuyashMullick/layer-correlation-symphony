# src/analysis/predict.py
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    alpha: float = 1.0,
    standardize: bool = True,
    feature_names: list[str] | None = None,
):
    """
    Train/test Ridge regression and return metrics + predictions + coefficients.
    Returns:
      {
        "metrics": {...},
        "y_test": np.ndarray,
        "y_pred": np.ndarray,
        "coefficients": np.ndarray | None,
      }
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=alpha, random_state=42)))
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Pearson correlation between y and ŷ (good scale-free metric)
    try:
        pr, _ = pearsonr(y_test, y_pred)
        pearson_r = float(pr)
    except Exception:
        pearson_r = float("nan")

    # Try to recover coefficients in original feature space (post-scaling if used)
    coefficients = None
    try:
        ridge = pipe.named_steps["ridge"]
        if standardize:
            scaler = pipe.named_steps["scaler"]
            # back-transform coefficients to original scale
            # coef_orig = coef_scaled / sigma_x * sigma_y  (not needed for ranking; report scaled)
            # For simplicity, we’ll report scaled-space coefficients which are still useful for ranking.
            coefficients = ridge.coef_.astype(float)
        else:
            coefficients = ridge.coef_.astype(float)
    except Exception:
        pass

    return {
        "metrics": {
            "r2": float(r2),
            "rmse": rmse,
            "mae": mae,
            "pearson_r": pearson_r,
            "n_test": int(y_test.size),
        },
        "y_test": y_test.astype(float),
        "y_pred": y_pred.astype(float),
        "coefficients": coefficients,
    }
