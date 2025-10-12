# src/analysis/corr.py
from __future__ import annotations
import numpy as np
from scipy import stats

def pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute Pearson & Spearman correlations using only finite pairs.
    Returns a dict with coefficients, p-values, and sample count.
    """
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y, dtype="float32")

    m = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[m], y[m]
    n = int(x2.size)
    if n < 2:
        return {"pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan, "n": n}

    # handle constant arrays
    if np.nanstd(x2) == 0 or np.nanstd(y2) == 0:
        return {"pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan, "n": n}

    pr, pp = stats.pearsonr(x2, y2)
    sr, sp = stats.spearmanr(x2, y2)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
            "n": n}
