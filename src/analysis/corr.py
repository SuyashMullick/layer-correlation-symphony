# src/analysis/corr.py
from __future__ import annotations

import numpy as np
from scipy import stats


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute Pearson & Spearman correlation using only finite pairs.
    Returns dict with r, p-values.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[m], y[m]
    n = int(x2.size)
    if n < 2:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan"),
                "n": n}

    pr, pp = stats.pearsonr(x2, y2)
    sr, sp = stats.spearmanr(x2, y2)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
            "n": n}
