# src/viz/diagnostics.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "Predicted vs Observed"):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]

    if y_true.size == 0:
        return

    fig = plt.figure(figsize=(6.5, 6))
    plt.scatter(y_true, y_pred, s=2, alpha=0.4)
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx], linewidth=2)  # 1:1 line
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_residual_hist(residuals: np.ndarray, out_path: Path, title: str = "Residuals"):
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return
    fig = plt.figure(figsize=(6.5, 4))
    plt.hist(residuals, bins=60)
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
