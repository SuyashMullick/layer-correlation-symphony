# src/viz/scatter.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    r_pearson: float,
    r_spearman: float,
    out_path: Path,
    max_points: int = 200_000,
    x_label: str = "Layer A",
    y_label: str = "Layer B",
    title: Optional[str] = None,
):
    # keep finite only
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    # subsample for plotting if needed
    if x.size > max_points:
        idx = np.random.default_rng(42).choice(x.size, size=max_points, replace=False)
        x_plot, y_plot = x[idx], y[idx]
    else:
        x_plot, y_plot = x, y

    # fit simple linear trend for visual cue (ignore if degenerate)
    line = None
    if x_plot.size >= 2 and np.nanstd(x_plot) > 0:
        slope, intercept = np.polyfit(x_plot, y_plot, 1)
        line = (slope, intercept)

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(x_plot, y_plot, s=1, alpha=0.5)
    if line:
        xs = np.linspace(np.nanmin(x_plot), np.nanmax(x_plot), 200)
        ys = line[0] * xs + line[1]
        plt.plot(xs, ys, linewidth=2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)

    # stats box
    txt = f"Pearson r = {r_pearson:.3f}\nSpearman r = {r_spearman:.3f}\nN = {x.size:,}"
    plt.gca().text(
        0.02, 0.98, txt, transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
