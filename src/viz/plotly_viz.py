# src/viz/plotly_viz.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline


def _ensure_vec(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).astype(float)
    return a[np.isfinite(a)]


def _polyfit_line(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    if x.size >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        m, b = np.polyfit(x, y, 1)
        return float(m), float(b)
    return None


def write_parity_plotly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_html: Path,
    title: str = "Predicted vs Observed",
    max_points: int = 300_000,
):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = np.asarray(y_true)[m].astype(float), np.asarray(y_pred)[m].astype(float)
    if yt.size == 0:
        return

    # Subsample for responsiveness
    if yt.size > max_points:
        idx = np.random.default_rng(42).choice(yt.size, size=max_points, replace=False)
        yt, yp = yt[idx], yp[idx]

    line = _polyfit_line(yt, yp)

    fig = make_subplots(rows=1, cols=1)

    # main cloud (WebGL)
    fig.add_trace(
        go.Scattergl(
            x=yt, y=yp, mode="markers", name="points",
            marker=dict(size=3, opacity=0.35),
            hovertemplate="Observed=%{x:.4f}<br>Predicted=%{y:.4f}<extra></extra>",
        )
    )

    # identity 1:1
    lo = float(np.nanmin([yt.min(), yp.min()]))
    hi = float(np.nanmax([yt.max(), yp.max()]))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", name="1:1",
            line=dict(width=2, dash="dash"),
            hoverinfo="skip"
        )
    )

    # OLS trend
    if line:
        xs = np.linspace(lo, hi, 200)
        ys = line[0] * xs + line[1]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", name=f"trend (m={line[0]:.3f})",
                line=dict(width=2), hoverinfo="skip"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Observed",
        yaxis_title="Predicted",
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        width=700, height=700,  # make it a square canvas
    )
    # enforce equal scaling
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    plot_offline(fig, filename=str(out_html), auto_open=False, include_plotlyjs=True)


def write_residuals_plotly(
    residuals: np.ndarray,
    out_html: Path,
    title: str = "Residuals",
    bins: int = 80,
    log_count: bool = False,
):
    r = _ensure_vec(residuals)
    if r.size == 0:
        return

    # nice symmetric bins around 0
    lo, hi = np.quantile(r, [0.01, 0.99])
    bound = float(max(abs(lo), abs(hi)))
    fig = px.histogram(
        x=np.clip(r, -bound, bound),
        nbins=bins,
        labels={"x": "Residual (y - ŷ)"},
        title=title,
    )
    fig.update_traces(hovertemplate="Residual=%{x:.4f}<br>Count=%{y}<extra></extra>")
    if log_count:
        fig.update_yaxes(type="log", title="Count (log)")
    else:
        fig.update_yaxes(title="Count")
    fig.update_layout(template="simple_white")

    # vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_width=2)

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    plot_offline(fig, filename=str(out_html), auto_open=False, include_plotlyjs=True)


def write_scatter_plotly(
    x: np.ndarray,
    y: np.ndarray,
    r_pearson: float,
    r_spearman: float,
    out_html: Path,
    x_label: str = "Layer A",
    y_label: str = "Layer B",
    title: Optional[str] = None,
    max_points: int = 300_000,
    density_cutover: int = 700_000,
):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m].astype(float), np.asarray(y)[m].astype(float)
    if x.size == 0:
        return

    # Decide representation based on size
    use_density = x.size > density_cutover

    if x.size > max_points:
        idx = np.random.default_rng(42).choice(x.size, size=max_points, replace=False)
        x, y = x[idx], y[idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=3, opacity=0.4, color="royalblue"),
            name="points",
            hovertemplate=f"{x_label}=%{{x:.4f}}<br>{y_label}=%{{y:.4f}}<extra></extra>",
        )
    )

    # OLS trend
    line = _polyfit_line(x, y)
    if line:
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        xs = np.linspace(lo, hi, 200)
        ys = line[0] * xs + line[1]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", name=f"trend (m={line[0]:.3f})",
                line=dict(width=2)
            )
        )

    # Stats box
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False, align="left",
        text=f"<b>Pearson r</b> = {r_pearson:.3f}<br>"
             f"<b>Spearman r</b> = {r_spearman:.3f}<br>"
             f"<b>N</b> = {x.size:,}",
        bordercolor="black", borderwidth=1, bgcolor="white", opacity=0.85
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        width=700,
        height=700,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")  # prevents weird stretching

    # save HTML (inline JS for offline use)
    import plotly.io as pio
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, str(out_html), include_plotlyjs="inline", full_html=True)
