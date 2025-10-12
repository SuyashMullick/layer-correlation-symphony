from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.geo.io import read_pair_as_arrays
from src.analysis.corr import pearson_spearman
from src.viz.plotly_viz import write_scatter_plotly
from src.viz.heatmap import save_correlation_heatmap # <--- ADDED IMPORT

def main():
    ap = argparse.ArgumentParser(
        description="Compare two raster layers: compute correlation and save scatter plot and heatmap."
    )
    ap.add_argument("--a", required=True, help="Path to first raster (.tif)")
    ap.add_argument("--b", required=True, help="Path to second raster (.tif)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--nodata", default="0,-9999", help="Extra nodata sentinels")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_nodata = [float(x) for x in args.nodata.split(",") if x.strip()]

    x, y, meta = read_pair_as_arrays(args.a, args.b, extra_nodata=extra_nodata)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if x.size < 2:
        raise SystemExit("Not enough overlapping valid pixels to compute correlation.")

    stats = pearson_spearman(x, y)
    stats["n_pixels"] = int(x.size)
    stats["layer_a"] = Path(args.a).stem
    stats["layer_b"] = Path(args.b).stem

    pd.DataFrame([stats]).to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- ADDED: Generate DataFrame and save heatmap ---
    df = pd.DataFrame({stats["layer_a"]: x, stats["layer_b"]: y})
    save_correlation_heatmap(
        df=df,
        out_path=out_dir / "predictor_heatmap.png",
        title="Layer Correlation Heatmap"
    )
    # --------------------------------------------------

    try:
        write_scatter_plotly(
            x, y,
            r_pearson=stats["pearson_r"],
            r_spearman=stats["spearman_r"],
            out_html=out_dir / "scatter.html",
            x_label=stats["layer_a"],
            y_label=stats["layer_b"],
            title=f"{stats['layer_a']} vs {stats['layer_b']}"
        )
        print(f"[symph-compare] wrote: {out_dir/'scatter.html'}")
    except Exception as e:
        print(f"[plotly warning] Could not create interactive scatter: {e}")

    print(f"[symph-compare] All results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

