# cli/symph-predict.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.geo.io import read_many_as_matrix
from src.analysis.predict import fit_model_with_options
from src.viz.diagnostics import save_parity_plot, save_residual_hist


def main():
    ap = argparse.ArgumentParser(
        description="Predict a target raster from multiple predictor rasters."
    )
    ap.add_argument("--target", required=True, help="Path to target raster (.tif)")
    ap.add_argument("--predictors", nargs="+", required=True, help="Paths to predictor rasters (.tif)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--nodata", default="0,-9999", help="Extra nodata sentinels CSV. Default: 0,-9999")
    ap.add_argument("--sample", type=int, default=200_000, help="Max pixels to use for training+testing")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout ratio (default 0.2)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Alpha for ridge/lasso")
    ap.add_argument("--standardize", action="store_true", help="Standardize features (linear models only)")
    ap.add_argument("--model", choices=["ridge", "lasso", "rf"], default="ridge",
                    help="Model type: ridge, lasso, or rf (Random Forest)")

    # RF-specific flags (ignored for other models)
    ap.add_argument("--rf-n-estimators", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-max-features", type=float, default=1.0,
                    help="Fraction of features to consider at each split (1.0 = all)")

    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_nodata = [float(x) for x in args.nodata.split(",") if x.strip()]

    # Read aligned intersection across target + predictors -> X (n, p), y (n,)
    paths = [args.target] + args.predictors
    X, y, meta = read_many_as_matrix(paths, extra_nodata=extra_nodata)

    # keep only rows where all predictors and target are finite
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]

    if X.shape[0] < 10:
        raise SystemExit("Not enough overlapping valid pixels to train.")

    # optional subsample to keep it snappy
    if X.shape[0] > args.sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=args.sample, replace=False)
        X = X[idx]
        y = y[idx]

    feat_names = [Path(p).stem for p in args.predictors]
    target_name = Path(args.target).stem

    rf_params = dict(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        max_features=args.rf_max_features,
    )

    results = fit_model_with_options(
        X, y,
        model=args.model,
        test_size=args.test_size,
        alpha=args.alpha,
        standardize=args.standardize,
        feature_names=feat_names,
        rf_params=rf_params,
    )

    # Save metrics
    metrics = {
        "target": target_name,
        "predictors": feat_names,
        "n_samples": int(X.shape[0]),
        "test_size": args.test_size,
        "alpha": args.alpha,
        "standardize": bool(args.standardize),
        "model": results["model_name"],
        **results["metrics"]
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    # Save coefficients / importances
    if results.get("coefficients") is not None:
        coef_df = pd.DataFrame({
            "feature": feat_names,
            "weight": results["coefficients"]
        }).sort_values("weight", ascending=False)
        coef_df.to_csv(out_dir / "feature_weights.csv", index=False)
    elif results.get("importances") is not None:
        imp_df = pd.DataFrame({
            "feature": feat_names,
            "weight": results["importances"]
        }).sort_values("weight", ascending=False)
        imp_df.to_csv(out_dir / "feature_importances.csv", index=False)

    # Plots
    save_parity_plot(
        y_true=results["y_test"],
        y_pred=results["y_pred"],
        out_path=out_dir / "parity.png",
        title=f"{target_name}: predicted vs observed"
    )
    save_residual_hist(
        residuals=results["y_test"] - results["y_pred"],
        out_path=out_dir / "residuals.png",
        title=f"{target_name}: residuals"
    )

    print(f"[symph-predict] model={metrics['model']} n={metrics['n_samples']} "
          f"R2={metrics['r2']:.3f} RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} "
          f"Pearson(y,ŷ)={metrics['pearson_r']:.3f}")
    print(f"[symph-predict] wrote: {out_dir/'metrics.json'}, {out_dir/'metrics.csv'}")
    if (out_dir / "feature_weights.csv").exists():
        print(f"[symph-predict] wrote: {out_dir/'feature_weights.csv'}")
    if (out_dir / "feature_importances.csv").exists():
        print(f"[symph-predict] wrote: {out_dir/'feature_importances.csv'}")
    print(f"[symph-predict] wrote: {out_dir/'parity.png'}, {out_dir/'residuals.png'}")


if __name__ == "__main__":
    main()
