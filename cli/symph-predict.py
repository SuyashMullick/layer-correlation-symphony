from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from src.geo.io import read_many_as_matrix
from src.viz.diagnostics import save_parity_plot, save_residual_hist
from src.viz.plotly_viz import write_parity_plotly, write_residuals_plotly
from src.viz.heatmap import save_correlation_heatmap
import xgboost as xgb


def transform_target(y: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply transformation to the target variable."""
    if transform_type == "log1p":
        y_transformed = np.log1p(y)
        print(f"[transform] Applied log1p transformation to target.")
        return y_transformed
    return y

def inverse_transform_target(y_transformed: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply inverse transformation to the predicted target."""
    if transform_type == "log1p":
        y_original = np.expm1(y_transformed)
        y_original[y_original < 0] = 0
        return y_original
    return y_transformed


def detect_and_remove_outliers(X: np.ndarray, y: np.ndarray, n_sigma: float = 3.5) -> tuple:
    """Remove outliers using IQR method for robustness."""
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower = q1 - n_sigma * iqr
    upper = q3 + n_sigma * iqr
    mask_y = (y >= lower) & (y <= upper)
    
    mask_X = np.ones(X.shape[0], dtype=bool)
    for i in range(X.shape[1]):
        q1, q3 = np.percentile(X[:, i], [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - n_sigma * iqr
            upper = q3 + n_sigma * iqr
            mask_X &= (X[:, i] >= lower) & (X[:, i] <= upper)
    
    mask = mask_y & mask_X
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"[outlier removal] Removed {n_removed} outliers ({100*n_removed/len(mask):.1f}%)")
    
    return X[mask], y[mask]


def fit_random_forest(X: np.ndarray, y: np.ndarray, test_size: float, transform_y: str = "none") -> Dict[str, Any]:
    """Fit Random Forest Regressor."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"[RF] Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=20, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_transformed = rf.predict(X_test)

    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)
    
    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    return {"metrics": {"r2": r2, "rmse": rmse, "mae": mae, "pearson_r": pearson_r},
            "y_test": y_test_original, "y_pred": y_pred_original, "feature_importance": rf.feature_importances_.tolist()}


def fit_gradient_boosting(X: np.ndarray, y: np.ndarray, test_size: float, transform_y: str = "none") -> Dict[str, Any]:
    """Fit Gradient Boosting Regressor (GBM)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"[GBM] Training Gradient Boosting...")
    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, min_samples_split=20, min_samples_leaf=10, random_state=42)
    gbm.fit(X_train, y_train)
    y_pred_transformed = gbm.predict(X_test)
    
    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)

    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    return {"metrics": {"r2": r2, "rmse": rmse, "mae": mae, "pearson_r": pearson_r},
            "y_test": y_test_original, "y_pred": y_pred_original, "feature_importance": gbm.feature_importances_.tolist()}

def fit_xgboost(X: np.ndarray, y: np.ndarray, test_size: float, transform_y: str = "none") -> Dict[str, Any]:
    """Fit XGBoost Regressor (XGB)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"[XGB] Training XGBoost...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, objective='reg:squarederror')
    model.fit(X_train, y_train)
    y_pred_transformed = model.predict(X_test)
    
    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)

    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    return {"metrics": {"r2": r2, "rmse": rmse, "mae": mae, "pearson_r": pearson_r},
            "y_test": y_test_original, "y_pred": y_pred_original, "feature_importance": model.feature_importances_.tolist()}


def fit_neural_network(X: np.ndarray, y: np.ndarray, test_size: float, transform_y: str = "none") -> Dict[str, Any]:
    """Fit a Multi-layer Perceptron (Neural Network) Regressor."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"[NN] Training Neural Network (MLP)...")
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=20, tol=1e-4, verbose=False)
    model.fit(X_train, y_train)
    y_pred_transformed = model.predict(X_test)
    
    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)

    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    return {"metrics": {"r2": r2, "rmse": rmse, "mae": mae, "pearson_r": pearson_r},
            "y_test": y_test_original, "y_pred": y_pred_original, "feature_importance": None}


def main():
    ap = argparse.ArgumentParser(description="Predict a target raster from multiple predictor rasters.")
    ap.add_argument("--target", required=True, help="Path to target raster (.tif)")
    ap.add_argument("--predictors", nargs="+", required=True, help="Paths to predictor rasters (.tif)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--nodata", default="-9999", help="Extra nodata sentinels CSV. Default: -9999 (NOT 0!)")
    ap.add_argument("--sample", type=int, default=200_000, help="Max pixels for training+testing")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout ratio")
    ap.add_argument("--model", default="auto", choices=["rf", "gbm", "xgb", "nn", "auto"], help="Model type: rf, gbm, xgb, nn, or auto")
    ap.add_argument("--remove_outliers", action="store_true", help="Remove outliers using IQR")
    ap.add_argument("--outlier_sigma", type=float, default=3.0, help="IQR multiplier for outlier detection")
    ap.add_argument("--transform_y", default="none", choices=["none", "log1p"], help="Apply transformation to the target variable (y)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_nodata = [float(x.strip()) for x in args.nodata.split(",") if x.strip()]
    
    print(f"\n[config] Treating these values as nodata: {extra_nodata}\n")

    print(f"[loading] Reading target + {len(args.predictors)} predictor rasters...")
    paths = [args.target] + args.predictors
    X, y, meta = read_many_as_matrix(paths, extra_nodata=extra_nodata)

    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]
    print(f"[loading] Found {X.shape[0]:,} valid pixels")

    if X.shape[0] < 50:
        raise SystemExit(f"ERROR: Only {X.shape[0]} valid pixels found. Check your nodata settings!")

    feat_names = [Path(p).stem for p in args.predictors]
    target_name = Path(args.target).stem

    predictor_df = pd.DataFrame(X, columns=feat_names)
    save_correlation_heatmap(
        df=predictor_df,
        out_path=out_dir / "predictor_heatmap.png",
        title=f"Predictor Correlation Matrix"
    )

    y = transform_target(y, args.transform_y)

    if args.remove_outliers:
        X, y = detect_and_remove_outliers(X, y, n_sigma=args.outlier_sigma)
        print(f"[outliers] Remaining: {X.shape[0]:,} pixels")

    if X.shape[0] > args.sample:
        rng = np.random.default_rng(42)
        mask_non_zero = y > 1e-6
        X_nz, y_nz = X[mask_non_zero], y[mask_non_zero]
        X_z, y_z = X[~mask_non_zero], y[~mask_non_zero]

        n_nz_kept = min(len(X_nz), args.sample // 2)
        
        if n_nz_kept > 0:
            idx_nz = rng.choice(len(X_nz), size=n_nz_kept, replace=False)
            n_z_needed = args.sample - n_nz_kept
            n_z_kept = min(len(X_z), n_z_needed)
            idx_z = rng.choice(len(X_z), size=n_z_kept, replace=False)
            
            X = np.concatenate((X_nz[idx_nz], X_z[idx_z]), axis=0)
            y = np.concatenate((y_nz[idx_nz], y_z[idx_z]), axis=0)
            
            p = rng.permutation(len(X))
            X, y = X[p], y[p]
            print(f"[sampling] Subsampled to {len(X):,} pixels ({n_nz_kept:,} non-zero targets).")
        else:
            idx = rng.choice(X.shape[0], size=args.sample, replace=False)
            X, y = X[idx], y[idx]
            print(f"[sampling] Subsampled to {args.sample:,} pixels (uniform sample).")

    original_feat_names = feat_names.copy()

    results_dict = {}
    if args.model in ["rf", "auto"]:
        results_dict["rf"] = {"results": fit_random_forest(X, y, args.test_size, args.transform_y)}
    if args.model in ["gbm", "auto"]:
        results_dict["gbm"] = {"results": fit_gradient_boosting(X, y, args.test_size, args.transform_y)}
    if args.model in ["xgb", "auto"]:
        results_dict["xgb"] = {"results": fit_xgboost(X, y, args.test_size, args.transform_y)}
    if args.model in ["nn", "auto"]:
        results_dict["nn"] = {"results": fit_neural_network(X, y, args.test_size, args.transform_y)}

    if not results_dict:
        raise SystemExit("ERROR: No models were trained.")
        
    best_model = max(results_dict.items(), key=lambda x: x[1]["results"]["metrics"]["r2"])
    best_name, best_result = best_model[0], best_model[1]["results"]
    
    print(f"\nBEST MODEL: {best_name.upper()}")

    metrics = {"target": target_name, "predictors": original_feat_names, "n_samples": X.shape[0], "test_size": args.test_size, "best_model": best_name, **best_result["metrics"]}
    
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    
    if best_result.get("feature_importance"):
        imp_df = pd.DataFrame({"feature": original_feat_names, "importance": best_result["feature_importance"]}).sort_values("importance", ascending=False)
        imp_df.to_csv(out_dir / "feature_importance.csv", index=False)

    if len(results_dict) > 1:
        comparison = [{"model": name, **data["results"]["metrics"]} for name, data in results_dict.items()]
        pd.DataFrame(comparison).to_csv(out_dir / "model_comparison.csv", index=False)

    save_parity_plot(y_true=best_result["y_test"], y_pred=best_result["y_pred"], out_path=out_dir / "parity.png", title=f"{target_name} ({best_name})")
    save_residual_hist(residuals=best_result["y_test"] - best_result["y_pred"], out_path=out_dir / "residuals.png", title=f"{target_name} ({best_name})")
    
    try:
        write_parity_plotly(y_true=best_result["y_test"], y_pred=best_result["y_pred"], out_html=out_dir / "parity.html", title=f"{target_name} ({best_name})")
        write_residuals_plotly(residuals=best_result["y_test"] - best_result["y_pred"], out_html=out_dir / "residuals.html", title=f"{target_name} ({best_name})")
    except Exception as e:
        print(f"[plotly warning] Could not create interactive plots: {e}")

    print(f"\n[output] Results saved to: {out_dir}/")

if __name__ == "__main__":
    main()

