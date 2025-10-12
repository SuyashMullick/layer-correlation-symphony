from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.geo.io import read_many_as_matrix
from src.viz.diagnostics import save_parity_plot, save_residual_hist
from src.viz.plotly_viz import write_parity_plotly, write_residuals_plotly
import xgboost as xgb


def transform_target(y: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply transformation to the target variable."""
    if transform_type == "log1p":
        # log1p is log(1 + y), excellent for zero-inflated data
        y_transformed = np.log1p(y)
        print(f"[transform] Applied log1p transformation to target.")
        return y_transformed
    return y

def inverse_transform_target(y_transformed: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply inverse transformation to the predicted target."""
    if transform_type == "log1p":
        # expm1 is exp(y) - 1, the inverse of log1p
        y_original = np.expm1(y_transformed)
        # Ensure values don't drop below zero due to model error on expm1
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


def check_multicollinearity(X: np.ndarray, feat_names: list, threshold: float = 0.95) -> None:
    """Check for highly correlated predictors."""
    if X.shape[1] < 2:
        return
    
    corr_matrix = np.corrcoef(X.T)
    high_corr = []
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr.append((feat_names[i], feat_names[j], corr_matrix[i, j]))
    
    if high_corr:
        print("\n[WARNING] High correlation detected between predictors:")
        for f1, f2, corr in high_corr:
            print(f"  {f1} <-> {f2}: r={corr:.3f}")
        print("  Consider removing redundant predictors.\n")


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Add polynomial and interaction features."""
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    print(f"[features] Added polynomial features (degree={degree}): {X.shape[1]} → {X_poly.shape[1]} features")
    return X_poly


def fit_random_forest(X: np.ndarray, y: np.ndarray, test_size: float, 
                      feature_names: list, transform_y: str = "none") -> Dict[str, Any]:
    """Fit Random Forest Regressor."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"[RF] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_transformed = rf.predict(X_test)

    # Apply INVERSE transform before calculating final metrics/returning
    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)
    
    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    # Feature importance
    importances = rf.feature_importances_
    
    return {
        "metrics": {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "pearson_r": float(pearson_r)
        },
        "y_test": y_test_original,
        "y_pred": y_pred_original,
        "feature_importance": importances.tolist()
    }


def fit_gradient_boosting(X: np.ndarray, y: np.ndarray, test_size: float,
                          feature_names: list, transform_y: str = "none") -> Dict[str, Any]:
    """Fit Gradient Boosting Regressor (GBM)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"[GBM] Training Gradient Boosting...")
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    gbm.fit(X_train, y_train)
    y_pred_transformed = gbm.predict(X_test)
    
    # Apply INVERSE transform before calculating final metrics/returning
    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)

    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    return {
        "metrics": {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "pearson_r": float(pearson_r)
        },
        "y_test": y_test_original,
        "y_pred": y_pred_original,
        "feature_importance": gbm.feature_importances_.tolist()
    }

def fit_xgboost(X: np.ndarray, y: np.ndarray, test_size: float,
                feature_names: list, transform_y: str = "none") -> Dict[str, Any]:
    """Fit XGBoost Regressor (XGB)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"[XGB] Training XGBoost...")
    # Using a common set of robust parameters
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror' # Standard objective for regression
    )
    model.fit(X_train, y_train)
    y_pred_transformed = model.predict(X_test)
    
    # Apply INVERSE transform before calculating final metrics/returning
    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)

    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    return {
        "metrics": {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "pearson_r": float(pearson_r)
        },
        "y_test": y_test_original,
        "y_pred": y_pred_original,
        "feature_importance": model.feature_importances_.tolist()
    }


def fit_neural_network(X: np.ndarray, y: np.ndarray, test_size: float,
                       feature_names: list, transform_y: str = "none") -> Dict[str, Any]:
    """Fit a Multi-layer Perceptron (Neural Network) Regressor."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # NNs perform best with standardized features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"[NN] Training Neural Network (MLP)...")
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50), # 2 hidden layers with 100 and 50 neurons
        activation='relu',
        solver='adam',
        max_iter=500, # Max iterations
        random_state=42,
        early_stopping=True, # Stop early if validation score is not improving
        n_iter_no_change=20, 
        tol=1e-4,
        verbose=False
    )
    # Using the scaled data for training
    model.fit(X_train, y_train)
    y_pred_transformed = model.predict(X_test)
    
    # Apply INVERSE transform before calculating final metrics/returning
    y_test_original = inverse_transform_target(y_test, transform_y)
    y_pred_original = inverse_transform_target(y_pred_transformed, transform_y)

    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    pearson_r, _ = pearsonr(y_test_original, y_pred_original)
    
    # MLPRegressor does not provide feature importance directly.
    return {
        "metrics": {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "pearson_r": float(pearson_r)
        },
        "y_test": y_test_original,
        "y_pred": y_pred_original,
        "feature_importance": None # No direct feature importance for MLP
    }


def main():
    ap = argparse.ArgumentParser(
        description="Predict a target raster from multiple predictor rasters."
    )
    ap.add_argument("--target", required=True, help="Path to target raster (.tif)")
    ap.add_argument("--predictors", nargs="+", required=True, help="Paths to predictor rasters (.tif)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--nodata", default="-9999", help="Extra nodata sentinels CSV. Default: -9999 (NOT 0!)")
    ap.add_argument("--sample", type=int, default=200_000, help="Max pixels for training+testing")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout ratio")
    
    # UPDATED: Added 'nn' to model choices
    ap.add_argument("--model", default="auto", choices=["rf", "gbm", "xgb", "nn", "auto"],
                    help="Model type: rf (random forest), gbm (gradient boosting), xgb (XGBoost), nn (neural network), or auto (try all)")
    
    # REMOVED: --alpha, --polynomial, --standardize arguments
    
    ap.add_argument("--remove_outliers", action="store_true", help="Remove outliers using IQR")
    ap.add_argument("--outlier_sigma", type=float, default=3.0, help="IQR multiplier for outlier detection")
    ap.add_argument("--transform_y", default="none", choices=["none", "log1p"],
                    help="Apply transformation to the target variable (y)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse nodata - IMPORTANT: split and convert properly
    extra_nodata = []
    if args.nodata:
        for x in args.nodata.split(","):
            x = x.strip()
            if x:
                try:
                    extra_nodata.append(float(x))
                except ValueError:
                    print(f"[WARNING] Invalid nodata value: {x}")
    
    print(f"\n[config] Treating these values as nodata: {extra_nodata}")
    print(f"[config] Zero values WILL be preserved as real data!\n")

    # Read data
    print(f"[loading] Reading target + {len(args.predictors)} predictor rasters...")
    paths = [args.target] + args.predictors
    X, y, meta = read_many_as_matrix(paths, extra_nodata=extra_nodata)

    # Keep only finite values
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]
    print(f"[loading] Found {X.shape[0]:,} valid pixels")

    if X.shape[0] < 50:
        raise SystemExit(f"ERROR: Only {X.shape[0]} valid pixels found. Check your nodata settings!")

    feat_names = [Path(p).stem for p in args.predictors]
    target_name = Path(args.target).stem

    # Apply Target Transformation
    y = transform_target(y, args.transform_y)

    # Remove outliers
    if args.remove_outliers:
        X, y = detect_and_remove_outliers(X, y, n_sigma=args.outlier_sigma)
        print(f"[outliers] Remaining: {X.shape[0]:,} pixels")

    # Subsample (FINAL CORRECTED LOGIC: Prioritizing Non-Zero Targets)
    if X.shape[0] > args.sample:
        rng = np.random.default_rng(42)
        
        # 1. Identify non-zero samples. After log1p, zero remains 0.
        # We assume the target "y" is log-transformed at this point.
        mask_non_zero = y > 1e-6 # True non-zero values will be slightly greater than 0
        X_nz, y_nz = X[mask_non_zero], y[mask_non_zero]
        X_z, y_z = X[~mask_non_zero], y[~mask_non_zero]

        # 2. Decide how many non-zero samples to keep (up to half the budget)
        n_nz_kept = min(len(X_nz), args.sample // 2)
        
        if n_nz_kept > 0:
            # Sample non-zero points
            idx_nz = rng.choice(len(X_nz), size=n_nz_kept, replace=False)
            
            # Sample zero points for the remaining budget
            n_z_needed = args.sample - n_nz_kept
            n_z_kept = min(len(X_z), n_z_needed)
            idx_z = rng.choice(len(X_z), size=n_z_kept, replace=False)
            
            # 3. Combine and shuffle the sampled data
            X = np.concatenate((X_nz[idx_nz], X_z[idx_z]), axis=0)
            y = np.concatenate((y_nz[idx_nz], y_z[idx_z]), axis=0)
            
            p = rng.permutation(len(X))
            X = X[p]
            y = y[p]
            
            print(f"[sampling] Subsampled to {len(X):,} pixels ({n_nz_kept:,} non-zero targets).")
        else:
            # If no non-zero points exist after filtering, proceed with uniform sample (this indicates an issue with the source data)
            idx = rng.choice(X.shape[0], size=args.sample, replace=False)
            X = X[idx]
            y = y[idx]
            print(f"[sampling] Subsampled to {args.sample:,} pixels (uniform sample, NO non-zero targets found).")

    # Data summary
    print(f"\n[data summary]")
    print(f"  Target ({target_name}):")
    # Use the inverse transform for summary statistics if y was transformed
    y_summary = inverse_transform_target(y, args.transform_y) 
    print(f"    mean={y_summary.mean():.4f}, std={y_summary.std():.4f}")
    print(f"    min={y_summary.min():.4f}, max={y_summary.max():.4f}")
    print(f"    zeros: {(y_summary == 0).sum()} ({100*(y_summary == 0).sum()/len(y_summary):.1f}%)")
    
    for i, name in enumerate(feat_names):
        print(f"  Predictor: {name}")
        print(f"    mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}")
        print(f"    min={X[:, i].min():.4f}, max={X[:, i].max():.4f}")

    # Check correlations
    check_multicollinearity(X, feat_names)

    # Store original feature names
    original_feat_names = feat_names.copy()

    # Train models
    results_dict = {}
    
    if args.model in ["rf", "auto"]:
        print(f"\n{'='*60}")
        print(f"RANDOM FOREST (Non-linear)")
        print(f"{'='*60}")
        rf_results = fit_random_forest(X, y, args.test_size, original_feat_names, args.transform_y)
        results_dict["rf"] = {"results": rf_results}
    
    if args.model in ["gbm", "auto"]:
        print(f"\n{'='*60}")
        print(f"GRADIENT BOOSTING (Non-linear)")
        print(f"{'='*60}")
        gbm_results = fit_gradient_boosting(X, y, args.test_size, original_feat_names, args.transform_y)
        results_dict["gbm"] = {"results": gbm_results}

    # XGBoost Block
    if args.model in ["xgb", "auto"]:
        print(f"\n{'='*60}")
        print(f"XGBOOST (Non-linear)")
        print(f"{'='*60}")
        xgb_results = fit_xgboost(X, y, args.test_size, original_feat_names, args.transform_y)
        results_dict["xgb"] = {"results": xgb_results}

    # ADDED: Neural Network Block
    if args.model in ["nn", "auto"]:
        print(f"\n{'='*60}")
        print(f"NEURAL NETWORK (NN - MLPRegressor)")
        print(f"{'='*60}")
        nn_results = fit_neural_network(X, y, args.test_size, original_feat_names, args.transform_y)
        results_dict["nn"] = {"results": nn_results}

    # Find best model
    if not results_dict:
        raise SystemExit("ERROR: No models were trained. Check the '--model' argument.")
        
    best_model = max(results_dict.items(), 
                      key=lambda x: x[1]["results"]["metrics"]["r2"])
    best_name = best_model[0]
    best_result = best_model[1]["results"]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name.upper()}")
    print(f"{'='*60}")

    # Save results for best model
    metrics = {
        "target": target_name,
        "predictors": original_feat_names,
        "n_samples": int(X.shape[0]),
        "test_size": args.test_size,
        "best_model": best_name,
        **best_result["metrics"]
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    # Save feature importance (now includes None check for NN)
    if best_result.get("feature_importance") is not None:
        imp_df = pd.DataFrame({
            "feature": original_feat_names,
            "importance": best_result["feature_importance"]
        }).sort_values("importance", ascending=False)
        imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
        
        print("\n[Feature Importance] Top predictors:")
        for _, row in imp_df.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save comparison of all models
    if len(results_dict) > 1:
        comparison = []
        for model_name, model_data in results_dict.items():
            comparison.append({
                "model": model_name,
                **model_data["results"]["metrics"]
            })
        pd.DataFrame(comparison).to_csv(out_dir / "model_comparison.csv", index=False)
        
        print(f"\n[Model Comparison]")
        for row in comparison:
            print(f"  {row['model']:12s}: R²={row['r2']:.4f}  RMSE={row['rmse']:.4f}")

    # Plots
    save_parity_plot(
        y_true=best_result["y_test"],
        y_pred=best_result["y_pred"],
        out_path=out_dir / "parity.png",
        title=f"{target_name} ({best_name}): predicted vs observed"
    )
    save_residual_hist(
        residuals=best_result["y_test"] - best_result["y_pred"],
        out_path=out_dir / "residuals.png",
        title=f"{target_name} ({best_name}): residuals"
    )
    
    try:
        write_parity_plotly(
            y_true=best_result["y_test"],
            y_pred=best_result["y_pred"],
            out_html=out_dir / "parity.html",
            title=f"{target_name} ({best_name}): predicted vs observed"
        )
        write_residuals_plotly(
            residuals=best_result["y_test"] - best_result["y_pred"],
            out_html=out_dir / "residuals.html",
            title=f"{target_name} ({best_name}): residuals"
        )
        print(f"[viz] wrote interactive parity/residuals HTML")
    except Exception as e:
        print(f"[plotly warning] Could not create interactive plots: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"[FINAL RESULTS] {target_name} - {best_name.upper()}")
    print(f"{'='*60}")
    print(f"  Samples:         {metrics['n_samples']:,}")
    print(f"  R²:              {metrics['r2']:.4f}")
    print(f"  RMSE:            {metrics['rmse']:.4f}")
    print(f"  MAE:             {metrics['mae']:.4f}")
    print(f"  Pearson r:       {metrics['pearson_r']:.4f}")
    print(f"{'='*60}")
    
    print(f"\n[output] Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()