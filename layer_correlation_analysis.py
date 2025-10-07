import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def list_layers(base_dir="data"):
    files = sorted(Path(base_dir).rglob("*.tif"))
    return {f.stem: f for f in files}

def _open_and_prepare(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        # Treat zeros as nodata
        arr[arr == 0] = np.nan
        profile = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }
    return arr, profile

def _match_by_partial(name_or_substring, name_to_path):
    # Try exact stem first, then substring match (case-insensitive)
    if name_or_substring in name_to_path:
        return name_to_path[name_or_substring]
    name_lower = name_or_substring.lower()
    matches = [p for stem, p in name_to_path.items() if name_lower in stem.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Ambiguous '{name_or_substring}'. Matches: "
                         + ", ".join([p.stem for p in matches]))
    else:
        raise FileNotFoundError(f"No layer matching '{name_or_substring}'")

def correlate_two_layers(layer_a, layer_b, base_dir="data", show_plots=True):
    name_to_path = list_layers(base_dir)
    pa = _match_by_partial(layer_a, name_to_path)
    pb = _match_by_partial(layer_b, name_to_path)

    A, pa_prof = _open_and_prepare(pa)
    B, pb_prof = _open_and_prepare(pb)

    # Safety checks
    same_shape = (pa_prof["width"] == pb_prof["width"]) and (pa_prof["height"] == pb_prof["height"])
    same_geo = (pa_prof["crs"] == pb_prof["crs"]) and (pa_prof["transform"] == pb_prof["transform"])
    if not same_shape:
        raise ValueError(f"Different raster shapes: {pa.stem}={A.shape}, {pb.stem}={B.shape}")
    if not same_geo:
        print("[WARN] CRS/transform differ; pixelwise correlation may be invalid.")

    # Mask to pixels valid in BOTH layers
    mask = ~np.isnan(A) & ~np.isnan(B)
    x = A[mask].ravel()
    y = B[mask].ravel()

    if x.size == 0:
        raise ValueError("No overlapping valid pixels to correlate (all NaN after masking).")

    # Correlation (Pearson)
    df = pd.DataFrame({pa.stem: x, pb.stem: y})
    corr = df.corr(method="pearson")

    if show_plots:
        # 1) Heatmap
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation")
        plt.tight_layout()
        plt.show()

        # 2) Scatter (quick sanity check)
        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, s=1, alpha=0.5)
        plt.xlabel(pa.stem)
        plt.ylabel(pb.stem)
        plt.title("Pixel-wise Scatter")
        plt.tight_layout()
        plt.show()

    return corr, df  # df is the paired, valid pixels

if __name__ == "__main__":
    # e.g. "01Porpoise_Baltic" or "Porpoise_Baltic"
    corr, df = correlate_two_layers("01Porpoise_Baltic", "02Porpoise_Beltsea", base_dir="data")
    print(corr)
