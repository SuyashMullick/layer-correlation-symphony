# C2B2 Symphony Raster Analytics MVP

A Python toolkit for **marine spatial data analytics**, built during the **Mistra C2B2 Hackathon #1**.

It enables correlation and predictive modeling across **raster layers (GeoTIFFs)** used in the **Symphony** ecosystem model by the Swedish Agency for Marine and Water Management.

---

## Features

| CLI               | Description                                                                                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `symph-correlate` | Compute **pairwise** (and stratified) **Pearson/Spearman correlations** between layers with FDR correction, heatmaps, and hexbins.                                              |
| `symph-predict`   | Predict a **target raster** from multiple **predictor layers** using Ridge or Random Forest regression, with **spatially blocked cross-validation** and explainability exports. |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/symphony-tools.git
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
# OR
.venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Setup

Before running any CLI tools, you must download and prepare the raster data.

### 1. Download datasets

Download the 2018 Symphony ecosystem and pressure layers from the Swedish Agency for Marine and Water Management:

* [Bottniska viken](https://www.havochvatten.se/download/18.3b63ec651740ce15990ccab1/1708680057942/naturvarden-och-belastningar-bottniska-viken-2018.zip)
* [Västerhavet](https://www.havochvatten.se/download/18.3b63ec651740ce15990ccab2/1708680063585/naturvarden-och-belastningar-vasterhavet-2018.zip)
* [Östersjön](https://www.havochvatten.se/download/18.3b63ec651740ce15990ccab0/1708680054290/naturvarden-.och-belastningar-ostersjon-2018.zip)

Unzip these files into `data/raw/` so the structure looks like:

```
data/
 └─ raw/
     ├─ naturvarden-och-belastningar-bottniska-viken-2018/
     ├─ naturvarden-och-belastningar-vasterhavet-2018/
     └─ naturvarden-.och-belastningar-ostersjon-2018/
```

### 2. Check alignment

You can verify if all rasters share the same grid using:

```bash
python -m scripts.check_grid_alignment \
  --ref data/raw/naturvarden-.och-belastningar-ostersjon-2018/National_eco_E_2018/01Porpoise_Baltic.tif \
  data/raw/**/*.tif
```

Optional flags:

* `--out` Save detailed CSV report
* `--debug-crs` Print CRS mismatch details

### 3. Align to reference grid

To automatically align all rasters (and copy them into `data/aligned/`), run:

```bash
python -m scripts.align_to_ref_and_copy \
  --ref data/raw/naturvarden-.och-belastningar-ostersjon-2018/National_eco_E_2018/01Porpoise_Baltic.tif \
  --in-root data/raw \
  --out-root data/aligned \
  --report out/alignment_report.csv
```

This will copy aligned files to `data/aligned/` and fix any CRS or extent mismatches.

> 💡 Tip: Choose the raster with the **largest dimensions** as the reference grid.

---

## Usage

### 🔹 1. Compare Two Layers

Compare two raster layers pixel-wise to measure linear (Pearson) and rank (Spearman) correlations.

**Usage**

```bash
python -m cli.symph-compare \
  --a <path_to_layer_a.tif> \
  --b <path_to_layer_b.tif> \
  --out <output_folder> \
  [--nodata <nodata_values>] \
  [--sample <max_points>]
```

**Key options**

| Flag         | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `--a`, `--b` | Input raster layers to compare                                |
| `--out`      | Output directory for results                                  |
| `--nodata`   | Comma-separated list of nodata sentinels (default: `0,-9999`) |
| `--sample`   | Max points to plot in scatter (default: 200k)                 |

**Example**

```bash
python -m cli.symph-compare \
  --a data/aligned/.../01Porpoise_Baltic.tif \
  --b data/aligned/.../32Nitrogen_Background.tif \
  --out out/compare/porpoise_vs_nitrogen \
  --nodata "-9999"
```

**Outputs**

```
out/compare/<name>/
 ├─ compare_summary.json
 ├─ scatter.png
 ├─ pearson_spearman.csv
 └─ logs.txt
```

> 💡 For a full list of options, run `python -m cli.symph-compare -h`

---

### 🔹 2. Predict Target from Multiple Layers

Train a model to predict one target raster from several predictor rasters.

**Usage**

```bash
python -m cli.symph-predict \
  --target <target.tif> \
  --predictors <x1.tif> <x2.tif> ... \
  --out <output_dir> \
  [--sample <n>] [--test_size <ratio>] \
  [--model {rf,gbm,xgb,nn,auto}] \
  [--remove_outliers] [--transform_y {none,log1p}]
```

**Key options**

| Flag                | Description                                     |
| ------------------- | ----------------------------------------------- |
| `--target`          | Target raster (dependent variable)              |
| `--predictors`      | Predictor rasters (independent variables)       |
| `--out`             | Output folder for results                       |
| `--sample`          | Max pixels to use (default: all)                |
| `--test_size`       | Fraction of data held out for testing           |
| `--model`           | Model type: `rf`, `gbm`, `xgb`, `nn`, or `auto` |
| `--transform_y`     | Apply transform to target variable              |
| `--remove_outliers` | Remove statistical outliers before fitting      |

**Example**

```bash
export BASE="data/aligned_and_copied/naturvarden-och-belastningar-bottniska-viken-2018"  # On macOS/Linux
# OR
$BASE = "data/aligned_and_copied/naturvarden-och-belastningar-bottniska-viken-2018"      # On Windows

python -m cli.symph-predict \
  --target "$BASE/National_eco_N_2018/01Porpoise_Baltic.tif" \
  --predictors \
    "$BASE/National_eco_N_2018/20sill_lognorm_v2.tif" \
    "$BASE/National_eco_N_2018/21skarpsill_lognorm_v2.tif" \
    "$BASE/National_eco_N_2018/19torsk_lognorm_v2.tif" \
    "$BASE/National_press_N_2018/17Noise_2000Hz_Shipping_20181122.tif" \
    "$BASE/National_press_N_2018/18Boating.tif" \
  --out "out/predict/N_porpoise_prey_noise_boating" \
  --sample 200000 \
  --transform_y log1p \
  --model rf
```

**Outputs**

```
out/predict/<name>/
 ├─ prediction.tif
 ├─ residuals.tif
 ├─ uncertainty.tif
 ├─ importance.csv / importance.png
 ├─ pdp_<feature>.png
 ├─ metrics.json
 ├─ extrapolation.tif
 └─ model_card.json
```

> 💡 For a full list of options, run `python -m cli.symph-predict -h`

---

## 📁 Repository Structure

```
symphony-tools/
├─ cli/
│  ├─ symph-correlate.py       # CLI for correlation analysis
│  └─ symph-predict.py         # CLI for predictive mapping
├─ src/
│  ├─ io_stack.py              # Raster I/O, reprojection, masking, COG writer
│  ├─ stats_corr.py            # Correlation calculations
│  ├─ viz.py                   # Heatmaps, hexbins, importance, PDP/ALE
│  ├─ model_cv.py              # Blocked CV, training, metrics, exports
│  └─ guardrails.py            # Extrapolation flags, QA checks
├─ data/                       # Input GeoTIFF layers
├─ out/                        # Generated outputs
├─ requirements.txt            # Dependencies
└─ README.md                   # This file
```

---

## 🧠 Notes

* Works with **float32** raster arrays.
* **NaN** is used to mask invalid pixels.
* Resampling: `bilinear` (continuous), `nearest` (categorical).
* Blocked CV ensures spatially realistic evaluation.
* RF uncertainty = per-tree prediction variance.

---

## 🛠️ Troubleshooting

| Issue              | Cause                      | Fix                                    |
| ------------------ | -------------------------- | -------------------------------------- |
| Blank outputs      | Incorrect NoData mask      | Check `--nodata-values`                |
| Mismatched extents | CRS/resolution differences | Use `--ref` and `--resampling` options |
| CLI not found      | Environment not activated  | `source .venv/bin/activate`            |
| High memory use    | Large rasters              | Use smaller sample or downsample       |

---

## 👥 Credits

Developed by **Team CS-LNU**
for the **Mistra C2B2 Hackathon #1 (2025)**
with support from the **Swedish Agency for Marine and Water Management (SwAM)**.

---

## 📜 License

MIT License — free to use, modify, and extend with attribution.
