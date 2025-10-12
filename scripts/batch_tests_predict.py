#!/usr/bin/env python3
"""
Run symph-predict over all 5-layer predictor combinations
within the same sea directory for a given target layer.
"""

from itertools import combinations
import subprocess
import pathlib

# === CONFIG ===
base = pathlib.Path("data/aligned/naturvarden-och-belastningar-bottniska-viken-2018")
target = base / "National_eco_N_2018/01Porpoise_Baltic.tif"
num_predictors = 5                     # << set how many predictors per run
outdir = pathlib.Path("out/scan_pred_5")
sample = 50000
test_size = 0.2
model = "rf"
nodata = "-9999"

# === SCRIPT ===
outdir.mkdir(parents=True, exist_ok=True)

# collect all predictors except the target itself
all_layers = list(base.rglob("*.tif"))
predictors = [p for p in all_layers if p != target]

print(f"Found {len(predictors)} candidate predictor layers.")
print(f"Running all combinations of {num_predictors} predictors at once...")

for combo in combinations(predictors, num_predictors):
    name = "_".join(p.stem for p in combo[:num_predictors])
    out_path = outdir / name
    if out_path.exists():
        continue  # skip already done combos

    cmd = [
        "python", "-m", "cli.symph-predict",
        "--target", str(target),
        "--predictors", *map(str, combo),
        "--out", str(out_path),
        "--sample", str(sample),
        "--test_size", str(test_size),
        "--model", model,
        "--nodata", nodata,
    ]

    print(f"\n➡️ Running {name}")
    subprocess.run(cmd)
