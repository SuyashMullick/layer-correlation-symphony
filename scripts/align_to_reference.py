#!/usr/bin/env python3
"""
Resample rasters to exactly match a reference raster's grid
(CRS, transform, resolution, width, height).

Usage examples
--------------
# Use the CSV from check_grid_alignment.py (only rows with needs_alignment=True)
python scripts/align_to_reference.py --ref path/to/ref.tif \
    --report reports/grid_alignment_report.csv \
    --outdir data_aligned

# Or give globs directly (all will be aligned)
python scripts/align_to_reference.py --ref path/to/ref.tif \
    "data/**/*.tif" "more/*.tif" \
    --outdir data_aligned

Notes
-----
- Resampling is chosen from source dtype: integers → nearest, floats → bilinear.
  You can override via --method nearest|bilinear|cubic|average.
- NODATA is preserved from the source file when present.
- Output keeps source dtype, but uses the reference grid.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject

METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}

def pick_method(dtype: str) -> Resampling:
    return Resampling.nearest if dtype.startswith(("int", "uint")) else Resampling.bilinear

def list_from_report(report_csv: Path) -> List[Path]:
    out: List[Path] = []
    with open(report_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("error"):
                continue
            needs = str(row.get("needs_alignment", "")).strip().lower() in ("true", "1", "yes")
            if needs:
                out.append(Path(row["path"]))
    return out

def expand_globs(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path().glob(pat))
    # keep only tifs
    files = [p for p in files if p.suffix.lower() in (".tif", ".tiff")]
    # unique, sorted
    return sorted(set(files))

def align_one(src_path: Path, ref_ds: rio.DatasetReader, outdir: Path,
              method_override: Optional[str] = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (src_path.stem + "_aligned.tif")

    with rio.open(src_path) as src:
        # Choose resampling
        method = METHODS.get(method_override, pick_method(src.dtypes[0]))

        # Build destination profile from reference grid, but with source dtype/bands/nodata
        dst_profile = ref_ds.profile.copy()
        dst_profile.update(
            driver="GTiff",
            dtype=src.dtypes[0],
            count=src.count,
            nodata=src.nodata,  # preserve source nodata if present
            compress="LZW",
            tiled=True,
            blockxsize=min(512, ref_ds.width),
            blockysize=min(512, ref_ds.height),
            BIGTIFF="IF_SAFER",
        )

        # Create destination and reproject band-by-band
        with rio.open(out_path, "w", **dst_profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, b),
                    destination=rio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=ref_ds.transform,
                    dst_crs=ref_ds.crs,
                    dst_nodata=src.nodata,
                    resampling=method,
                )

    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference raster path (defines target grid).")
    ap.add_argument("--report", help="CSV from check_grid_alignment.py to pick files needing alignment.")
    ap.add_argument("--method", choices=tuple(METHODS.keys()),
                    help="Force resampling method (overrides dtype-based default).")
    ap.add_argument("--outdir", default="data_aligned", help="Where to write aligned rasters.")
    ap.add_argument("rasters", nargs="*", help="Optional glob patterns for rasters to align.")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # Gather input files
    files: List[Path] = []
    if args.report:
        files.extend(list_from_report(Path(args.report)))
    if args.rasters:
        files.extend(expand_globs(args.rasters))
    # unique
    files = sorted(set(files))

    if not files:
        print("No rasters to align. Provide --report and/or raster globs.")
        return

    with rio.open(args.ref) as ref_ds:
        print(f"Reference grid: {args.ref}")
        print(f"  CRS={ref_ds.crs}  res=({ref_ds.transform.a}, {abs(ref_ds.transform.e)})  size={ref_ds.width}x{ref_ds.height}")
        print(f"Writing aligned rasters to: {outdir}\n")

        for i, f in enumerate(files, 1):
            try:
                out_path = align_one(f, ref_ds, outdir, args.method)
                print(f"[{i}/{len(files)}] ✓ {f}  →  {out_path.name}")
            except Exception as e:
                print(f"[{i}/{len(files)}] ✗ {f}  ERROR: {e}")

if __name__ == "__main__":
    main()
