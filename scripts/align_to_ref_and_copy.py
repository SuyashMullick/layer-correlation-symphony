#!/usr/bin/env python3
"""
Mirror an input tree of rasters into an output tree that is fully aligned
to a reference grid. Already-aligned rasters are copied as-is; misaligned
rasters are resampled/reprojected to match the reference's CRS/transform/shape.

Usage:
  python scripts/align_or_copy_to_dir.py \
    --ref path/to/reference.tif \
    --in-root data \
    --out-root data/aligned \
    --report reports/alignment_actions.csv \
    [--overwrite] [--debug-crs]

Notes:
- Preserves directory structure relative to --in-root.
- Reprojection uses 'nearest' for integer rasters, 'bilinear' for floats.
- Writes Cloud-Optimized-ish GeoTIFFs (tiled + LZW).
"""

import argparse
import csv
from pathlib import Path
import shutil

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.warp import reproject

TOL = 1e-6

def almost_equal(a, b, tol=TOL):
    try:
        return abs(a - b) <= tol
    except Exception:
        return False

def res_from_transform(transform):
    return abs(transform.a), abs(transform.e)

def normalize_proj4(crs_obj: CRS) -> str:
    try:
        p4 = crs_obj.to_proj4() or ""
    except Exception:
        return ""
    if not p4:
        return ""
    toks = [t.strip().lower() for t in p4.strip().split() if t.strip()]
    drop_prefixes = {"+type=", "+no_defs", "+towgs84="}
    keep = []
    for t in toks:
        if any(t.startswith(dp) for dp in drop_prefixes):
            continue
        keep.append(t)
    return " ".join(sorted(keep))

def crs_equal(a, b, debug=False, label_a="ref", label_b="tgt"):
    if not a or not b:
        if debug:
            print(f"[DEBUG] CRS missing: {label_a}={a!r}, {label_b}={b!r}")
        return False
    try:
        ca = CRS.from_user_input(a); cb = CRS.from_user_input(b)
    except Exception as e:
        if debug:
            print(f"[DEBUG] CRS parse error: {e} | {label_a}={a!r}, {label_b}={b!r}")
        return False
    try:
        if ca.equals(cb):
            return True
    except Exception:
        pass
    try:
        ea, eb = ca.to_epsg(), cb.to_epsg()
        if ea is not None and eb is not None and ea == eb:
            return True
    except Exception:
        pass
    pa, pb = normalize_proj4(ca), normalize_proj4(cb)
    if pa and pb and pa == pb:
        return True
    if debug:
        print("[DEBUG] CRS mismatch:")
        try:
            print(f"  {label_a}: epsg={ca.to_epsg()} proj4={normalize_proj4(ca)}")
            print(f"  {label_b}: epsg={cb.to_epsg()} proj4={normalize_proj4(cb)}")
        except Exception:
            pass
    return False

def is_aligned_same_grid(ref_ds, tgt_ds, debug_crs=False):
    same_crs = crs_equal(ref_ds.crs, tgt_ds.crs, debug=debug_crs, label_a="ref", label_b="tgt")
    ref_res = res_from_transform(ref_ds.transform)
    tgt_res = res_from_transform(tgt_ds.transform)
    same_res = all(almost_equal(r, t) for r, t in zip(ref_res, tgt_res))
    tr_ok = (
        almost_equal(ref_ds.transform.a, tgt_ds.transform.a) and
        almost_equal(ref_ds.transform.b, tgt_ds.transform.b) and
        almost_equal(ref_ds.transform.d, tgt_ds.transform.d) and
        almost_equal(ref_ds.transform.e, tgt_ds.transform.e) and
        almost_equal(ref_ds.transform.c, tgt_ds.transform.c) and
        almost_equal(ref_ds.transform.f, tgt_ds.transform.f)
    )
    same_shape = (ref_ds.width == tgt_ds.width) and (ref_ds.height == tgt_ds.height)
    return same_crs, same_res, tr_ok, same_shape

def recommend_resampling(src):
    try:
        if src.dtypes and src.dtypes[0].startswith("int"):
            return Resampling.nearest
    except Exception:
        pass
    return Resampling.bilinear

def reproject_to_match_grid(src_ds, ref_ds, out_path, resampling):
    """Reproject/resample src_ds into the exact grid of ref_ds and write GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dst_crs = ref_ds.crs
    dst_transform = ref_ds.transform
    dst_width, dst_height = ref_ds.width, ref_ds.height
    count = src_ds.count
    dtype = src_ds.dtypes[0]
    nodata = src_ds.nodata

    # Create destination dataset
    profile = src_ds.profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "transform": dst_transform,
        "crs": dst_crs,
        "dtype": dtype,
        "count": count,
        "nodata": nodata,
        # good defaults
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "LZW",
        "predictor": 2 if np.dtype(dtype).kind in ("f",) else 1,
        "BIGTIFF": "IF_SAFER",
    })

    with rio.open(out_path, "w", **profile) as dst:
        for b in range(1, count + 1):
            src = src_ds.read(b)
            dst_arr = np.full((dst_height, dst_width), nodata, dtype=dtype) if nodata is not None else np.zeros((dst_height, dst_width), dtype=dtype)

            reproject(
                source=src,
                destination=dst_arr,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=nodata,
                dst_nodata=nodata,
                num_threads=0,  # let GDAL pick
            )
            dst.write(dst_arr, b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference raster to define target grid.")
    ap.add_argument("--in-root", required=True, help="Root directory to scan for rasters (recursively).")
    ap.add_argument("--out-root", default="data/aligned", help="Where to mirror aligned/copy of rasters.")
    ap.add_argument("--report", default=None, help="Optional CSV action log.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")
    ap.add_argument("--debug-crs", action="store_true", help="Print CRS debug info for mismatches.")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted([p for p in in_root.rglob("*") if p.suffix.lower() in (".tif", ".tiff")])

    if not tif_paths:
        print(f"No rasters found under {in_root}")
        return

    with rio.open(args.ref) as ref_ds:
        print("Reference grid:")
        print(f"  {args.ref}")
        print(f"  CRS={ref_ds.crs}, res={res_from_transform(ref_ds.transform)}, size={ref_ds.width}x{ref_ds.height}\n")

        rows = []
        for src_path in tif_paths:
            rel = src_path.relative_to(in_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with rio.open(src_path) as ds:
                    same_crs, same_res, same_transform, same_shape = is_aligned_same_grid(ref_ds, ds, debug_crs=args.debug_crs)
                    aligned = all([same_crs, same_res, same_transform, same_shape])

                    if aligned:
                        if out_path.exists() and not args.overwrite:
                            action = "skipped(copy exists)"
                        else:
                            shutil.copy2(src_path, out_path)
                            action = "copied(aligned)"
                    else:
                        if out_path.exists() and not args.overwrite:
                            action = "skipped(align exists)"
                        else:
                            resamp = recommend_resampling(ds)
                            reproject_to_match_grid(ds, ref_ds, out_path, resamp)
                            action = f"aligned(resample={resamp.name})"

                    print(f"{rel} → {out_path.relative_to(out_root)} : {action}")
                    rows.append({
                        "src": str(src_path),
                        "dst": str(out_path),
                        "crs_match": same_crs,
                        "res_match": same_res,
                        "transform_match": same_transform,
                        "shape_match": same_shape,
                        "action": action,
                        "dtype": ds.dtypes[0],
                        "nodata": ds.nodata,
                    })

            except Exception as e:
                print(f"{rel} : ERROR {e}")
                rows.append({"src": str(src_path), "dst": str(out_path), "error": str(e)})

        if args.report and rows:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nWrote report → {args.report}")

if __name__ == "__main__":
    main()
