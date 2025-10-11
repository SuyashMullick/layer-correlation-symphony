#!/usr/bin/env python3
"""
Check whether rasters are truly grid-aligned to a reference (same CRS, res, transform, shape).
Prints a compact table and optionally writes a CSV report.

Usage:
  python scripts/check_grid_alignment.py --ref path/to/reference.tif data/**/*.tif \
      --out reports/grid_alignment_report.csv

Notes:
- No reprojection is performed here; this only *checks* alignment.
- Tolerance is set so tiny floating-point wiggles don't trigger false alarms.
- CRS is compared by EPSG code if available (prevents false mismatches like
  "ETRS89 / LAEA Europe" vs "ETRS89-extended / LAEA Europe").
"""

import argparse
import csv
from pathlib import Path
import rasterio as rio
from rasterio.crs import CRS

TOL = 1e-6  # numeric tolerance for equality checks

def almost_equal(a, b, tol=TOL):
    return abs(a - b) <= tol

def res_from_transform(transform):
    # transform = [a, b, c, d, e, f] → a = pixel width, e = -pixel height
    return abs(transform.a), abs(transform.e)

def crs_equal(a, b):
    """
    Compare CRS semantically.
    - Prefer pyproj semantic equality (handles WKT vs EPSG tag).
    - Fall back to EPSG code compare if available.
    - Final fallback compares PROJ strings in a normalized way.
    """
    try:
        crs_a = CRS.from_user_input(a)
        crs_b = CRS.from_user_input(b)

        # 1) semantic equality (best)
        if crs_a.equals(crs_b):   # pyproj semantic compare
            return True

        # 2) EPSG code equality (good)
        ea, eb = crs_a.to_epsg(), crs_b.to_epsg()
        if ea is not None and eb is not None:
            return ea == eb

        # 3) PROJ string equality (last resort; robust to param ordering)
        pa = crs_a.to_proj4() or ""
        pb = crs_b.to_proj4() or ""
        return pa.strip() == pb.strip()
    except Exception:
        return False

def is_aligned_same_grid(ref, tgt):
    """Same CRS, same resolution, same transform (origin + rotation), same shape."""
    same_crs = crs_equal(ref.crs, tgt.crs)

    ref_res = res_from_transform(ref.transform)
    tgt_res = res_from_transform(tgt.transform)
    same_res = all(almost_equal(r, t) for r, t in zip(ref_res, tgt_res))

    # Same transform (including origin). Allow tiny float noise.
    tr_ok = (
        almost_equal(ref.transform.a, tgt.transform.a)
        and almost_equal(ref.transform.b, tgt.transform.b)
        and almost_equal(ref.transform.d, tgt.transform.d)
        and almost_equal(ref.transform.e, tgt.transform.e)
        and almost_equal(ref.transform.c, tgt.transform.c)  # x origin
        and almost_equal(ref.transform.f, tgt.transform.f)  # y origin
    )

    same_shape = (ref.width == tgt.width) and (ref.height == tgt.height)
    return same_crs, same_res, tr_ok, same_shape

def recommend_resampling(src):
    """Heuristic: integers → nearest; floats → bilinear."""
    if src.dtypes and src.dtypes[0].startswith("int"):
        return "nearest"
    return "bilinear"

def bounds_from(ds):
    left = ds.transform.c
    top = ds.transform.f
    xres = ds.transform.a
    yres = -ds.transform.e
    right = left + ds.width * xres
    bottom = top - ds.height * yres
    return (left, bottom, right, top)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Path to reference raster (the grid to align to).")
    p.add_argument("rasters", nargs="+", help="Rasters to check (glob patterns ok).")
    p.add_argument("--out", default=None, help="Optional CSV path for a full report.")
    args = p.parse_args()

    # Expand any globs
    files = []
    for pattern in args.rasters:
        files.extend(Path().glob(pattern))
    files = [Path(f) for f in files if f.suffix.lower() in (".tif", ".tiff")]
    files = sorted(set(files))

    if not files:
        print("No rasters matched.")
        return

    with rio.open(args.ref) as ref:
        ref_info = {
            "crs": ref.crs,
            "res": res_from_transform(ref.transform),
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "bounds": bounds_from(ref),
        }

        rows = []
        print("\nReference grid:")
        print(f"  {args.ref}")
        print(f"  CRS={ref_info['crs']}, res={ref_info['res']}, size={ref.width}x{ref.height}")
        print(f"  bounds={tuple(round(v, 3) for v in ref_info['bounds'])}\n")

        print("Check results (✓ aligned / ✗ needs alignment):")
        header = f"{'Raster':60} {'CRS':5} {'Res':5} {'Transform':9} {'Shape':6} {'Action'}"
        print(header)
        print("-" * len(header))

        for f in files:
            try:
                with rio.open(f) as ds:
                    same_crs, same_res, same_transform, same_shape = is_aligned_same_grid(ref, ds)
                    aligned = all([same_crs, same_res, same_transform, same_shape])
                    action = "OK (aligned)" if aligned else "ALIGN → resample: " + recommend_resampling(ds)
                    print(f"{str(f)[:60]:60} "
                          f"{'✓' if same_crs else '✗':5} "
                          f"{'✓' if same_res else '✗':5} "
                          f"{'✓' if same_transform else '✗':9} "
                          f"{'✓' if same_shape else '✗':6} "
                          f"{action}")

                    rows.append({
                        "path": str(f),
                        "crs_match": same_crs,
                        "res_match": same_res,
                        "transform_match": same_transform,
                        "shape_match": same_shape,
                        "width": ds.width,
                        "height": ds.height,
                        "xres": res_from_transform(ds.transform)[0],
                        "yres": res_from_transform(ds.transform)[1],
                        "bounds_left": bounds_from(ds)[0],
                        "bounds_bottom": bounds_from(ds)[1],
                        "bounds_right": bounds_from(ds)[2],
                        "bounds_top": bounds_from(ds)[3],
                        "dtype": ds.dtypes[0],
                        "nodata": ds.nodata,
                        "suggested_resampling": recommend_resampling(ds),
                        "needs_alignment": not aligned
                    })
            except Exception as e:
                print(f"{str(f)[:60]:60} ERROR {e}")
                rows.append({"path": str(f), "error": str(e)})

        if args.out and rows:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nWrote CSV report → {args.out}")

if __name__ == "__main__":
    main()
