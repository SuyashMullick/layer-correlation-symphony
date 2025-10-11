#!/usr/bin/env python3
"""
Check whether rasters are truly grid-aligned to a reference (same CRS, res, transform, shape).
Prints a compact table and optionally writes a CSV report.

Usage:
  python scripts/check_grid_alignment.py --ref path/to/reference.tif "data/**/*.tif" \
      --out reports/grid_alignment_report.csv [--debug-crs]

Notes:
- No reprojection is performed here; this only *checks* alignment.
- Tolerance is set so tiny floating-point wiggles don't trigger false alarms.
"""

import argparse
import csv
from pathlib import Path

import rasterio as rio
from rasterio.crs import CRS

TOL = 1e-6  # numeric tolerance for equality checks

def almost_equal(a, b, tol=TOL):
    try:
        return abs(a - b) <= tol
    except Exception:
        return False

def res_from_transform(transform):
    # transform.a = pixel width, transform.e = -pixel height
    return abs(transform.a), abs(transform.e)

def normalize_proj4(crs_obj: CRS) -> str:
    """
    Turn PROJ string into a canonical form for comparison:
      - lowercased
      - tokens sorted
      - drop fields that often vary but don't affect the projected grid
    """
    try:
        p4 = crs_obj.to_proj4() or ""
    except Exception:
        return ""
    if not p4:
        return ""
    toks = [t.strip().lower() for t in p4.strip().split() if t.strip()]
    # filter noisy tokens
    drop_prefixes = {"+type=", "+no_defs", "+towgs84="}  # add others if needed
    keep = []
    for t in toks:
        if any(t.startswith(dp) for dp in drop_prefixes):
            continue
        keep.append(t)
    return " ".join(sorted(keep))

def crs_equal(a, b, debug=False, label_a="ref", label_b="tgt"):
    """
    Compare CRS semantically:
      1) pyproj semantic equality (CRS.equals)
      2) EPSG code equality if both resolvable
      3) Normalized PROJ4 string equality
    Handles None/missing CRS gracefully.
    """
    # Handle None/empty up front
    if not a or not b:
        if debug:
            print(f"[DEBUG] CRS missing: {label_a}={a!r}, {label_b}={b!r}")
        return False

    try:
        crs_a = CRS.from_user_input(a)
        crs_b = CRS.from_user_input(b)
    except Exception as e:
        if debug:
            print(f"[DEBUG] CRS parse error: {e} | {label_a}={a!r}, {label_b}={b!r}")
        return False

    # 1) Best: semantic equality
    try:
        if crs_a.equals(crs_b):
            return True
    except Exception:
        pass

    # 2) EPSG numeric equality (good)
    try:
        ea, eb = crs_a.to_epsg(), crs_b.to_epsg()
        if ea is not None and eb is not None:
            if ea == eb:
                return True
    except Exception:
        pass

    # 3) Normalized proj4 equality (robust)
    pa = normalize_proj4(crs_a)
    pb = normalize_proj4(crs_b)
    if pa and pb and pa == pb:
        return True

    if debug:
        print("[DEBUG] CRS mismatch details:")
        try:
            print(f"  {label_a}: to_epsg={crs_a.to_epsg()} | wkt_short={crs_a.to_wkt('WKT1_GDAL')[:120]}...")
            print(f"          proj4={normalize_proj4(crs_a)}")
        except Exception:
            pass
        try:
            print(f"  {label_b}: to_epsg={crs_b.to_epsg()} | wkt_short={crs_b.to_wkt('WKT1_GDAL')[:120]}...")
            print(f"          proj4={normalize_proj4(crs_b)}")
        except Exception:
            pass
    return False

def is_aligned_same_grid(ref_ds, tgt_ds, debug_crs=False):
    """Same CRS, same resolution, same transform (origin + rotation), same shape."""
    same_crs = crs_equal(ref_ds.crs, tgt_ds.crs, debug=debug_crs,
                         label_a="ref", label_b="tgt")

    ref_res = res_from_transform(ref_ds.transform)
    tgt_res = res_from_transform(tgt_ds.transform)
    same_res = all(almost_equal(r, t) for r, t in zip(ref_res, tgt_res))

    tr_ok = (
        almost_equal(ref_ds.transform.a, tgt_ds.transform.a) and
        almost_equal(ref_ds.transform.b, tgt_ds.transform.b) and
        almost_equal(ref_ds.transform.d, tgt_ds.transform.d) and
        almost_equal(ref_ds.transform.e, tgt_ds.transform.e) and
        almost_equal(ref_ds.transform.c, tgt_ds.transform.c) and  # x origin
        almost_equal(ref_ds.transform.f, tgt_ds.transform.f)      # y origin
    )

    same_shape = (ref_ds.width == tgt_ds.width) and (ref_ds.height == tgt_ds.height)
    return same_crs, same_res, tr_ok, same_shape

def recommend_resampling(src):
    """Heuristic: integers → nearest; floats → bilinear."""
    try:
        if src.dtypes and src.dtypes[0].startswith("int"):
            return "nearest"
    except Exception:
        pass
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
    p.add_argument("--debug-crs", action="store_true", help="Print CRS debug info for mismatches.")
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
                    same_crs, same_res, same_transform, same_shape = is_aligned_same_grid(
                        ref, ds, debug_crs=args.debug_crs
                    )
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
