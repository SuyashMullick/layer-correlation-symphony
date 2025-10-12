# src/geo/io.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from affine import Affine
from pyproj import CRS as PJCRS  # NEW


def _crs_equivalent(crs_a, crs_b) -> bool:
    """
    Robust CRS equality across WKT/EPSG variants:
    1) pyproj equals()
    2) same EPSG code
    3) same LAEA core proj params (proj, lat_0, lon_0, x_0, y_0, units)
    """
    try:
        A = PJCRS.from_user_input(crs_a)
        B = PJCRS.from_user_input(crs_b)

        # 1) strict equality (axes/order-normalized)
        if A == B or A.equals(B):
            return True

        # 2) EPSG code equality (covers EPSG:3035 vs WKT3035)
        ea, eb = A.to_epsg(), B.to_epsg()
        if ea is not None and eb is not None and ea == eb:
            return True

        # 3) proj param equality for LAEA flavors (tolerant compare)
        def proj_core(c: PJCRS):
            d = c.to_dict()
            # keys commonly present in 3035 definitions
            keep = ("proj", "lat_0", "lon_0", "x_0", "y_0", "units", "datum", "type")
            out = {k: d.get(k) for k in keep}
            # normalize numbers to avoid float wiggles
            for k in ("lat_0", "lon_0", "x_0", "y_0"):
                if out.get(k) is not None:
                    out[k] = round(float(out[k]), 9)
            return out

        if proj_core(A) == proj_core(B):
            return True

        return False
    except Exception:
        # last resort: try rasterio's CRS equals (in case pyproj parsing failed)
        try:
            RA = CRS.from_user_input(crs_a)
            RB = CRS.from_user_input(crs_b)
            return RA.equals(RB)
        except Exception:
            return False


def _crs_str_for_msg(crs) -> str:
    try:
        c = PJCRS.from_user_input(crs)
        epsg = c.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        return c.to_wkt()
    except Exception:
        try:
            c = CRS.from_user_input(crs)
            return c.to_wkt()
        except Exception:
            return str(crs)

def _snap_to_grid(x: float, origin: float, step: float, mode: str) -> float:
    """Snap coordinate x to grid defined by (origin, step). mode: 'ceil' or 'floor'."""
    k = (x - origin) / step
    if mode == "ceil":
        from math import ceil
        return origin + ceil(k - 1e-9) * step
    else:
        from math import floor
        return origin + floor(k + 1e-9) * step


def _to_float_and_nan(arr: np.ndarray, nodata_values: List[float]) -> np.ndarray:
    a = arr.astype("float32", copy=False)
    for nv in nodata_values:
        a[a == nv] = np.nan
    return a


def _read_window(ds, left, bottom, right, top):
    win = from_bounds(left, bottom, right, top, transform=ds.transform)
    win = win.round_offsets().round_lengths()
    arr = ds.read(1, window=win)
    # compute new transform for the window
    col_off, row_off = int(win.col_off), int(win.row_off)
    t = ds.transform
    t_win = Affine(t.a, t.b, t.c + col_off * t.a,
                   t.d, t.e, t.f + row_off * t.e)
    return arr, t_win


def read_pair_as_arrays(path_a: str, path_b: str, extra_nodata: List[float] = None
                        ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Read two rasters on a common intersection window. Returns (x, y, meta).
    x and y are 1D float32 arrays (raveled) with NaNs for nodata; caller should mask for finite.
    meta includes crs, transform, shape.
    """
    if extra_nodata is None:
        extra_nodata = []

    pa, pb = Path(path_a), Path(path_b)
    if not pa.exists():
        raise FileNotFoundError(f"Not found: {path_a}")
    if not pb.exists():
        raise FileNotFoundError(f"Not found: {path_b}")

    with rasterio.open(pa) as A, rasterio.open(pb) as B:
        # robust CRS check
        if not _crs_equivalent(A.crs, B.crs):
            raise ValueError(
                "CRS mismatch:\n"
                f"  A: {_crs_str_for_msg(A.crs)}\n"
                f"  B: {_crs_str_for_msg(B.crs)}"
            )
        ax, ay = A.res
        bx, by = B.res
        if abs(ax - bx) > 1e-6 or abs(ay - by) > 1e-6:
            raise ValueError(f"Resolution mismatch: {A.res} vs {B.res}")

        # intersection of bounds
        left = max(A.bounds.left, B.bounds.left)
        right = min(A.bounds.right, B.bounds.right)
        bottom = max(A.bounds.bottom, B.bounds.bottom)
        top = min(A.bounds.top, B.bounds.top)
        if not (left < right and bottom < top):
            raise ValueError("No spatial intersection between the two rasters.")

        # snap inward to grid of A (identical step for B by check above)
        x0, y0 = A.transform.c, A.transform.f
        left = _snap_to_grid(left, x0, ax, "ceil")
        right = _snap_to_grid(right, x0, ax, "floor")
        top = _snap_to_grid(top, y0, ay, "floor")
        bottom = _snap_to_grid(bottom, y0, ay, "ceil")

        # read both windows
        a_win, t_win = _read_window(A, left, bottom, right, top)
        b_win, _ = _read_window(B, left, bottom, right, top)

        # nodata → NaN (never treat 0 as nodata)
        a = a_win.astype("float32", copy=False)
        b = b_win.astype("float32", copy=False)

        if A.nodata is not None:
            a[a == A.nodata] = np.nan
        for nv in extra_nodata:
            if nv is not None:
                a[a == nv] = np.nan

        if B.nodata is not None:
            b[b == B.nodata] = np.nan
        for nv in extra_nodata:
            if nv is not None:
                b[b == nv] = np.nan

        # ravel to 1D
        x = a.ravel()
        y = b.ravel()

        meta = dict(
            crs=A.crs,
            transform=t_win,
            width=a_win.shape[1],
            height=a_win.shape[0],
            res=A.res,
            layer_a=pa.stem,
            layer_b=pb.stem,
        )

    return x, y, meta


def read_many_as_matrix(paths: list[str], extra_nodata: list[float] | None = None):
    """
    Read multiple rasters and return a common intersection as:
      X: (n_pixels, p_predictors)   for paths[1:]
      y: (n_pixels,)                for paths[0] (target)
    Returns X, y, meta dict.
    """
    if extra_nodata is None:
        extra_nodata = []

    if len(paths) < 2:
        raise ValueError("Provide at least one target and one predictor.")

    # Open all datasets
    dsets = []
    for p in paths:
        ds = rasterio.open(p)
        dsets.append(ds)

    try:
        # Check CRS + resolution consistency (robust CRS compare)
        ref = dsets[0]
        ref_crs = ref.crs
        ref_res = ref.res

        for ds in dsets[1:]:
            if not _crs_equivalent(ref_crs, ds.crs):
                raise ValueError(
                    "CRS mismatch among inputs:\n"
                    f"  ref: {_crs_str_for_msg(ref_crs)}\n"
                    f"  got: {_crs_str_for_msg(ds.crs)}"
                )
            if abs(ds.res[0] - ref_res[0]) > 1e-6 or abs(ds.res[1] - ref_res[1]) > 1e-6:
                raise ValueError(f"Resolution mismatch: {ref_res} vs {ds.res}")

        # Intersection of bounds across all rasters
        left = max(ds.bounds.left for ds in dsets)
        right = min(ds.bounds.right for ds in dsets)
        bottom = max(ds.bounds.bottom for ds in dsets)
        top = min(ds.bounds.top for ds in dsets)
        if not (left < right and bottom < top):
            raise ValueError("No common spatial intersection among rasters.")

        # Snap to grid of first dataset
        ax, ay = ref_res
        x0, y0 = ref.transform.c, ref.transform.f
        left = _snap_to_grid(left, x0, ax, "ceil")
        right = _snap_to_grid(right, x0, ax, "floor")
        top = _snap_to_grid(top, y0, ay, "floor")
        bottom = _snap_to_grid(bottom, y0, ay, "ceil")

        # Build window and read
        win0 = from_bounds(left, bottom, right, top, transform=ref.transform).round_offsets().round_lengths()

        arrays = []
        for ds in dsets:
            win = from_bounds(left, bottom, right, top, transform=ds.transform).round_offsets().round_lengths()
            a = ds.read(1, window=win).astype("float32", copy=False)

            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
            for nv in extra_nodata:
                if nv is not None:
                    a[a == nv] = np.nan

            arrays.append(a)

        # Stack predictors (skip first which is target)
        target = arrays[0].ravel()
        predictors = [arr.ravel() for arr in arrays[1:]]
        X = np.vstack(predictors).T  # shape (n, p)
        y = target

        meta = dict(
            crs=ref.crs,
            transform=ref.window_transform(win0),
            res=ref_res,
            width=int(win0.width),
            height=int(win0.height),
            target=Path(paths[0]).stem,
            predictors=[Path(p).stem for p in paths[1:]],
        )

        return X, y, meta

    finally:
        for ds in dsets:
            ds.close()
