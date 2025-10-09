# src/geo/mask.py
from __future__ import annotations
import numpy as np
from typing import List

def mask_nodata(arr: np.ndarray, nodata_values: List[float]) -> np.ndarray:
    a = arr.astype("float32", copy=False)
    for v in nodata_values:
        if v is not None:
            a[a == v] = np.nan
    return a
