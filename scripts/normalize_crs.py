# scripts/normalize_crs.py
from pathlib import Path
import rasterio as rio
from rasterio.crs import CRS
import shutil

def normalize_crs(in_path, out_path):
    with rio.open(in_path) as src:
        profile = src.profile.copy()
        profile.update(crs=CRS.from_epsg(3035))  # set tag to EPSG form only

        # write a pixel-identical copy, just with a clean CRS tag
        with rio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)

if __name__ == "__main__":
    import sys
    ip, op = sys.argv[1], sys.argv[2]
    normalize_crs(ip, op)
