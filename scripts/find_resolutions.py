import rasterio
from pathlib import Path

def inspect_rasters(base_dir="data"):
    raster_files = sorted(Path(base_dir).rglob("*.tif"))
    if not raster_files:
        print("No .tif files found.")
        return

    for f in raster_files:
        with rasterio.open(f) as src:
            print(f"\n🗺️ {f.name}")
            print(f"  CRS:         {src.crs}")
            print(f"  Resolution:  {src.res}")       # (xres, yres)
            print(f"  Size:        {src.width} x {src.height}")
            print(f"  Bounds:      {src.bounds}")
            print(f"  Dtype:       {src.dtypes[0]}")
            print(f"  NoData:      {src.nodata}")

if __name__ == "__main__":
    inspect_rasters("data")  # or your directory
