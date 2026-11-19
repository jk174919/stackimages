# Raster Band Stack + Pyramids

A local Streamlit app for batch-stacking raster bands **2, 3, and 4** into three-band GeoTIFFs and building overview pyramids.

## What it does

- Scans one or more selected folders, optionally including their subfolders.
- Groups files in the same folder when their names match apart from a recognized band marker, such as `B2`, `B02`, `BAND_2`, or `Band2`.
- Writes each stack beside its source bands using the folder name, such as `R2F01NOV2026080619009400045SSANSTUC00GTDB.tif`.
- Keeps stack order fixed as Band 2, Band 3, Band 4.
- Uses Band 2's grid as the output grid and aligns Bands 3 and 4 when their georeferencing differs.
- Builds internal GeoTIFF overviews at the selected pyramid levels.
- Uses larger buffered writes and GDAL multithreading to speed up stacking and overview generation.
- Skips an existing folder-named output unless overwrite is enabled.

Incomplete groups and groups with multiple files for the same band are shown for review and not processed. If one folder contains multiple complete band sets, the app flags the filename conflict so the sets cannot overwrite one another.

## Run on Windows

Use Python 3.10 or newer. Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

For double-click setup, run `install_windows.bat` once and then `run_windows.bat`. The installer creates a local `.venv` and installs the packages from `requirements.txt`.

If you use Conda, create the environment with `conda env create -f environment.yml`, then start the app with `conda run -n raster-band-stack-pyramids streamlit run app.py`.

The app opens in your browser. Click **Choose folders** and select a common parent once; a checkbox list opens with its source folders selected. Check or uncheck folders, then click **Use selected folders**. If your folders are in separate locations, paste all their paths into the source-folder box, one per line. The picker requires running the app on the same PC that stores the imagery. Each stack is written beside its matching source bands, including when several folders are processed in one batch.

## Options

- **Also scan subfolders**: recursively discover matching rasters.
- **Resampling if grids differ**: choose nearest, bilinear, or cubic resampling for alignment to Band 2.
- **Pyramid resampling**: choose how overview pixels are calculated.
- **Overview pyramid levels**: select reduction factors from 2× through 32×.
- **Processing threads**: use all CPU cores or limit GDAL to 4, 2, or 1 worker threads.
- **GeoTIFF compression**: choose a codec or disable compression for faster writes and larger files. ZSTD can be faster and smaller, but older GIS software may not support it.
- **Output NoData override**: optionally set a common NoData value for the stack.
- **Overwrite**: replace a stack that already exists.

Band filenames should include a recognizable band token, for example:

```text
LC08_scene_B2.tif
LC08_scene_B3.tif
LC08_scene_B4.tif
```

## Supported raster formats

The scanner looks for GeoTIFF (`.tif`, `.tiff`), JPEG 2000 (`.jp2`), ERDAS Imagine (`.img`), VRT, and common GDAL raster formats (`.dat`, `.bil`, `.asc`). Actual format support depends on the Rasterio/GDAL build installed with your Python environment.

