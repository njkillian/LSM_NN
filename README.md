# Zarr → TIFF → Visualization Pipeline

## Quick Steps

1. **Create Zarr**: `python local_downsample.py`
   - Downsamples raw data to `.zarr` format

2. **Convert to TIFF**: `python tifFromZarr.py`
   - Zarr → TIFF with auto-contrast

3. **Make Videos**: `python allVideos.py`
   - Creates rotating volume renderings

4. **View & Screenshot**: `python tifViewer.py`
   - Interactive viewer for captures of rendering views

## Flow
```
Raw Data → Zarr → TIFF → Videos/Screenshots
```
