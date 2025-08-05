"""
tifFromZarr.py

Converts zarr volumes to TIFF format with auto-contrast and rotation, if needed.
NKillian 2025
"""

import os
import numpy as np
import dask.array as da
import tifffile
from tifffile import TiffWriter
from skimage import transform

##############################################################################
# CONFIG
##############################################################################

# Files and z-ranges of volume
FOLDERS_AND_RANGES = {
    "down_20241127_10_23_04_1SB28_TRAP594Sox3GFPCola1647_Destripe_DONE.zarr": (
        (1, 277),
        "day8",
    ),
    "down_20241124_00_54_37_SB28Day15_Destripe_DONE.zarr": (
         (64, 450),
         "day15",
    ),
    "down_20241127_14_29_13_SB28_TRAP594Sox2GFPCola1647_Destripe_DONE.zarr": (
        (1, 485),
        "late2",
    ),
}

LP = 90  # Lower percentile for auto-contrast
UP = 99  # Upper percentile for auto-contrast
MAX_GB = 32  # Maximum GB for single-file TIFF

##############################################################################
# Helper Functions
##############################################################################

def estimate_size_gb(darr):
    """Estimate the size of a dask array in GB."""
    shape = darr.shape
    dtype = darr.dtype
    nelems = np.prod(shape)
    return nelems * np.dtype(dtype).itemsize / (1024**3)

def auto_contrast_8bit(darr_channel, lower_percentile=90, upper_percentile=99):
    """Apply auto-contrast and convert to 8-bit."""
    arr_local = darr_channel.compute()
    p_low, p_high = np.percentile(arr_local, [lower_percentile, upper_percentile])
    if p_low >= p_high:
        p_high = p_low + 1
    arr_clipped = np.clip(arr_local, p_low, p_high)
    arr_scaled = (arr_clipped - p_low) / (p_high - p_low)
    arr_byte = (arr_scaled * 255).astype("uint8")
    return arr_byte

def rotate_volume_2d(arr_4d, angle_degs=170):
    """Rotate each Z-slice of a 4D volume (C,Z,Y,X) by the specified angle."""
    angle = abs(angle_degs)
    c, z, y, x = arr_4d.shape
    arr_local = arr_4d.compute()
    out_local = np.zeros_like(arr_local, dtype=arr_local.dtype)
    for cc in range(c):
        for zz in range(z):
            out_local[cc, zz] = transform.rotate(
                arr_local[cc, zz],
                angle=angle,
                preserve_range=True,
                resize=False
            )
    return da.from_array(out_local, chunks=arr_4d.chunksize)

##############################################################################
# Main Processing Function
##############################################################################

def process_zarr(
    zarr_path,
    zlow,
    zhigh,
    shorthand,
    lp=90,
    up=99
):
    """Process a zarr file and save as TIFF."""
    out_tif = f"{shorthand}.tif"
    print(f"\n=== Processing {zarr_path} => {out_tif} ===")
    print(f"  Z-range=[{zlow}:{zhigh})")

    # (1) Load zarr => shape (C,Z,Y,X)
    arr = da.from_zarr(zarr_path)
    print(f"  Original shape: {arr.shape}, dtype={arr.dtype}")

    # (2) Slice Z dimension
    arr = arr[:, zlow:zhigh, :, :]
    c, z, y, x = arr.shape
    print(f"  After Z slicing => shape: {arr.shape} => (C,Z,Y,X)")
    print(f"  => Y={y}, X={x}")

    # (3) Apply rotation for specific datasets
    if "down_20241127_10_23_04_1SB28_TRAP594Sox3GFPCola1647_Destripe_DONE.zarr" in zarr_path:
        print("  Rotating day8 => 170 deg ccw.")
        arr = rotate_volume_2d(arr, 170)
    elif "down_20241127_14_29_13_SB28_TRAP594Sox2GFPCola1647_Destripe_DONE.zarr" in zarr_path:
        print("  Rotating late2 => 180 deg ccw.")
        arr = rotate_volume_2d(arr, 180)

    # (4) Extract channels: arr[0]=Green, arr[1]=Red, arr[2]=Blue
    c_green = arr[0]
    c_red   = arr[1]
    c_blue  = arr[2]

    # (5) Apply auto-contrast to convert to 8-bit
    print(f"  Auto-contrast => p{lp}-{up}")
    red_8   = auto_contrast_8bit(c_red, lp, up)
    green_8 = auto_contrast_8bit(c_green, lp, up)
    blue_8  = auto_contrast_8bit(c_blue, lp, up)
    
    # Stack channels back together => (3,Z,Y,X)
    arr_byte = da.stack([red_8, green_8, blue_8], axis=0)

    # (6) Reshape for ImageJ format: (3,Z,Y,X) => (1,Z,3,Y,X)
    arr_5d = arr_byte[None, ...]  # Add time dimension
    arr_5d = arr_5d.transpose(0, 2, 1, 3, 4)  # => (T=1,Z,C=3,Y,X)

    T, Z, C, Y, X = arr_5d.shape
    size_gb = estimate_size_gb(arr_5d)
    print(f"  Final shape => (T,Z,C,Y,X)={arr_5d.shape}")
    print(f"  Estimate ~{size_gb:.2f} GB in memory.")

    # ImageJ metadata
    ij_metadata = {
        "axes": "TZCYX",
        "hyperstack": True,
        "mode": "composite",
        "channels": C,
        "slices": Z,
        "frames": T,
    }

    # (7) Save as TIFF
    if size_gb <= MAX_GB:
        print(f"  -> Writing single-file TIFF => {out_tif}")
        arr_np = arr_5d.compute()
        tifffile.imwrite(
            out_tif,
            arr_np,
            imagej=True,
            metadata=ij_metadata
        )
    else:
        print(f"  -> Data ~{size_gb:.2f} GB, writing multi-slice TIFF.")
        with TiffWriter(out_tif, bigtiff=True) as tif:
            for z_idx in range(Z):
                slice_data = arr_5d[:, z_idx:z_idx+1, :, :, :].compute()
                slice_data = slice_data.reshape(T, C, Y, X)
                for t_idx in range(T):
                    tif.write(
                        slice_data[t_idx],
                        photometric="minisblack",
                        metadata={"axes": "CYX"}
                    )

    print(f"  Done => {out_tif}\n")

##############################################################################
# Main Entry Point
##############################################################################

def main():
    """Process all configured zarr files."""
    print("Starting batch processing...\n")
    
    for zarr_path, (zrange, shorthand) in FOLDERS_AND_RANGES.items():
        zlow, zhigh = zrange
        
        if not os.path.exists(zarr_path):
            print(f"WARNING: {zarr_path} not found, skipping.")
            continue

        process_zarr(
            zarr_path=zarr_path,
            zlow=zlow,
            zhigh=zhigh,
            shorthand=shorthand,
            lp=LP,
            up=UP
        )
    
    print("\nAll done.\n")


if __name__ == "__main__":
    main()