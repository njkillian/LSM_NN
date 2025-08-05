# local_downsample.py
# 
# This script processes LSM main folders, each containing subfolders:
#    Ex_488_Ch1_stitched (as "gfp")
#    Ex_561_Ch2_stitched (as "rfp")
#    Ex_639_Ch3_stitched (as "cfp")
#
# Steps:
#   1) Gather .tif files in each subfolder (median background subtraction).
#   2) Downsample each dimension by 8 (Z, Y, X).
#   3) Stack => (C, Z, Y, X), rechunk, save to "down_<foldername>.zarr".
#   4) Use a single Dask worker to avoid concurrency issues on network paths.
#
# If you see "Sending large graph" warnings from Dask, it's typically harmless.
#
# NKillian 2025

import os
import sys
import platform
import numpy as np
from skimage.io import imread
from dask import delayed
import dask.array as da
from dask.distributed import Client, LocalCluster

##############################################################################
# Main folders on drive Y: (mapped from UNC)
##############################################################################
MAIN_FOLDERS = [
    r"Y:\2024_11_27\20241127_10_23_04_1SB28_TRAP594Sox3GFPCola1647_Destripe_DONE", # day 8
    r"Y:\2024_11_24\20241124_00_54_37_SB28Day15_Destripe_DONE", # day 15
    r"Y:\2024_11_27\20241127_14_29_13_SB28_TRAP594Sox2GFPCola1647_Destripe_DONE", # late stage
]

# Subfolders -> channel name
SUBFOLDERS = [
    ("gfp", "Ex_488_Ch1_stitched"),
    ("rfp", "Ex_561_Ch2_stitched"),
    ("cfp", "Ex_639_Ch3_stitched"),
]

def windows_long_path(path_str: str) -> str:
    """
    Prepend '\\\\?\\' to handle paths > 260 chars on Windows,
    requiring 'LongPathsEnabled' in the registry.
    """
    if platform.system().lower().startswith("win"):
        abs_p = os.path.abspath(path_str)
        # Check if it already starts with '\\?\'
        if not abs_p.startswith('\\\\?\\'):
            abs_p = '\\\\?\\' + abs_p
        return abs_p
    else:
        return path_str  # No change on non-Windows

def import_process(main_folder: str, channel_mapping) -> dict:
    """
    1) Gather .tif files in subfolders (channel_name, subfolder_name).
    2) Build Dask arrays => (Z, Y, X).
    3) Median-subtract across channels => return {chan: dask_array}.
    """
    file_dict = {}

    for chan_name, subf in channel_mapping:
        folder_path = os.path.join(main_folder, subf)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Subfolder not found: {folder_path}")

        # Gather all .tif
        all_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".tif"))
        if not all_files:
            raise FileNotFoundError(f"No TIFF found in {folder_path}")

        # Convert each file path to long-path form
        full_paths = [windows_long_path(os.path.join(folder_path, f)) for f in all_files]
        file_dict[chan_name] = full_paths

    print("Gathered filenames. Building Dask arrays...")

    # Sample shape/dtype from first file in the first channel
    first_channel = list(file_dict.keys())[0]
    sample_file = file_dict[first_channel][0]
    sample_im = imread(sample_file)

    lazy_imread = delayed(imread)
    image_d = {}
    for chan, fpaths in file_dict.items():
        lazy_arrays = [lazy_imread(fp) for fp in fpaths]
        dask_arrays = [
            da.from_delayed(im, shape=sample_im.shape, dtype=sample_im.dtype)
            for im in lazy_arrays
        ]
        image_d[chan] = da.stack(dask_arrays, axis=0)  # => (Z, Y, X)

    # Median across channels => shape (C, Z, Y, X)
    all_chan = da.stack([image_d[ch] for ch in image_d], axis=0)
    bg_median = da.median(all_chan, axis=0)  # => (Z, Y, X)

    # subtract median
    subtracted = {}
    for chan in image_d:
        subtracted[chan] = image_d[chan] - bg_median

    return subtracted

def process_folder(main_folder: str):
    """
    Downsample data from main_folder, save .zarr to current dir.
    """
    print(f"\n=== Processing folder: {main_folder} ===")

    sb_import = import_process(main_folder, SUBFOLDERS)
    print("Import / background subtraction successful.")

    # Downsample by 8 in Z, Y, X
    down_d = {}
    for chan, arr in sb_import.items():
        down_d[chan] = arr[::8, ::8, ::8]  # => (Z, Y, X)

    # Stack => (C, Z, Y, X)
    down_ = da.stack(list(down_d.values()), axis=0)
    # Rechunk
    down_ = down_.rechunk((3, 250, 500, 500))

    folder_name = os.path.basename(os.path.normpath(main_folder))
    out_zarr = f"down_{folder_name}.zarr"
    print(f"Saving Zarr => {out_zarr} ...")

    down_.to_zarr(out_zarr, overwrite=True)
    print(f"Downscaling & saving completed: {out_zarr}")

def main():
    # Single worker to avoid concurrency
    total_cores = os.cpu_count() or 1
    n_workers = 1
    print(f"Creating local Dask cluster with {n_workers} worker (of {total_cores} cores).")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    print("Dask dashboard link:", client.dashboard_link)

    for fldr in MAIN_FOLDERS:
        if not os.path.isdir(fldr):
            print(f"WARNING: main folder not found => {fldr}")
            continue

        try:
            process_folder(fldr)
        except Exception as exc:
            print(f"ERROR processing folder {fldr}: {exc}")

    client.close()
    cluster.close()
    print("\nAll done.")

if __name__ == "__main__":
    main()
