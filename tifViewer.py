"""
tifViewer.py

LSM Volume View Screenshots
For visualization, not analysis
NKillian 2025
"""

import os
import sys
import napari
import tifffile
import numpy as np
from qtpy.QtCore import QTimer
from PIL import Image

################################################################################
# CONFIG: width & height for all images:
################################################################################

WIDTH = 900   # total pixels in X dimension
HEIGHT = 900  # total pixels in Y dimension

################################################################################
# FILES
################################################################################

files = [
    {
        "filename": "day8.tif",
        "shorthand": "day8",
        "x_origin": 504,
        "y_origin": 861,
        "contrast_red": (175, 255),
        "contrast_green": (240, 255),
        "opacity_red": 0.65,
        "opacity_green": 0.4,
    },
    {
        "filename": "day15.tif",
        "shorthand": "day15",
        "x_origin": 466,
        "y_origin": 657,
        "contrast_red": (150, 255),
        "contrast_green": (225, 255),
        "opacity_red": 0.65,
        "opacity_green": 0.4,
    },
    {
        "filename": "late2.tif",
        "shorthand": "late2",
        "x_origin": 422,
        "y_origin": 550,
        "contrast_red": (200, 255),
        "contrast_green": (225, 255),
        "opacity_red": 0.65,
        "opacity_green": 0.4,
    },
]


def load_and_crop(tif_path, x_origin, y_origin, width, height):
    """
    1) Load 4D/5D TIF => (Z, C, Y, X). If T>1, use T=0.
    2) Reorder to (C, Z, Y, X).
    3) Crop around (x_origin, y_origin), size=(width, height).
    4) If the crop extends beyond the image, pad with zeros so final
       shape is (C, Z, height, width).
    """
    arr = tifffile.imread(tif_path)
    # Handle (T, Z, C, Y, X) if 5D
    if arr.ndim == 5:
        if arr.shape[0] == 1:
            arr = arr[0]  # => (Z, C, Y, X)
        else:
            print("Warning: T>1, taking T=0 only.")
            arr = arr[0]
    elif arr.ndim != 4:
        raise ValueError(f"Expected 4D/5D, got {arr.shape}")

    # Reorder => (C, Z, Y, X)
    arr = np.transpose(arr, (1, 0, 2, 3))  # shape (C, Z, Y, X)

    # Compute desired bounding box in X and Y
    x0 = x_origin - width // 2
    x1 = x0 + width
    y0 = y_origin - height // 2
    y1 = y0 + height

    C, Z, max_y, max_x = arr.shape

    # track how far we are "out of bounds" on each side
    left_outside = 0
    right_outside = 0
    top_outside = 0
    bottom_outside = 0

    # Check left/right edges
    if x0 < 0:
        left_outside = -x0
        x0 = 0
    if x1 > max_x:
        right_outside = x1 - max_x
        x1 = max_x

    # Check top/bottom edges
    if y0 < 0:
        top_outside = -y0
        y0 = 0
    if y1 > max_y:
        bottom_outside = y1 - max_y
        y1 = max_y

    # Slice the region that actually exists within the image
    cropped = arr[..., y0:y1, x0:x1]  # shape => (C, Z, newH, newW)

    # Check how many pixels
    _, _, cropped_h, cropped_w = cropped.shape

    # The final size must be exactly (height, width).
    # pad with zeros for outside the image.
    pad_width = (
        (0, 0),  # C dimension
        (0, 0),  # Z dimension
        (top_outside, bottom_outside),    # Y dimension
        (left_outside, right_outside),    # X dimension
    )

    # in case the image was smaller on both sides
    needed_h = height - cropped_h - (top_outside + bottom_outside)
    needed_w = width - cropped_w - (left_outside + right_outside)

    if needed_h > 0:
        extra_top = needed_h // 2
        extra_bottom = needed_h - extra_top
        pad_width = (
            (0, 0),
            (0, 0),
            (pad_width[2][0] + extra_top, pad_width[2][1] + extra_bottom),
            (pad_width[3][0], pad_width[3][1]),
        )

    if needed_w > 0:
        extra_left = needed_w // 2
        extra_right = needed_w - extra_left
        pad_width = (
            (0, 0),
            (0, 0),
            (pad_width[2][0], pad_width[2][1]),
            (pad_width[3][0] + extra_left, pad_width[3][1] + extra_right),
        )

    # Now pad with zeros
    padded = np.pad(cropped, pad_width, mode="constant", constant_values=0)

    final_c, final_z, final_h, final_w = padded.shape
    if final_h != height or final_w != width:
        print(
            f"Warning: after padding, got shape (C,Z,H,W)=({final_c},{final_z},{final_h},{final_w}), "
            f"expected H={height},W={width}."
        )

    return padded


def process_tif(info):
    """
    1) Load + reorder + crop/pad => shape (C, Z, H, W).
    2) Create napari 3D.
    3) Add channels 
    4) Screenshot => <shorthand>.png
    5) Close viewer.
    """
    tif_path = info["filename"]
    shorthand = info["shorthand"]

    # Contrast:
    rx_low, rx_high = info["contrast_red"]
    gx_low, gx_high = info["contrast_green"]

    # Opacities:
    opacity_red = info.get("opacity_red", 0.65)      # default 0.65
    opacity_green = info.get("opacity_green", 0.4)   # default 0.4

    # Load + crop + pad
    arr = load_and_crop(
        tif_path,
        x_origin=info["x_origin"],
        y_origin=info["y_origin"],
        width=WIDTH,
        height=HEIGHT,
    )

    print(f"\nFile: {tif_path}")
    print(f"  Final crop/pad shape => {arr.shape} (C,Z,H,W)")
    print(f"  Saving screenshot => {shorthand}.png")

    # Create viewer
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3

    # Add red channel
    if arr.shape[0] >= 1:
        red = arr[0]
        viewer.add_image(
            red,
            name="Red",
            blending="additive",
            colormap="red",
            interpolation3d="cubic",
            rendering="attenuated_mip",
            opacity=opacity_red,
            contrast_limits=(int(rx_low), int(rx_high)),
        )

    # Add green channel
    if arr.shape[0] >= 2:
        green = arr[1]
        viewer.add_image(
            green,
            name="Green",
            blending="additive",
            colormap="green",
            interpolation3d="cubic",
            rendering="attenuated_mip",
            opacity=opacity_green,
            contrast_limits=(int(gx_low), int(gx_high)),
        )

    def take_screenshot():
        screenshot_array = viewer.screenshot(canvas_only=True)
        out_name = f"{shorthand}.png"
        Image.fromarray(screenshot_array).save(out_name)
        print(f"  Screenshot saved: {out_name}")
        viewer.close()

    QTimer.singleShot(500, take_screenshot)
    napari.run()


def main():
    for f in files:
        if not os.path.isfile(f["filename"]):
            print(f"ERROR: file not found => {f['filename']}")
            continue
        process_tif(f)

    print("\nAll done.")


if __name__ == "__main__":
    main()
