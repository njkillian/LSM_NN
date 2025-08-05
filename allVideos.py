"""
allVideos.py

LSM Volume Rotation Videos
For visualization, not analysis
NKillian 2025
"""

import os
import time
import numpy as np
import tifffile
import napari
import imageio.v2 as iio
from qtpy.QtCore import QTimer
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize

##############################################################################
# Run modes / flags
##############################################################################

useEllipse     = True   # If True => mask data outside ellipse
showZmmLabel   = False   # If True => show the top-left "Z=..." text
mode_2D        = False   # If True => 2D slice-by-slice
mode_3D_partial= False   # If True => 3D partial stack each frame
mode_3D_rotate = True    # If True => rotate around Y axis

firstSliceIndex = 1
lastSliceIndex  = None  # if None => entire Z dimension
zSpacing        = 0.016

CROP_SIZE         = 900
DOWNSAMPLE_FACTOR = 1.0
FINAL_DIM         = 912

FPS          = 30
DURATION_SEC = 10
TOTAL_FRAMES = FPS * DURATION_SEC

ELLIPSE_MAJOR_900   = 856
ELLIPSE_MINOR_900   = 624
SCALEBAR_LENGTH_900 = 69
LABEL_FONT_SIZE_900 = 30
LABEL_STR           = "1 mm"

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

##############################################################################
# 1) LOAD & CROP
##############################################################################

def load_and_crop_900x900(tif_path, x_origin, y_origin, size=900):
    print(f"Loading TIF => {tif_path}")
    arr = tifffile.imread(tif_path)

    # If 5D => T=0
    if arr.ndim == 5:
        if arr.shape[0] > 1:
            print("Warning: T>1 => using T=0 only.")
        arr = arr[0]
    elif arr.ndim != 4:
        raise ValueError(f"Expected 4D or 5D => got {arr.shape}")

    # reorder => (C,Z,Y,X)
    arr = np.transpose(arr, (1,0,2,3))
    C, Z, max_y, max_x = arr.shape
    print(f"  shape => (C={C},Z={Z},{max_y}×{max_x}) before crop")

    # Crop => 900×900 around (x_origin, y_origin)
    x0 = x_origin - size//2
    x1 = x0 + size
    y0 = y_origin - size//2
    y1 = y0 + size

    left_out=0; right_out=0
    top_out=0;  bottom_out=0
    if x0<0:
        left_out=-x0; x0=0
    if x1>max_x:
        right_out=x1-max_x; x1=max_x
    if y0<0:
        top_out=-y0; y0=0
    if y1>max_y:
        bottom_out=y1-max_y; y1=max_y

    cropped= arr[..., y0:y1, x0:x1]  # => (C,Z,newH,newW)
    _,_,ch,cw= cropped.shape

    needed_h= size-ch-(top_out+bottom_out)
    needed_w= size-cw-(left_out+right_out)
    pad_width=(
        (0,0),
        (0,0),
        (top_out,bottom_out),
        (left_out,right_out),
    )
    if needed_h>0:
        extra_top= needed_h//2
        extra_bottom= needed_h- extra_top
        pad_width=(
            (0,0),
            (0,0),
            (pad_width[2][0]+extra_top, pad_width[2][1]+extra_bottom),
            (pad_width[3][0], pad_width[3][1]),
        )
    if needed_w>0:
        extra_left= needed_w//2
        extra_right= needed_w- extra_left
        pad_width=(
            (0,0),
            (0,0),
            (pad_width[2][0], pad_width[2][1]),
            (pad_width[3][0]+extra_left, pad_width[3][1]+extra_right),
        )
    final_900= np.pad(cropped, pad_width, mode="constant", constant_values=0)
    print(f"  after crop => {final_900.shape}")
    return final_900

def rotate_180(arr_4d):
    print("  Rotating 180° => reversing Y & X.")
    return arr_4d[..., ::-1, ::-1]

def downsample_xy(data_4d, factor=1.0):
    if factor == 1.0:
        print("  No downsample => factor=1 => skipping.")
        return data_4d
    from skimage.transform import resize
    C,Z,H,W = data_4d.shape
    new_h= int(round(H/factor))
    new_w= int(round(W/factor))
    print(f"  Downsample => factor={factor}, => {new_h}×{new_w}")
    out= np.zeros((C,Z,new_h,new_w), dtype=data_4d.dtype)
    for c_idx in range(C):
        for z_idx in range(Z):
            ds= resize(data_4d[c_idx,z_idx],(new_h,new_w),
                       preserve_range=True, anti_aliasing=True, order=1)
            out[c_idx,z_idx]= ds.astype(data_4d.dtype)
    return out

def pad_to_dim(data_4d, final_dim):
    C,Z,H,W= data_4d.shape
    if (H==final_dim and W==final_dim):
        return data_4d
    print(f"  Pad => {final_dim}×{final_dim} from {H}×{W}")
    pad_top= max(0,(final_dim-H)//2)
    pad_left= max(0,(final_dim-W)//2)
    pad_bottom= final_dim-H-pad_top if H<final_dim else 0
    pad_right= final_dim-W-pad_left if W<final_dim else 0
    pad_width=(
        (0,0),
        (0,0),
        (pad_top,pad_bottom),
        (pad_left,pad_right),
    )
    return np.pad(data_4d, pad_width, mode="constant", constant_values=0)

##############################################################################
# 2) Ellipse & scale bar
##############################################################################

def create_ellipse_mask(h,w, major, minor):
    print(f"  Creating ellipse => {h}×{w}, major={major}, minor={minor}")
    cy=h/2; cx=w/2
    ry=major/2; rx=minor/2
    Y,X= np.ogrid[:h,:w]
    dist= ((X-cx)/rx)**2 + ((Y-cy)/ry)**2
    return dist<=1.0

def center_crop_screenshot(rgba, final_dim):
    sh,sw,_= rgba.shape
    if (sh==final_dim and sw==final_dim):
        return rgba
    print(f"    Napari screenshot => {sh}×{sw}, center-crop => {final_dim}×{final_dim}")
    start_y= max(0,(sh-final_dim)//2)
    start_x= max(0,(sw-final_dim)//2)
    end_y= start_y+final_dim
    end_x= start_x+final_dim
    end_y= min(end_y, sh)
    end_x= min(end_x, sw)
    cropped= rgba[start_y:end_y, start_x:end_x]
    ch,cw,_= cropped.shape
    if ch<final_dim or cw<final_dim:
        pad_top=(final_dim-ch)//2
        pad_left=(final_dim-cw)//2
        pad_bottom=final_dim-ch-pad_top
        pad_right= final_dim-cw-pad_left
        pad_width=((pad_top,pad_bottom),(pad_left,pad_right),(0,0))
        cropped= np.pad(cropped, pad_width, constant_values=0)
    return cropped

def post_process_frame(rgba, ellipse_mask, scale_len, font_size, frame_idx, zmm):
    """
    If useEllipse => outside => black.
    Draw scale bar.
    Optionally draw Z label if showZmmLabel=True.
    """
    h,w,c= rgba.shape
    if c==3:
        alpha= np.full((h,w,1),255,dtype=np.uint8)
        rgba= np.concatenate([rgba,alpha],axis=2)

    if useEllipse:
        outside= ~ellipse_mask
        rgba[outside,0]=0
        rgba[outside,1]=0
        rgba[outside,2]=0

    pil_im= Image.fromarray(rgba[...,:3], mode='RGB')
    draw= ImageDraw.Draw(pil_im)
    try:
        font= ImageFont.truetype("arial.ttf", size=font_size)
    except:
        font= ImageFont.load_default()

    # scale bar
    bar_x2= w-20
    bar_x1= bar_x2 - scale_len
    bar_y=  h-30
    draw.line([(bar_x1,bar_y),(bar_x2,bar_y)], fill=(255,255,255), width=4)
    label_str= LABEL_STR
    text_x= bar_x1-5
    text_y= bar_y-(font_size+5)
    draw.text((text_x,text_y), label_str, fill=(255,255,255), font=font)

    # optional z label
    if showZmmLabel:
        draw.text((10,10), f"Z={zmm:.2f} mm", fill=(255,255,255), font=font)

    return np.array(pil_im)

##############################################################################
# 3) PROCESS (both Red & Green in one rendering)
##############################################################################

def process_tif(info):
    fname= info["filename"]
    short= info["shorthand"]
    print(f"\nProcessing => {fname} (both red & green together)")

    # Load & crop
    arr_900= load_and_crop_900x900(fname, info["x_origin"], info["y_origin"], CROP_SIZE)
    if short=="late2":
        print("  Rotating 180° now, post-crop.")
        arr_900= rotate_180(arr_900)

    # Downsample & pad
    arr_450= downsample_xy(arr_900, DOWNSAMPLE_FACTOR)
    arr_464= pad_to_dim(arr_450, FINAL_DIM)
    C,Z,hf,wf= arr_464.shape
    print(f"  final => (C={C},Z={Z},{hf}×{wf})")

    # Must have at least two channels
    if C < 2:
        print("  This TIFF has <2 channels => can't do red+green together.")
        return

    if useEllipse:
        ellipse_mask = create_ellipse_mask(hf, wf, 
                                           ELLIPSE_MAJOR_900/DOWNSAMPLE_FACTOR,
                                           ELLIPSE_MINOR_900/DOWNSAMPLE_FACTOR)
        # Zero out the volume outside ellipse:
        for z in range(Z):
            arr_464[:, z, ~ellipse_mask] = 0
        

    # Check slice range
    global lastSliceIndex
    if not lastSliceIndex:
        lastSlice= Z
    else:
        lastSlice= min(lastSliceIndex, Z)
    fs= firstSliceIndex
    slices_to_use= lastSlice - fs + 1
    frames_to_make= min(slices_to_use, TOTAL_FRAMES)

    print(f"  We'll produce frames from sliceIndex={fs}..{lastSlice}, total => {frames_to_make}")

    rx_min, rx_max = info["contrast_red"]
    gx_min, gx_max = info["contrast_green"]
    op_r = info.get("opacity_red", 0.65)
    op_g = info.get("opacity_green", 0.4)

    scale_len= int(round(SCALEBAR_LENGTH_900 / DOWNSAMPLE_FACTOR))
    font_size= int(round(LABEL_FONT_SIZE_900 / DOWNSAMPLE_FACTOR))

    frames_dir= f"frames_{short}_both"
    os.makedirs(frames_dir, exist_ok=True)

    # how to produce frames (2D, partial 3D, or 3D rotate)
    if mode_3D_rotate:
        print("  Running in 3D rotation mode => both channels visible.")
        anglePerFrame= 360.0 / TOTAL_FRAMES
        frames_done= 0

        viewer= napari.Viewer()
        viewer.dims.ndisplay=3

        # Add the Red channel
        viewer.add_image(
            arr_464[0],     # shape => (Z,H,W)
            name="Red3D",
            blending="additive",
            colormap="red",
            interpolation3d="cubic",
            rendering="attenuated_mip",
            opacity=op_r,
            contrast_limits=(rx_min, rx_max),
        )
        # Add the Green channel
        viewer.add_image(
            arr_464[1],     # shape => (Z,H,W)
            name="Green3D",
            blending="additive",
            colormap="green",
            interpolation3d="cubic",
            rendering="attenuated_mip",
            opacity=op_g,
            contrast_limits=(gx_min, gx_max),
        )

        for i in range(TOTAL_FRAMES):
            angleY= i * anglePerFrame
            print(f"  Frame {i}/{TOTAL_FRAMES-1}, angleY={angleY:.2f} deg")
            # Adjust camera
            viewer.camera.angles= (0, angleY, 90)

            time.sleep(0.05)
            shot_rgba= viewer.screenshot(canvas_only=True, size=(wf,wf))
            shot_rgba= center_crop_screenshot(shot_rgba, FINAL_DIM)

            ellipse_mask_2d = None
            ellipse_mask_2d = np.ones((FINAL_DIM, FINAL_DIM), dtype=bool)

            frame_out= post_process_frame(
                shot_rgba,
                ellipse_mask_2d,
                scale_len,
                font_size,
                frame_idx=i,
                zmm=0
            )
            frame_path= os.path.join(frames_dir, f"frame{i:04d}_both.png")
            iio.imwrite(frame_path, frame_out)
            frames_done= i+1

        viewer.close()

        out_mp4= f"{short}_fitrotate_both.mp4"
        print(f"  => Creating {out_mp4} from {frames_done} frames...")
        with iio.get_writer(
            out_mp4,
            format="FFMPEG",
            mode="I",
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            quality=6,
        ) as writer:
            for fidx in range(frames_done):
                frame_path= os.path.join(frames_dir, f"frame{fidx:04d}_both.png")
                frm= iio.imread(frame_path)
                writer.append_data(frm)
        print(f"  Done => {out_mp4}")
        return

    if mode_2D:
        print("  Running in 2D slice-by-slice mode, both channels.")
        viewer= napari.Viewer()
        viewer.dims.ndisplay=2
        frames_done= 0

        for i in range(frames_to_make):
            zIndex= fs + i
            if zIndex> Z:
                break
            zmm= (zIndex - fs)* zSpacing

            # pick the single slice from each channel
            redSlice   = arr_464[0, zIndex-1]  # shape => (H,W)
            greenSlice = arr_464[1, zIndex-1]  # shape => (H,W)

            viewer.layers.clear()
            viewer.add_image(
                redSlice,
                name="Red2D",
                colormap="red",
                blending="additive",
                opacity=op_r,
                contrast_limits=(rx_min, rx_max),
                interpolation="nearest",
            )
            viewer.add_image(
                greenSlice,
                name="Green2D",
                colormap="green",
                blending="additive",
                opacity=op_g,
                contrast_limits=(gx_min, gx_max),
                interpolation="nearest",
            )

            time.sleep(0.05)
            shot_rgba= viewer.screenshot(canvas_only=True, size=(wf,wf))
            shot_rgba= center_crop_screenshot(shot_rgba, FINAL_DIM)

            ellipse_mask_2d = None
            if useEllipse:
                ellipse_mask_2d = create_ellipse_mask(
                    FINAL_DIM, FINAL_DIM, 
                    ELLIPSE_MAJOR_900/DOWNSAMPLE_FACTOR,
                    ELLIPSE_MINOR_900/DOWNSAMPLE_FACTOR
                )
            else:
                ellipse_mask_2d = np.ones((FINAL_DIM, FINAL_DIM), dtype=bool)

            frame_out= post_process_frame(shot_rgba, ellipse_mask_2d, scale_len, font_size, i, zmm)
            frame_path= os.path.join(frames_dir, f"frame{i:04d}_both.png")
            iio.imwrite(frame_path, frame_out)
            frames_done= i+1

        viewer.close()
        out_mp4= f"{short}_2D_both.mp4"
        print(f"  => Creating {out_mp4} from {frames_done} frames...")
        with iio.get_writer(
            out_mp4,
            format="FFMPEG",
            mode="I",
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            quality=6,
        ) as writer:
            for fidx in range(frames_done):
                frame_path= os.path.join(frames_dir, f"frame{fidx:04d}_both.png")
                frm= iio.imread(frame_path)
                writer.append_data(frm)
        print(f"  Done => {out_mp4}")
        return

    if mode_3D_partial:
        print("  Running in 3D partial stack mode, both channels.")
        frames_done= 0
        viewer= napari.Viewer()
        viewer.dims.ndisplay=3

        for i in range(frames_to_make):
            zIndex= fs + i
            if zIndex> Z:
                break
            zmm= (zIndex - fs)* zSpacing

            # partial => from zIndex..end => shape => (C, ZRemaining, H, W)
            partial= arr_464[:, (zIndex-1):, ...]
            viewer.layers.clear()

            # Add red
            viewer.add_image(
                partial[0],      # shape => (Zremain,H,W)
                name="Red3D",
                colormap="red",
                blending="additive",
                interpolation3d="cubic",
                rendering="attenuated_mip",
                opacity=op_r,
                contrast_limits=(rx_min, rx_max),
            )
            # Add green
            viewer.add_image(
                partial[1],
                name="Green3D",
                colormap="green",
                blending="additive",
                interpolation3d="cubic",
                rendering="attenuated_mip",
                opacity=op_g,
                contrast_limits=(gx_min, gx_max),
            )

            time.sleep(0.05)
            shot_rgba= viewer.screenshot(canvas_only=True, size=(wf,wf))
            shot_rgba= center_crop_screenshot(shot_rgba, FINAL_DIM)

            ellipse_mask_2d = None
            if useEllipse:
                ellipse_mask_2d = create_ellipse_mask(
                    FINAL_DIM, FINAL_DIM, 
                    ELLIPSE_MAJOR_900/DOWNSAMPLE_FACTOR,
                    ELLIPSE_MINOR_900/DOWNSAMPLE_FACTOR
                )
            else:
                ellipse_mask_2d = np.ones((FINAL_DIM, FINAL_DIM), dtype=bool)

            frame_out= post_process_frame(shot_rgba, ellipse_mask_2d, scale_len, font_size, i, zmm)
            frame_path= os.path.join(frames_dir, f"frame{i:04d}_both.png")
            iio.imwrite(frame_path, frame_out)
            frames_done= i+1

        viewer.close()
        out_mp4= f"{short}_3D_both.mp4"
        print(f"  => Creating {out_mp4} from {frames_done} frames...")
        with iio.get_writer(
            out_mp4,
            format="FFMPEG",
            mode="I",
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            quality=6,
        ) as writer:
            for fidx in range(frames_done):
                frame_path= os.path.join(frames_dir, f"frame{fidx:04d}_both.png")
                frm= iio.imread(frame_path)
                writer.append_data(frm)
        print(f"  Done => {out_mp4}")
        return

    print("No valid mode chosen => do nothing.")

##############################################################################
def main():
    print(f"useEllipse={useEllipse}, showZmmLabel={showZmmLabel}, "
          f"mode_2D={mode_2D}, mode_3D_partial={mode_3D_partial}, mode_3D_rotate={mode_3D_rotate}")

    for info in files:
        path= info["filename"]
        if not os.path.isfile(path):
            print(f"File not found => {path}, skipping.")
            continue
        global lastSliceIndex
        lastSliceIndex = None
        process_tif(info)

    print("\nAll done!")

if __name__ == "__main__":
    main()
