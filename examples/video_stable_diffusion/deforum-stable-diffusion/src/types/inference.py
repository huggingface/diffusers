from datetime import time
import io
import os
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
import requests

import torch
from helpers.save_images import get_output_folder
from PIL import Image
from PIL.GifImagePlugin import GifImageFile
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from PIL.TiffImagePlugin import TiffImageFile
from pydantic import (
    BaseConfig,
    BaseModel,
    Field,
    validator,
)
import validators
from devtools import debug

ImageType = Union[JpegImageFile, PngImageFile, GifImageFile, TiffImageFile, Image.Image]


def is_image(v):
    return (
        isinstance(v, PngImageFile)
        or isinstance(v, JpegImageFile)
        or isinstance(v, GifImageFile)
        or isinstance(v, TiffImageFile)
        or isinstance(v, Image.Image)
    )


# Helper for images
def validate_image(v, throw=True):
    if v is None:
        return v
    elif (
        isinstance(v, PngImageFile)
        or isinstance(v, JpegImageFile)
        or isinstance(v, GifImageFile)
        or isinstance(v, TiffImageFile)
        or isinstance(v, Image.Image)
    ):
        return v
    elif isinstance(v, bytes) or isinstance(v, io.BytesIO):
        return v
    elif isinstance(v, str):
        if validators.url(v):
            try:
                v = Image.open(requests.get(v, stream=True).raw).convert("RGB")
                return v
            except Exception as err:
                if throw:
                    raise ValueError(
                        "Invalid remote url, failed to parse image"
                    ) from err
                else:
                    return False
        elif os.path.isfile(v):
            try:
                with Image.open(v) as fd:
                    return fd.convert("RGB")
            except Exception as err:
                if throw:
                    raise ValueError(
                        "Invalid path, failed to parse image from local path"
                    ) from err
                else:
                    return False
        else:
            if throw:
                raise ValueError("Invalid string, no image or remote url")
            else:
                return False
    else:
        if throw:
            raise ValueError(
                f"Bad image type. Expected: bytes, Image, or Image url. Got: {debug.format(v)}"
            )
        else:
            return False


def output_folder_factory(output_path="outputs", batch_folder="deforum"):
    prefix = os.path.abspath(os.path.dirname(__file__))
    return get_output_folder(f"{prefix}/{output_path}".replace("//", "/"), batch_folder)


class DeforumArgs(BaseModel):
    W: Optional[int] = 512
    H: Optional[int] = 512
    seed: Optional[int] = -1
    sampler: Optional[
        Literal[
            "klms",
            "dpm2",
            "dpm2_ancestral",
            "heun",
            "euler",
            "euler_ancestral",
            "plms",
            "ddim",
        ]
    ] = "euler_ancestral"
    steps: Optional[int] = 80
    scale: Optional[int] = 7
    ddim_eta: Optional[float] = 0.0
    dynamic_threshold: Optional[float] = None
    static_threshold: Optional[float] = None
    save_samples: Optional[bool] = True
    save_settings: Optional[bool] = True
    display_samples: Optional[bool] = True
    save_sample_per_step: Optional[bool] = False
    show_sample_per_step: Optional[bool] = False
    prompt_weighting: Optional[bool] = False
    normalize_prompt_weights: Optional[bool] = False
    log_weighted_subprompts: Optional[bool] = False
    n_batch: Optional[int] = 1
    batch_name: Optional[str] = "StableFun"
    filename_format: Optional[
        Literal["{timestring}_{index}_{seed}.png", "{timestring}_{index}_{prompt}.png"]
    ] = "{timestring}_{index}_{prompt}.png"
    seed_behavior: Optional[Literal["iter", "constant", "random"]] = "iter"
    make_grid: Optional[bool] = False
    grid_rows: Optional[int] = 2
    outdir: Optional[str] = Field(default_factory=output_folder_factory)
    use_init: Optional[bool] = False
    strength: Optional[float] = 0.0
    strength_0_no_init: Optional[bool] = True
    init_image: Optional[
        ImageType
    ] = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
    use_mask: Optional[bool] = False
    use_alpha_as_mask: Optional[bool] = False
    mask_file: Optional[
        ImageType
    ] = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"
    invert_mask: Optional[bool] = False
    mask_brightness_adjust: Optional[float] = 1.0
    mask_contrast_adjust: Optional[float] = 1.0
    overlay_mask: Optional[bool] = True
    mask_overlay_blur: Optional[float] = 5
    mean_loss_scale: Optional[float] = 0
    var_loss_scale: Optional[float] = 0
    exposure_loss_scale: Optional[float] = 0
    exposure_target: Optional[float] = 0.5
    colormatch_loss_scale: Optional[float] = 0
    colormatch_image: Optional[
        ImageType
    ] = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"
    colormatch_n_colors: Optional[int] = 4
    ignore_sat_scale: Optional[float] = 0
    clip_name: Optional[
        Literal["ViT-L/14", "ViT-L/14@336px", "ViT-B/16", "ViT-B/32"]
    ] = "ViT-L/14"
    clip_loss_scale: Optional[float] = 0
    aesthetics_loss_scale: Optional[float] = 0
    cutn: Optional[int] = 1
    cut_pow: Optional[float] = 0.0001
    init_mse_scale: Optional[float] = 0
    blue_loss_scale: Optional[float] = 0
    gradient_wrt: Optional[Literal["x", "x0_pred"]] = "x0_pred"
    gradient_add_to: Optional[Literal["cond", "uncond", "both"]] = "both"
    decode_method: Optional[Literal["autoencoder", "linear"]] = "linear"
    grad_threshold_type: Optional[
        Literal["dynamic", "static", "mean", "schedule"]
    ] = "dynamic"
    clamp_grad_threshold: Optional[float] = 0.2
    clamp_start: Optional[float] = 0.2
    clamp_stop: Optional[float] = 0.01
    cond_uncond_sync: Optional[bool] = True
    n_samples: Optional[int] = 1
    precision: Optional[Literal["fp16", "autocast", "fp32"]] = "autocast"
    C: Optional[int] = 4
    f: Optional[int] = 8
    prompt: Optional[str] = ""
    timestring: Optional[str] = Field(
        default_factory=lambda: time.strftime("%Y%m%d%H%M%S")
    )
    init_latent: Optional[Union[float, torch.Tensor, np.ndarray]] = None
    init_sample: Optional[Union[float, torch.Tensor, np.ndarray]] = None
    init_c: Optional[Union[float, torch.Tensor, np.ndarray]] = None

    class Config(BaseConfig):
        arbitrary_types_allowed: Optional[bool] = True

    @validator("init_image", pre=True)
    def validate_image_init(cls, v):
        return validate_image(v)

    @validator("colormatch_image", pre=True)
    def validate_image_colormatch(cls, v):
        return validate_image(v)

    @validator("mask_file", pre=True)
    def validate_image_mask(cls, v):
        return validate_image(v)


class DeforumAnimArgs(BaseModel):
    animation_mode: Optional[
        Literal["None", "2D", "3D", "Video Input", "Interpolation"]
    ] = "None"
    max_frames: Optional[int] = 1000
    border: Optional[Literal["wrap", "replicate"]] = "replicate"
    angle: Optional[str] = "0:(0)"
    zoom: Optional[str] = "0:(1.04)"
    translation_x: Optional[str] = "0:(10*sin(2*3.14*t/10))"
    translation_y: Optional[str] = "0:(0)"
    translation_z: Optional[str] = "0:(10)"
    rotation_3d_x: Optional[str] = "0:(0)"
    rotation_3d_y: Optional[str] = "0:(0)"
    rotation_3d_z: Optional[str] = "0:(0)"
    flip_2d_perspective: Optional[bool] = False
    perspective_flip_theta: Optional[str] = "0:(0)"
    perspective_flip_phi: Optional[str] = "0:(t%15)"
    perspective_flip_gamma: Optional[str] = "0:(0)"
    perspective_flip_fv: Optional[str] = "0:(53)"
    noise_schedule: Optional[str] = "0: (0.02)"
    strength_schedule: Optional[str] = "0: (0.65)"
    contrast_schedule: Optional[str] = "0: (1.0)"
    color_coherence: Optional[str] = "Match Frame 0 LAB"
    diffusion_cadence: Optional[Literal["1", "2", "3", "4", "5", "6", "7", "8"]] = "1"
    use_depth_warping: Optional[bool] = True
    midas_weight: Optional[float] = 0.3
    near_plane: Optional[int] = 200
    far_plane: Optional[int] = 10000
    fov: Optional[float] = 40
    padding_mode: Optional[Literal["border", "reflection", "zeros"]] = "border"
    sampling_mode: Optional[Literal["bicubic", "bilinear", "nearest"]] = "bicubic"
    save_depth_maps: Optional[bool] = False
    video_init_path: Optional[str] = "/content/video_in.mp4"
    extract_nth_frame: Optional[int] = 1
    overwrite_extracted_frames: Optional[bool] = True
    use_mask_video: Optional[bool] = False
    video_mask_path: Optional[str] = "/content/video_in.mp4"
    interpolate_key_frames: Optional[bool] = False
    interpolate_x_frames: Optional[int] = 4
    resume_from_timestring: Optional[bool] = False
    resume_timestring: Optional[str] = "20220829210106"

    class Config(BaseConfig):
        arbitrary_types_allowed: Optional[bool] = True
