# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from .before_denoise import (
    WanAdditionalInputsStep,
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanTextInputStep,
)
from .decoders import WanVaeDecoderStep
from .denoise import (
    WanImage2VideoDenoiseStep,
)
from .encoders import (
    WanFirstLastFrameImageEncoderStep,
    WanFirstLastFrameVaeEncoderStep,
    WanImageCropResizeStep,
    WanImageEncoderStep,
    WanImageResizeStep,
    WanPrepareFirstFrameLatentsStep,
    WanPrepareFirstLastFrameLatentsStep,
    WanTextEncoderStep,
    WanVaeEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# ====================
# 1. IMAGE ENCODER
# ====================


# wan2.1 I2V (first frame only)
class WanImage2VideoImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanImageEncoderStep]
    block_names = ["image_resize", "image_encoder"]

    @property
    def description(self):
        return "Image2Video Image Encoder step that resize the image and encode the image to generate the image embeddings"


# wan2.1 FLF2V (first and last frame)
class WanFLF2VImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanImageCropResizeStep, WanFirstLastFrameImageEncoderStep]
    block_names = ["image_resize", "last_image_resize", "image_encoder"]

    @property
    def description(self):
        return "FLF2V Image Encoder step that resize and encode and encode the first and last frame images to generate the image embeddings"


# wan2.1 Auto Image Encoder
class WanAutoImageEncoderStep(AutoPipelineBlocks):
    block_classes = [WanFLF2VImageEncoderStep, WanImage2VideoImageEncoderStep]
    block_names = ["flf2v_image_encoder", "image2video_image_encoder"]
    block_trigger_inputs = ["last_image", "image"]
    model_name = "wan-i2v"

    @property
    def description(self):
        return (
            "Image Encoder step that encode the image to generate the image embeddings"
            + "This is an auto pipeline block that works for image2video tasks."
            + " - `WanFLF2VImageEncoderStep` (flf2v) is used when `last_image` is provided."
            + " - `WanImage2VideoImageEncoderStep` (image2video) is used when `image` is provided."
            + " - if `last_image` or `image` is not provided, step will be skipped."
        )


# ====================
# 2. VAE ENCODER
# ====================


# wan2.1 I2V (first frame only)
class WanImage2VideoVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanVaeEncoderStep, WanPrepareFirstFrameLatentsStep]
    block_names = ["image_resize", "vae_encoder", "prepare_first_frame_latents"]

    @property
    def description(self):
        return "Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent representation"


# wan2.1 FLF2V (first and last frame)
class WanFLF2VVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [
        WanImageResizeStep,
        WanImageCropResizeStep,
        WanFirstLastFrameVaeEncoderStep,
        WanPrepareFirstLastFrameLatentsStep,
    ]
    block_names = ["image_resize", "last_image_resize", "vae_encoder", "prepare_first_last_frame_latents"]

    @property
    def description(self):
        return "FLF2V Vae Image Encoder step that resize and encode and encode the first and last frame images to generate the latent conditions"


# wan2.1 Auto Vae Encoder
class WanAutoVaeEncoderStep(AutoPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [WanFLF2VVaeEncoderStep, WanImage2VideoVaeEncoderStep]
    block_names = ["flf2v_vae_encoder", "image2video_vae_encoder"]
    block_trigger_inputs = ["last_image", "image"]

    @property
    def description(self):
        return (
            "Vae Image Encoder step that encode the image to generate the image latents"
            + "This is an auto pipeline block that works for image2video tasks."
            + " - `WanFLF2VVaeEncoderStep` (flf2v) is used when `last_image` is provided."
            + " - `WanImage2VideoVaeEncoderStep` (image2video) is used when `image` is provided."
            + " - if `last_image` or `image` is not provided, step will be skipped."
        )


# ====================
# 3. DENOISE (inputs -> set_timesteps -> prepare_latents -> denoise)
# ====================


# wan2.1 I2V core denoise (support both I2V and FLF2V)
# inputs (text + image_condition_latents) -> set_timesteps -> prepare_latents -> denoise (latents)
class WanImage2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [
        WanTextInputStep,
        WanAdditionalInputsStep(image_latent_inputs=["image_condition_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanImage2VideoDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "set_timesteps",
        "prepare_latents",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "denoise block that takes encoded text and image latent conditions and runs the denoising process.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `WanTextInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `WanAdditionalInputsStep` is used to adjust the batch size of the latent conditions\n"
            + " - `WanSetTimestepsStep` is used to set the timesteps\n"
            + " - `WanPrepareLatentsStep` is used to prepare the latents\n"
            + " - `WanImage2VideoDenoiseStep` is used to denoise the latents\n"
        )


# ====================
# 4. BLOCKS (Wan2.1 Image2Video)
# ====================


# wan2.1 Image2Video Auto Blocks
class WanImage2VideoAutoBlocks(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [
        WanTextEncoderStep,
        WanAutoImageEncoderStep,
        WanAutoVaeEncoderStep,
        WanImage2VideoCoreDenoiseStep,
        WanVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "image_encoder",
        "vae_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for image-to-video using Wan.\n"
            + "- for I2V workflow, all you need to provide is `image`"
            + "- for FLF2V workflow, all you need to provide is `last_image` and `image`"
        )
