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
from ..modular_pipeline import SequentialPipelineBlocks
from .before_denoise import (
    WanAdditionalInputsStep,
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanTextInputStep,
)
from .decoders import WanVaeDecoderStep
from .denoise import (
    Wan22Image2VideoDenoiseStep,
)
from .encoders import (
    WanImageResizeStep,
    WanPrepareFirstFrameLatentsStep,
    WanTextEncoderStep,
    WanVaeEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. VAE ENCODER
# ====================


class WanImage2VideoVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanVaeEncoderStep, WanPrepareFirstFrameLatentsStep]
    block_names = ["image_resize", "vae_encoder", "prepare_first_frame_latents"]

    @property
    def description(self):
        return "Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent representation"


# ====================
# 2. DENOISE
# ====================


# inputs (text + image_condition_latents) -> set_timesteps -> prepare_latents -> denoise (latents)
class Wan22Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [
        WanTextInputStep,
        WanAdditionalInputsStep(image_latent_inputs=["image_condition_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        Wan22Image2VideoDenoiseStep,
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
            + " - `Wan22Image2VideoDenoiseStep` is used to denoise the latents in wan2.2\n"
        )


# ====================
# 3. BLOCKS (Wan2.2 Image2Video)
# ====================


class Wan22Image2VideoBlocks(SequentialPipelineBlocks):
    model_name = "wan-i2v"
    block_classes = [
        WanTextEncoderStep,
        WanImage2VideoVaeEncoderStep,
        Wan22Image2VideoCoreDenoiseStep,
        WanVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "vae_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Modular pipeline for image-to-video using Wan2.2.\n"
            + " - `WanTextEncoderStep` encodes the text\n"
            + " - `WanImage2VideoVaeEncoderStep` encodes the image\n"
            + " - `Wan22Image2VideoCoreDenoiseStep` denoes the latents\n"
            + " - `WanVaeDecoderStep` decodes the latents to video frames\n"
        )
