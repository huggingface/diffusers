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
from ..modular_pipeline_utils import InsertableDict
from .before_denoise import (
    WanInputStep,
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
)
from .decoders import WanDecodeStep
from .denoise import WanDenoiseStep, WanI2VDenoiseStep
from .encoders import WanImageEncoderStep, WanTextEncoderStep, WanVaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAutoImageEncoderStep(AutoPipelineBlocks):
    block_classes = [WanImageEncoderStep]
    block_names = ["image_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Image encoder step that encodes the image inputs into a conditioning embedding.\n"
            + "This is an auto pipeline block that works for both first-frame and first-last-frame conditioning tasks.\n"
            + " - `WanImageEncoderStep` (image_encoder) is used when `image`, and possibly `last_image` is provided."
            + " - if `image` is not provided, this step will be skipped."
        )


class WanAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [WanVaeEncoderStep]
    block_names = ["img2vid"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for both first-frame and first-last-frame conditioning tasks.\n"
            + " - `WanVaeEncoderStep` (img2vid) is used when `image`, and possibly `last_image` is provided."
            + " - if `image` is not provided, this step will be skipped."
        )


# before_denoise: text2vid
class WanBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanInputStep,
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `WanInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `WanSetTimestepsStep` is used to set the timesteps\n"
            + " - `WanPrepareLatentsStep` is used to prepare the latents\n"
        )


# before_denoise: img2vid
class WanI2VBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanInputStep,
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanImageEncoderStep,
        WanVaeEncoderStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "image_encoder", "vae_encoder"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for image-to-video and first-last-frame-to-video tasks.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `WanInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `WanSetTimestepsStep` is used to set the timesteps\n"
            + " - `WanPrepareLatentsStep` is used to prepare the latents\n"
            + " - `WanImageEncoderStep` is used to encode the image inputs into a conditioning embedding\n"
            + " - `WanVaeEncoderStep` is used to encode the image/last-image inputs into their latent representations\n"
        )


# before_denoise: all task (text2vid, img2vid)
class WanAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        WanBeforeDenoiseStep,
        WanI2VBeforeDenoiseStep,
    ]
    block_names = ["text2vid", "img2vid"]
    block_trigger_inputs = [None, "image"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2vid, img2vid, first-last-frame2vid.\n"
            + " - `WanBeforeDenoiseStep` (text2vid) is used.\n"
            + " - `WanI2VBeforeDenoiseStep` (img2vid) is used when `image` is provided.\n"
        )


# denoise: text2vid, img2vid
class WanAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        WanDenoiseStep,
        WanI2VDenoiseStep,
    ]
    block_names = ["denoise", "denoise_i2v"]
    block_trigger_inputs = [None, "image"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2vid and img2vid tasks..."
            " - `WanDenoiseStep` (denoise) for text2vid task."
            " - `WanI2VDenoiseStep` (denoise_i2v) for img2vid task, which is used when `image` is provided.\n"
        )


# decode: all task (text2img, img2img, inpainting)
class WanAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [WanDecodeStep]
    block_names = ["decode"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return "Decode step that decode the denoised latents into videos outputs.\n - `WanDecodeStep`"


# text2vid
class WanAutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        WanTextEncoderStep,
        WanAutoBeforeDenoiseStep,
        WanAutoDenoiseStep,
        WanAutoDecodeStep,
    ]
    block_names = [
        "text_encoder",
        "before_denoise",
        "denoise",
        "decoder",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-video using Wan.\n"
            + "- for text-to-video generation, all you need to provide is `prompt`"
        )


# img2vid and first-last-frame2vid
class WanI2VAutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        WanTextEncoderStep,
        WanAutoBeforeDenoiseStep,
        WanImageEncoderStep,
        WanAutoVaeEncoderStep,
        WanAutoDenoiseStep,
        WanAutoDecodeStep,
    ]
    block_names = [
        "text_encoder",
        "before_denoise",
        "image_encoder",
        "vae_encoder",
        "denoise",
        "decoder",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-video using Wan.\n"
            + "- for image-to-video and first-last-frame-to-video generation, you need to provide is `image`, and possibly `last_image`"
        )


TEXT2VIDEO_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("input", WanInputStep),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("denoise", WanDenoiseStep),
        ("decode", WanDecodeStep),
    ]
)


IMAGE2VIDEO_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("input", WanInputStep),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("image_encoder", WanImageEncoderStep),
        ("vae_encoder", WanVaeEncoderStep),
        ("denoise", WanI2VDenoiseStep),
        ("decode", WanDecodeStep),
    ]
)


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("image_encoder", WanAutoImageEncoderStep),
        ("vae_encoder", WanAutoVaeEncoderStep),
        ("before_denoise", WanAutoBeforeDenoiseStep),
        ("denoise", WanAutoDenoiseStep),
        ("decode", WanAutoDecodeStep),
    ]
)


ALL_BLOCKS = {
    "text2video": TEXT2VIDEO_BLOCKS,
    "image2video": IMAGE2VIDEO_BLOCKS,
    "auto": AUTO_BLOCKS,
}
