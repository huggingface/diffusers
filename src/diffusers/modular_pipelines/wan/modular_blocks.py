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
    WanAdditionalInputsStep,
    WanPrepareFirstFrameLatentsStep,
    WanPrepareFirstLastFrameLatentsStep,
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanTextInputStep,
)
from .decoders import WanImageVaeDecoderStep
from .denoise import (
    Wan22DenoiseStep,
    Wan22Image2VideoDenoiseStep,
    WanDenoiseStep,
    WanFLF2VDenoiseStep,
    WanImage2VideoDenoiseStep,
)
from .encoders import (
    WanFirstLastFrameImageEncoderStep,
    WanFirstLastFrameVaeImageEncoderStep,
    WanImageCropResizeStep,
    WanImageEncoderStep,
    WanImageResizeStep,
    WanTextEncoderStep,
    WanVaeImageEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# wan2.1
# wan2.1: text2vid
class WanCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanTextInputStep,
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return (
            "denoise block that takes encoded conditions and runs the denoising process.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `WanTextInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `WanSetTimestepsStep` is used to set the timesteps\n"
            + " - `WanPrepareLatentsStep` is used to prepare the latents\n"
            + " - `WanDenoiseStep` is used to denoise the latents\n"
        )


# wan2.1: image2video
## image encoder
class WanImage2VideoImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan"
    block_classes = [WanImageResizeStep, WanImageEncoderStep]
    block_names = ["image_resize", "image_encoder"]

    @property
    def description(self):
        return "Image2Video Image Encoder step that resize the image and encode the image to generate the image embeddings"


## vae encoder
class WanImage2VideoVaeImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan"
    block_classes = [WanImageResizeStep, WanVaeImageEncoderStep]
    block_names = ["image_resize", "vae_image_encoder"]

    @property
    def description(self):
        return "Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent representation"


## denoise
class WanImage2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanTextInputStep,
        WanAdditionalInputsStep(image_latent_inputs=["first_frame_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanPrepareFirstFrameLatentsStep,
        WanImage2VideoDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "set_timesteps",
        "prepare_latents",
        "prepare_first_frame_latents",
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
            + " - `WanPrepareFirstFrameLatentsStep` is used to prepare the first frame latent conditions\n"
            + " - `WanImage2VideoDenoiseStep` is used to denoise the latents\n"
        )


# wan2.1: FLF2v


## image encoder
class WanFLF2VImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan"
    block_classes = [WanImageResizeStep, WanImageCropResizeStep, WanFirstLastFrameImageEncoderStep]
    block_names = ["image_resize", "last_image_resize", "image_encoder"]

    @property
    def description(self):
        return "FLF2V Image Encoder step that resize and encode and encode the first and last frame images to generate the image embeddings"


## vae encoder
class WanFLF2VVaeImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan"
    block_classes = [WanImageResizeStep, WanImageCropResizeStep, WanFirstLastFrameVaeImageEncoderStep]
    block_names = ["image_resize", "last_image_resize", "vae_image_encoder"]

    @property
    def description(self):
        return "FLF2V Vae Image Encoder step that resize and encode and encode the first and last frame images to generate the latent conditions"


## denoise
class WanFLF2VCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanTextInputStep,
        WanAdditionalInputsStep(image_latent_inputs=["first_last_frame_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanPrepareFirstLastFrameLatentsStep,
        WanFLF2VDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "set_timesteps",
        "prepare_latents",
        "prepare_first_last_frame_latents",
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
            + " - `WanPrepareFirstLastFrameLatentsStep` is used to prepare the latent conditions\n"
            + " - `WanImage2VideoDenoiseStep` is used to denoise the latents\n"
        )


# wan2.1: auto blocks
## image encoder
class WanAutoImageEncoderStep(AutoPipelineBlocks):
    block_classes = [WanFLF2VImageEncoderStep, WanImage2VideoImageEncoderStep]
    block_names = ["flf2v_image_encoder", "image2video_image_encoder"]
    block_trigger_inputs = ["last_image", "image"]

    @property
    def description(self):
        return (
            "Image Encoder step that encode the image to generate the image embeddings"
            + "This is an auto pipeline block that works for image2video tasks."
            + " - `WanFLF2VImageEncoderStep` (flf2v) is used when `last_image` is provided."
            + " - `WanImage2VideoImageEncoderStep` (image2video) is used when `image` is provided."
            + " - if `last_image` or `image` is not provided, step will be skipped."
        )


## vae encoder
class WanAutoVaeImageEncoderStep(AutoPipelineBlocks):
    block_classes = [WanFLF2VVaeImageEncoderStep, WanImage2VideoVaeImageEncoderStep]
    block_names = ["flf2v_vae_image_encoder", "image2video_vae_image_encoder"]
    block_trigger_inputs = ["last_image", "image"]

    @property
    def description(self):
        return (
            "Vae Image Encoder step that encode the image to generate the image latents"
            + "This is an auto pipeline block that works for image2video tasks."
            + " - `WanFLF2VVaeImageEncoderStep` (flf2v) is used when `last_image` is provided."
            + " - `WanImage2VideoVaeImageEncoderStep` (image2video) is used when `image` is provided."
            + " - if `last_image` or `image` is not provided, step will be skipped."
        )


## denoise
class WanAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        WanFLF2VCoreDenoiseStep,
        WanImage2VideoCoreDenoiseStep,
        WanCoreDenoiseStep,
    ]
    block_names = ["flf2v", "image2video", "text2video"]
    block_trigger_inputs = ["first_last_frame_latents", "first_frame_latents", None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2video and image2video tasks."
            " - `WanCoreDenoiseStep` (text2video) for text2vid tasks."
            " - `WanCoreImage2VideoCoreDenoiseStep` (image2video) for image2video tasks."
            + " - if `first_frame_latents` is provided, `WanCoreImage2VideoDenoiseStep` will be used.\n"
            + " - if `first_frame_latents` is not provided, `WanCoreDenoiseStep` will be used.\n"
        )


# auto pipeline blocks
class WanAutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        WanTextEncoderStep,
        WanAutoImageEncoderStep,
        WanAutoVaeImageEncoderStep,
        WanAutoDenoiseStep,
        WanImageVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "image_encoder",
        "vae_image_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-video using Wan.\n"
            + "- for text-to-video generation, all you need to provide is `prompt`"
        )


# wan22
# wan2.2: text2vid


## denoise
class Wan22CoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanTextInputStep,
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        Wan22DenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return (
            "denoise block that takes encoded conditions and runs the denoising process.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `WanTextInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `WanSetTimestepsStep` is used to set the timesteps\n"
            + " - `WanPrepareLatentsStep` is used to prepare the latents\n"
            + " - `Wan22DenoiseStep` is used to denoise the latents in wan2.2\n"
        )


# wan2.2: image2video
## denoise
class Wan22Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanTextInputStep,
        WanAdditionalInputsStep(image_latent_inputs=["first_frame_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanPrepareFirstFrameLatentsStep,
        Wan22Image2VideoDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "set_timesteps",
        "prepare_latents",
        "prepare_first_frame_latents",
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
            + " - `WanPrepareFirstFrameLatentsStep` is used to prepare the first frame latent conditions\n"
            + " - `Wan22Image2VideoDenoiseStep` is used to denoise the latents in wan2.2\n"
        )


class Wan22AutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        Wan22Image2VideoCoreDenoiseStep,
        Wan22CoreDenoiseStep,
    ]
    block_names = ["image2video", "text2video"]
    block_trigger_inputs = ["first_frame_latents", None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2video and image2video tasks."
            " - `Wan22Image2VideoCoreDenoiseStep` (image2video) for image2video tasks."
            " - `Wan22CoreDenoiseStep` (text2video) for text2vid tasks."
            + " - if `first_frame_latents` is provided, `Wan22Image2VideoCoreDenoiseStep` will be used.\n"
            + " - if `first_frame_latents` is not provided, `Wan22CoreDenoiseStep` will be used.\n"
        )


class Wan22AutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        WanTextEncoderStep,
        WanAutoVaeImageEncoderStep,
        Wan22AutoDenoiseStep,
        WanImageVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "vae_image_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-video using Wan2.2.\n"
            + "- for text-to-video generation, all you need to provide is `prompt`"
        )


# presets for wan2.1 and wan2.2
# YiYi Notes: should we move these to doc?
# wan2.1
TEXT2VIDEO_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("input", WanTextInputStep),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("denoise", WanDenoiseStep),
        ("decode", WanImageVaeDecoderStep),
    ]
)

IMAGE2VIDEO_BLOCKS = InsertableDict(
    [
        ("image_resize", WanImageResizeStep),
        ("image_encoder", WanImage2VideoImageEncoderStep),
        ("vae_image_encoder", WanImage2VideoVaeImageEncoderStep),
        ("input", WanTextInputStep),
        ("additional_inputs", WanAdditionalInputsStep(image_latent_inputs=["first_frame_latents"])),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("prepare_first_frame_latents", WanPrepareFirstFrameLatentsStep),
        ("denoise", WanImage2VideoDenoiseStep),
        ("decode", WanImageVaeDecoderStep),
    ]
)


FLF2V_BLOCKS = InsertableDict(
    [
        ("image_resize", WanImageResizeStep),
        ("last_image_resize", WanImageCropResizeStep),
        ("image_encoder", WanFLF2VImageEncoderStep),
        ("vae_image_encoder", WanFLF2VVaeImageEncoderStep),
        ("input", WanTextInputStep),
        ("additional_inputs", WanAdditionalInputsStep(image_latent_inputs=["first_last_frame_latents"])),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("prepare_first_last_frame_latents", WanPrepareFirstLastFrameLatentsStep),
        ("denoise", WanFLF2VDenoiseStep),
        ("decode", WanImageVaeDecoderStep),
    ]
)

AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("image_encoder", WanAutoImageEncoderStep),
        ("vae_image_encoder", WanAutoVaeImageEncoderStep),
        ("denoise", WanAutoDenoiseStep),
        ("decode", WanImageVaeDecoderStep),
    ]
)

# wan2.2 presets

TEXT2VIDEO_BLOCKS_WAN22 = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("input", WanTextInputStep),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("denoise", Wan22DenoiseStep),
        ("decode", WanImageVaeDecoderStep),
    ]
)

IMAGE2VIDEO_BLOCKS_WAN22 = InsertableDict(
    [
        ("image_resize", WanImageResizeStep),
        ("vae_image_encoder", WanImage2VideoVaeImageEncoderStep),
        ("input", WanTextInputStep),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("denoise", Wan22DenoiseStep),
        ("decode", WanImageVaeDecoderStep),
    ]
)

AUTO_BLOCKS_WAN22 = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("vae_image_encoder", WanAutoVaeImageEncoderStep),
        ("denoise", Wan22AutoDenoiseStep),
        ("decode", WanImageVaeDecoderStep),
    ]
)

# presets all blocks (wan and wan22)


ALL_BLOCKS = {
    "wan2.1": {
        "text2video": TEXT2VIDEO_BLOCKS,
        "image2video": IMAGE2VIDEO_BLOCKS,
        "flf2v": FLF2V_BLOCKS,
        "auto": AUTO_BLOCKS,
    },
    "wan2.2": {
        "text2video": TEXT2VIDEO_BLOCKS_WAN22,
        "image2video": IMAGE2VIDEO_BLOCKS_WAN22,
        "auto": AUTO_BLOCKS_WAN22,
    },
}
