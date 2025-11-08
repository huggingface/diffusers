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
    WanTextInputStep,
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanInputsDynamicStep,
    WanPrepareFirstFrameLatentsStep,
)
from .decoders import WanImageVaeDecoderStep
from .denoise import WanDenoiseStep, WanImage2VideoDenoiseStep
from .encoders import WanTextEncoderStep, WanImageResizeStep, WanImageEncoderStep, WanVaeImageEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


#  text2vid
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


# image2video

## iamge encoder
class WanImage2VideoImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan"
    block_classes = [WanImageResizeStep, WanImageEncoderStep]
    block_names = ["image_resize", "image_encoder"]

    @property
    def description(self):
        return "Image2Video Image Encoder step that resize the image and encode the image to generate the image embeddings"



# vae encoder
class WanImage2VideoVaeImageEncoderStep(SequentialPipelineBlocks):
    model_name = "wan"
    block_classes = [WanImageResizeStep, WanVaeImageEncoderStep]
    block_names = ["image_resize", "vae_image_encoder"]

    @property
    def description(self):
        return "Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent representation"



class WanImage2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        WanTextInputStep,
        WanInputsDynamicStep(image_latent_inputs=["condition_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanPrepareFirstFrameLatentsStep,
        WanImage2VideoDenoiseStep,
    ]
    block_names = ["input", "additional_inputs", "set_timesteps", "prepare_latents", "prepare_first_frame_latents", "denoise"]

    @property
    def description(self):
        return (
            "denoise block that takes encoded text and image latent conditions and runs the denoising process.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `WanTextInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `WanInputsDynamicStep` is used to adjust the batch size of the latent conditions\n"
            + " - `WanSetTimestepsStep` is used to set the timesteps\n"
            + " - `WanPrepareLatentsStep` is used to prepare the latents\n"
            + " - `WanPrepareConditionLatentsStep` is used to prepare the latent conditions\n"
            + " - `WanImage2VideoDenoiseStep` is used to denoise the latents\n"
        )



# auto blocks

class WanAutoImageEncoderStep(AutoPipelineBlocks):
    block_classes = [WanImage2VideoImageEncoderStep]
    block_names = ["image_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return ("Image Encoder step that encode the image to generate the image embeddings"
                + "This is an auto pipeline block that works for image2video tasks."
                + " - `WanImage2VideoImageEncoderStep` (image2video) is used when `image` is provided."
                + " - if `image` is not provided, step will be skipped.")

class WanAutoVaeImageEncoderStep(AutoPipelineBlocks):
    block_classes = [WanImage2VideoVaeImageEncoderStep]
    block_names = ["vae_image_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return ("Vae Image Encoder step that encode the image to generate the image latents"
                + "This is an auto pipeline block that works for image2video tasks."
                + " - `WanImage2VideoVaeImageEncoderStep` (image2video) is used when `image` is provided."
                + " - if `image` is not provided, step will be skipped.")


class WanAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        WanImage2VideoCoreDenoiseStep,
        WanCoreDenoiseStep,
    ]
    block_names = ["image2video", "text2video"]
    block_trigger_inputs = ["condition_latents", None]

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


# text2vid
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
        ("additional_inputs", WanInputsDynamicStep(image_latent_inputs=["condition_latents"])),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
        ("denoise", WanImage2VideoCoreDenoiseStep),
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


ALL_BLOCKS = {
    "text2video": TEXT2VIDEO_BLOCKS,
    "image2video": IMAGE2VIDEO_BLOCKS,
    "auto": AUTO_BLOCKS,
}
