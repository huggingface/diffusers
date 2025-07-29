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
from .before_denoise import FluxInputStep, FluxPrepareLatentsStep, FluxSetTimestepsStep
from .decoders import FluxDecodeStep
from .denoise import FluxDenoiseStep
from .encoders import FluxTextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# before_denoise: text2vid
class FluxBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        FluxInputStep,
        FluxPrepareLatentsStep,
        FluxSetTimestepsStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `FluxInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `FluxPrepareLatentsStep` is used to prepare the latents\n"
            + " - `FluxSetTimestepsStep` is used to set the timesteps\n"
        )


# before_denoise: all task (text2vid,)
class FluxAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [FluxBeforeDenoiseStep]
    block_names = ["text2image"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2image.\n"
            + " - `FluxBeforeDenoiseStep` (text2image) is used.\n"
        )


# denoise: text2image
class FluxAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [FluxDenoiseStep]
    block_names = ["denoise"]
    block_trigger_inputs = [None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2image tasks."
            " - `FluxDenoiseStep` (denoise) for text2image tasks."
        )


# decode: all task (text2img, img2img, inpainting)
class FluxAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [FluxDecodeStep]
    block_names = ["non-inpaint"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return "Decode step that decode the denoised latents into videos outputs.\n - `FluxDecodeStep`"


# text2image
class FluxAutoBlocks(SequentialPipelineBlocks):
    block_classes = [FluxTextEncoderStep, FluxAutoBeforeDenoiseStep, FluxAutoDenoiseStep, FluxAutoDecodeStep]
    block_names = ["text_encoder", "before_denoise", "denoise", "decoder"]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image using Flux.\n"
            + "- for text-to-image generation, all you need to provide is `prompt`"
        )


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep),
        ("input", FluxInputStep),
        ("prepare_latents", FluxPrepareLatentsStep),
        # Setting it after preparation of latents because we rely on `latents`
        # to calculate `img_seq_len` for `shift`.
        ("set_timesteps", FluxSetTimestepsStep),
        ("denoise", FluxDenoiseStep),
        ("decode", FluxDecodeStep),
    ]
)


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep),
        ("before_denoise", FluxAutoBeforeDenoiseStep),
        ("denoise", FluxAutoDenoiseStep),
        ("decode", FluxAutoDecodeStep),
    ]
)


ALL_BLOCKS = {"text2image": TEXT2IMAGE_BLOCKS, "auto": AUTO_BLOCKS}
