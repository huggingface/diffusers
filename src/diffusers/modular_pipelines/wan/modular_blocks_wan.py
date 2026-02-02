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
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanTextInputStep,
)
from .decoders import WanVaeDecoderStep
from .denoise import (
    WanDenoiseStep,
)
from .encoders import (
    WanTextEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. DENOISE
# ====================


# inputs(text) -> set_timesteps -> prepare_latents -> denoise
class WanCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "wan"
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


# ====================
# 2. BLOCKS (Wan2.1 text2video)
# ====================


class WanBlocks(SequentialPipelineBlocks):
    model_name = "wan"
    block_classes = [
        WanTextEncoderStep,
        WanCoreDenoiseStep,
        WanVaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return (
            "Modular pipeline blocks for Wan2.1.\n"
            + "- `WanTextEncoderStep` is used to encode the text\n"
            + "- `WanCoreDenoiseStep` is used to denoise the latents\n"
            + "- `WanVaeDecoderStep` is used to decode the latents to images"
        )
