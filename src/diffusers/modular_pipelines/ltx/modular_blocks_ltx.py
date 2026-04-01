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
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    LTXImage2VideoPrepareLatentsStep,
    LTXPrepareLatentsStep,
    LTXSetTimestepsStep,
    LTXTextInputStep,
)
from .decoders import LTXVaeDecoderStep
from .denoise import LTXDenoiseStep, LTXImage2VideoDenoiseStep
from .encoders import LTXTextEncoderStep


logger = logging.get_logger(__name__)


# auto_docstring
class LTXCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "ltx"
    block_classes = [
        LTXTextInputStep,
        LTXSetTimestepsStep,
        LTXPrepareLatentsStep,
        LTXDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class LTXImage2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "ltx"
    block_classes = [
        LTXTextInputStep,
        LTXSetTimestepsStep,
        LTXImage2VideoPrepareLatentsStep,
        LTXImage2VideoDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block for image-to-video that takes encoded conditions and an image, and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class LTXBlocks(SequentialPipelineBlocks):
    model_name = "ltx"
    block_classes = [
        LTXTextEncoderStep,
        LTXCoreDenoiseStep,
        LTXVaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX Video text-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]


# auto_docstring
class LTXImage2VideoBlocks(SequentialPipelineBlocks):
    model_name = "ltx"
    block_classes = [
        LTXTextEncoderStep,
        LTXImage2VideoCoreDenoiseStep,
        LTXVaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX Video image-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
