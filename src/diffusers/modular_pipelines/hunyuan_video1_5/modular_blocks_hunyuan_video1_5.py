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
    HunyuanVideo15Image2VideoPrepareLatentsStep,
    HunyuanVideo15PrepareLatentsStep,
    HunyuanVideo15SetTimestepsStep,
    HunyuanVideo15TextInputStep,
)
from .decoders import HunyuanVideo15VaeDecoderStep
from .denoise import HunyuanVideo15DenoiseStep, HunyuanVideo15Image2VideoDenoiseStep
from .encoders import HunyuanVideo15TextEncoderStep


logger = logging.get_logger(__name__)


# auto_docstring
class HunyuanVideo15CoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextInputStep,
        HunyuanVideo15SetTimestepsStep,
        HunyuanVideo15PrepareLatentsStep,
        HunyuanVideo15DenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class HunyuanVideo15Blocks(SequentialPipelineBlocks):
    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextEncoderStep,
        HunyuanVideo15CoreDenoiseStep,
        HunyuanVideo15VaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for HunyuanVideo 1.5 text-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]


# auto_docstring
class HunyuanVideo15Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextInputStep,
        HunyuanVideo15SetTimestepsStep,
        HunyuanVideo15Image2VideoPrepareLatentsStep,
        HunyuanVideo15Image2VideoDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block for image-to-video that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class HunyuanVideo15Image2VideoBlocks(SequentialPipelineBlocks):
    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextEncoderStep,
        HunyuanVideo15Image2VideoCoreDenoiseStep,
        HunyuanVideo15VaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for HunyuanVideo 1.5 image-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
