# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import Krea2PositionIdsStep, Krea2PrepareLatentsStep, Krea2SetTimestepsStep
from .decoders import Krea2AfterDenoiseStep, Krea2DecoderStep, Krea2ProcessImagesOutputStep
from .denoise import Krea2DenoiseStep
from .encoders import Krea2TextEncoderStep
from .inputs import Krea2TextInputsStep


class Krea2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Prepare Krea 2 inputs and denoise packed latents.
    """

    model_name = "krea2"
    block_classes = [
        Krea2TextInputsStep(),
        Krea2PrepareLatentsStep(),
        Krea2SetTimestepsStep(),
        Krea2PositionIdsStep(),
        Krea2DenoiseStep(),
        Krea2AfterDenoiseStep(),
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_position_ids",
        "denoise",
        "after_denoise",
    ]

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


class Krea2DecodeStep(SequentialPipelineBlocks):
    """
    Decode Krea 2 latents and postprocess images.
    """

    model_name = "krea2"
    block_classes = [Krea2DecoderStep(), Krea2ProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def outputs(self):
        return [OutputParam.template("images")]


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", Krea2TextEncoderStep()),
        ("denoise", Krea2CoreDenoiseStep()),
        ("decode", Krea2DecodeStep()),
    ]
)


class Krea2AutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image generation using Krea 2.
    """

    model_name = "krea2"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()
    _workflow_map = {
        "text2image": {"prompt": True},
    }

    @property
    def outputs(self):
        return [OutputParam.template("images")]
