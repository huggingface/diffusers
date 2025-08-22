# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from .encoders import QwenImageTextEncoderStep
from .decoders import QwenImageDecodeStep
from .denoise import QwenImageDenoiseStep
from .before_denoise import QwenImageInputStep, QwenImagePrepareLatentsStep, QwenImageSetTimestepsStep, QwenImagePrepareAdditionalInputsStep

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict

logger = logging.get_logger(__name__)


class QwenImageBeforeDenoiseStep(SequentialPipelineBlocks):

    block_classes = [
        QwenImageInputStep,
        QwenImagePrepareLatentsStep,
        QwenImageSetTimestepsStep,
        QwenImagePrepareAdditionalInputsStep,
    ]

    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_additional_inputs",
    ]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `QwenImageInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `QwenImagePrepareLatentsStep` is used to prepare the latents\n"
            + " - `QwenImageSetTimestepsStep` is used to set the timesteps\n"
            + " - `QwenImagePrepareAdditionalInputsStep` is used to prepare the additional inputs for the model\n"
        )
        

TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep),
        ("input", QwenImageInputStep),
        ("prepare_latents", QwenImagePrepareLatentsStep),
        ("set_timesteps", QwenImageSetTimestepsStep),
        ("prepare_additional_inputs", QwenImagePrepareAdditionalInputsStep),
        ("denoise", QwenImageDenoiseStep),
        ("decode", QwenImageDecodeStep),
    ]
    )

ALL_BLOCKS = {"text2image": TEXT2IMAGE_BLOCKS}