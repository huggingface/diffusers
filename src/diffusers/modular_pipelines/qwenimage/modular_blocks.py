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

from .encoders import QwenImageTextEncoderStep, QwenImageEditTextEncoderStep, QwenImageVaeEncoderStep
from .decoders import QwenImageDecodeStep
from .denoise import QwenImageDenoiseStep, QwenImageEditDenoiseStep
from .before_denoise import QwenImageInputStep, QwenImagePrepareLatentsStep, QwenImageSetTimestepsStep, QwenImagePrepareAdditionalInputsStep, QwenImagePrepareImageLatentsStep, QwenImageEditPrepareAdditionalInputsStep, QwenImageImageResizeStep

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict

logger = logging.get_logger(__name__)


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

EDIT_BLOCKS = InsertableDict(
    [
        ("image_resize", QwenImageImageResizeStep),
        ("text_encoder", QwenImageEditTextEncoderStep),
        ("vae_encoder", QwenImageVaeEncoderStep),
        ("input", QwenImageInputStep),
        ("prepare_image_latents", QwenImagePrepareImageLatentsStep),
        ("prepare_latents", QwenImagePrepareLatentsStep),
        ("set_timesteps", QwenImageSetTimestepsStep),
        ("prepare_additional_inputs", QwenImageEditPrepareAdditionalInputsStep),
        ("denoise", QwenImageEditDenoiseStep),
        ("decode", QwenImageDecodeStep),
    ]
    )

ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "edit": EDIT_BLOCKS,
}