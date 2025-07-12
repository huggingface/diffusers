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
from .encoders import WanTextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


# text2vid
class WanAutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        WanTextEncoderStep,
        WanBeforeDenoiseStep,
    ]
    block_names = [
        "text_encoder",
        "before_denoise",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-video using Wan.\n"
            + "- for text-to-video generation, all you need to provide is `prompt`"
        )


# before_denoise: all task (text2vid,)
class WanAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        WanBeforeDenoiseStep,
    ]
    block_names = ["text2vid"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2vid.\n"
            + " - `WanBeforeDenoiseStep` (text2vid) is used.\n"
        )


TEXT2VIDEO_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("input", WanInputStep),
        ("set_timesteps", WanSetTimestepsStep),
        ("prepare_latents", WanPrepareLatentsStep),
    ]
)


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanTextEncoderStep),
        ("before_denoise", WanAutoBeforeDenoiseStep),
    ]
)


ALL_BLOCKS = {
    "text2video": TEXT2VIDEO_BLOCKS,
    "auto": AUTO_BLOCKS,
}
