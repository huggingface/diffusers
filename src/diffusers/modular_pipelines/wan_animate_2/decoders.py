# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import PIL.Image
import torch

from ...configuration_utils import FrozenDict
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .video_processor import WanAnimate2VideoProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAnimate2DecodeStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step that assembles the final video from the per-segment decoded frames: concatenates the segments, "
            "trims the zigzag padding, crops the reference image's letterbox bars back off, and postprocesses. "
            "The VAE decode itself happens per segment inside the denoise loop, because each segment conditions "
            "on the previous segment's decoded pixels."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "video_processor",
                WanAnimate2VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8, "spatial_patch_size": (2, 2), "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "segment_frames",
                required=True,
                type_hint=list[torch.Tensor],
                description="Per-segment decoded frames from the segment denoise loop, each `[1, 3, T, H, W]`",
            ),
            InputParam(
                "real_frame_len",
                required=True,
                type_hint=int,
                description="Number of frames in the driving video before zigzag padding; the output is trimmed to it",
            ),
            InputParam(
                "crop_region",
                required=True,
                type_hint=tuple[int, int, int, int],
                description="`(top, left, height, width)` of the reference image content inside the letterboxed "
                "frame, from the image preprocess step",
            ),
            InputParam(
                "output_type", default="np", type_hint=str, description="The output type of the decoded videos"
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "videos",
                type_hint=list[list[PIL.Image.Image]] | list[torch.Tensor] | list[np.ndarray],
                description="The generated videos, can be a PIL.Image.Image, torch.Tensor or a numpy array",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        video = torch.cat(block_state.segment_frames, dim=2)[:, :, : block_state.real_frame_len]
        crop_top, crop_left, crop_height, crop_width = block_state.crop_region
        video = video[:, :, :, crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
