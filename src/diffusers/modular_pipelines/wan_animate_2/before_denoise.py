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

import math

import numpy as np
import torch

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAnimate2PrepareSegmentsStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step that computes the segment-invariant geometry for the segment loop. The zigzag padding makes "
            "every segment exactly `segment_frame_length` frames, so the latent grid, the packed sequence lengths, and the "
            "noise shape are the same for all segments and are computed once here."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "segment_frame_length",
                type_hint=int,
                default=81,
                description="The number of frames in each inference segment",
            ),
            InputParam(
                "reference_image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The reference conditioning tensor `[20, 1, latent_height, latent_width]`; "
                "provides the latent grid",
            ),
            InputParam(
                "driving_video_pixels",
                required=True,
                type_hint=torch.Tensor,
                description="The preprocessed driving video `[1, 3, T, H, W]`, from the video preprocess step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "grid_sizes_ref",
                type_hint=torch.Tensor,
                description="Post-patch latent grid `[[T, H/2, W/2]]` of a driving-video segment, used as the "
                "offset grid of the reference-extraction pass and the reference grid of the denoising passes",
            ),
            OutputParam(
                "max_seq_len",
                type_hint=int,
                description="Packed sequence length of the generation tokens",
            ),
            OutputParam(
                "max_seq_len_ref",
                type_hint=int,
                description="Packed sequence length of the reference tokens",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latent_height, latent_width = block_state.reference_image_latents.shape[-2:]

        expected = (
            latent_height * components.vae_scale_factor_spatial,
            latent_width * components.vae_scale_factor_spatial,
        )
        if tuple(block_state.driving_video_pixels.shape[-2:]) != expected:
            raise ValueError(
                f"`driving_video_pixels` is letterboxed to {tuple(block_state.driving_video_pixels.shape[-2:])} but "
                f"the reference image conditioning is {expected} — the video and image preprocess steps must use the "
                "same `height`/`width`."
            )

        latent_segment_frames = (block_state.segment_frame_length - 1) // components.vae_scale_factor_temporal + 1
        ref_shape = [latent_segment_frames, latent_height, latent_width]
        ref_shape_post = [ref_shape[0], ref_shape[1] // 2, ref_shape[2] // 2]
        block_state.grid_sizes_ref = torch.tensor([ref_shape_post], dtype=torch.long)

        # The noise tensor carries one extra latent frame: the reference image's slot.
        latent_noise_frames = latent_segment_frames + 1
        block_state.max_seq_len = int(math.ceil(np.prod([latent_noise_frames, latent_height // 2, latent_width // 2])))
        block_state.max_seq_len_ref = int(math.ceil(np.prod(ref_shape) // 4))

        self.set_block_state(state, block_state)
        return components, state
