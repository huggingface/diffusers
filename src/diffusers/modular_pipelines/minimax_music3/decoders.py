# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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
import torch

from ...models import MiniMaxMusic3Vocoder
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxMusic3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Neighboring windows share 172 latent frames: when stitching the decoded waveforms, every window after the first
# drops its leading 86 latent frames and every window before the last drops its trailing 344 - 86 latent frames, so
# the kept spans tile the full song.
_CROP_LEFT_LATENT = 86
_CROP_RIGHT_LATENT = 344 - 86


class MiniMaxMusic3VocoderDecodeStep(ModularPipelineBlocks):
    model_name = "minimax-music3"

    @property
    def description(self) -> str:
        return (
            "Decode step that vocodes each window's Flow-VAE latents into a waveform, crops the overlapping spans, "
            "and stitches the windows into the final stereo waveform at 44.1 kHz."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vocoder", MiniMaxMusic3Vocoder)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latent_chunks",
                required=True,
                type_hint=list,
                description="List of per-window denoised latent tensors (uncropped). Can be generated in denoise step.",
            ),
            InputParam("output_type", default="np", type_hint=str, description="Output format: 'np' or 'pt'."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "audios",
                type_hint=torch.Tensor | np.ndarray,
                description="The generated stereo waveform of shape `(batch, channels, samples)` in `[-1, 1]`.",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.output_type not in ["np", "pt"]:
            raise ValueError(f"Invalid output_type: {block_state.output_type}")

    @torch.no_grad()
    def __call__(self, components: MiniMaxMusic3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        hop_length = components.latent_hop_length
        num_chunks = len(block_state.latent_chunks)
        waveform_chunks = []
        for chunk_index, latents in enumerate(block_state.latent_chunks):
            waveform = components.vocoder(latents.to(components.vocoder.dtype))
            left = 0 if chunk_index == 0 else _CROP_LEFT_LATENT * hop_length
            right = 0 if chunk_index == num_chunks - 1 else _CROP_RIGHT_LATENT * hop_length
            waveform_chunks.append(waveform[..., left : waveform.shape[-1] - right])

        audios = torch.cat(waveform_chunks, dim=-1).float().clamp(-1.0, 1.0)
        if block_state.output_type == "np":
            audios = audios.cpu().numpy()
        block_state.audios = audios

        self.set_block_state(state, block_state)
        return components, state
