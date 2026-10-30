# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLHunyuanVideo
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import KandinskyS2VModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def decode_latents(vae: AutoencoderKLHunyuanVideo, latents: torch.Tensor) -> torch.Tensor:
    """``(T, H, W, C)`` latent -> ``(1, 3, T*4-3, H*8, W*8)`` in ``[-1, 1]``.

    The decode runs under bf16 autocast rather than in the VAE's own fp16, as the native pipeline does: decoding the
    same latent the two ways differs by 4.4e-3 relative.
    """
    images = latents.reshape(1, *latents.shape)
    images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
    device_type = latents.device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        decoded = vae.decode(images.to(vae.dtype)).sample
    # Autocast is a choice about the precision of the decode, not about the dtype handed back; the widening is exact.
    return decoded.float()


class KandinskyS2VDecodeStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Decode step that decodes each chunk's latents, joins the chunks and trims the video back to the length "
            "of the speech track."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLHunyuanVideo),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latent_chunks",
                required=True,
                type_hint=list,
                description="Denoised latents of each chunk, each shaped "
                "(num_latent_frames, latent_height, latent_width, channels).",
            ),
            InputParam("track_seconds", required=True, type_hint=float),
            InputParam("fps", default=25, type_hint=int),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        decoded = [decode_latents(components.vae, latents).cpu() for latents in block_state.latent_chunks]
        frames = torch.cat(decoded, dim=2)  # (1, C, T, H, W)
        frames = frames[:, :, : min(frames.shape[2], int(block_state.track_seconds * block_state.fps))]
        block_state.videos = components.video_processor.postprocess_video(frames, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
