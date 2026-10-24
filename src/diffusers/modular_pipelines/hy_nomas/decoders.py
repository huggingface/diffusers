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
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...models.autoencoders.autoencoder_pixelumm import PixelUMMDecoder
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HyNomasGenerationState, HyNomasModularPipeline


# auto_docstring
class HyNomasDecodeStep(ModularPipelineBlocks):
    """
    Decode the accumulated world latents with the distilled PixelUMM diffusion decoder.

      Components:
          decoder (`PixelUMMDecoder`) video_processor (`VideoProcessor`)

      Inputs:
          latents (`Tensor`):
              Denoised latents for the accumulated world, including all earlier segments, shaped (1, 32, latent_frames,
              latent_height, latent_width).
          session (`HyNomasGenerationState`, *optional*):
              History carrying each segment's exact visible pixel length so its decoder padding can be removed; omitted
              states use (latent_frames - 1) * 4 + 1 frames.
          output_type (`str`, *optional*, defaults to np):
              Output representation: np for NumPy arrays, pt for tensors, pil for frame images, or latent to skip
              decoding.
          decoder_generator (`Generator`, *optional*):
              Independent PixelUMM noise generator; only its initial seed is used, without advancing its state. If
              omitted, the decoder configuration seed is used (42 by default).

      Outputs:
          videos (`Tensor | ndarray | list`):
              Complete accumulated video including prior history: pt has shape (1, frames, 3, height, width), np has
              shape (1, frames, height, width, 3), and pil is a nested list of frames. The latent output preserves the
              input latent tensor.
    """

    model_name = "hy-nomas"

    @property
    def description(self):
        return "Decode the accumulated world latents with the distilled PixelUMM diffusion decoder."

    @property
    def expected_components(self):
        return [
            ComponentSpec("decoder", PixelUMMDecoder),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self):
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Denoised latents for the accumulated world, including all earlier segments, shaped (1, 32, latent_frames, latent_height, latent_width).",
                required=True,
            ),
            InputParam(
                "session",
                type_hint=HyNomasGenerationState,
                description="History carrying each segment's exact visible pixel length so its decoder padding can be removed; omitted states use (latent_frames - 1) * 4 + 1 frames.",
            ),
            InputParam(
                "output_type",
                type_hint=str,
                description="Output representation: np for NumPy arrays, pt for tensors, pil for frame images, or latent to skip decoding.",
                default="np",
            ),
            InputParam(
                "decoder_generator",
                type_hint=torch.Generator,
                description="Independent PixelUMM noise generator; only its initial seed is used, without advancing its state. If omitted, the decoder configuration seed is used (42 by default).",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "videos",
                type_hint=torch.Tensor | np.ndarray | list[list[PIL.Image.Image]],
                description="Complete accumulated video including prior history: pt has shape (1, frames, 3, height, width), np has shape (1, frames, height, width, 3), and pil is a nested list of frames. The latent output preserves the input latent tensor.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state) -> tuple[HyNomasModularPipeline, PipelineState]:
        s = self.get_block_state(state)
        if s.output_type == "latent":
            s.videos = s.latents
        else:
            video = components.decoder.decode(
                s.latents.to(device=components._execution_device, dtype=components.decoder.dtype),
                generator=s.decoder_generator,
                return_dict=False,
            )[0]
            if s.session is None:
                video = video[:, :, : (s.latents.shape[2] - 1) * 4 + 1]
            else:
                chunks = video.split([chunk.shape[2] * 4 for chunk in s.session.latent_chunks], dim=2)
                video = torch.cat(
                    [chunk[:, :, :length] for chunk, length in zip(chunks, s.session.pixel_lengths)], dim=2
                )
            # Preserve the released decoder's arithmetic precision before converting output formats.
            video = (video * 0.5 + 0.5).clamp(0, 1)
            s.videos = components.video_processor.postprocess_video(video, output_type=s.output_type)
        self.set_block_state(state, s)
        return components, state
