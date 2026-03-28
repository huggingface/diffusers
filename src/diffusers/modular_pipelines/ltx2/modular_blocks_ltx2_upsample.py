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

import torch

from ...configuration_utils import FrozenDict
from ...models.autoencoders import AutoencoderKLLTX2Video
from ...pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


class LTX2UpsamplePrepareStep(ModularPipelineBlocks):
    """Prepare latents for upsampling: accepts either video frames or pre-computed latents."""

    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Prepare latents for the latent upsampler, from either video input or pre-computed latents"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("video", description="Video frames to encode and upsample"),
            InputParam("latents", type_hint=torch.Tensor, description="Pre-computed latents to upsample"),
            InputParam("latents_normalized", default=False, type_hint=bool),
            InputParam("height", default=512, type_hint=int),
            InputParam("width", default=768, type_hint=int),
            InputParam("num_frames", default=121, type_hint=int),
            InputParam("spatial_patch_size", default=1, type_hint=int),
            InputParam("temporal_patch_size", default=1, type_hint=int),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Prepared latents for upsampling"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        video = block_state.video
        latents = block_state.latents
        height = block_state.height
        width = block_state.width
        num_frames = block_state.num_frames
        generator = block_state.generator

        vae_spatial_compression_ratio = components.vae.spatial_compression_ratio
        vae_temporal_compression_ratio = components.vae.temporal_compression_ratio

        if latents is not None:
            if latents.ndim == 3:
                latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1
                latent_height = height // vae_spatial_compression_ratio
                latent_width = width // vae_spatial_compression_ratio
                latents = _unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    block_state.spatial_patch_size,
                    block_state.temporal_patch_size,
                )
            if block_state.latents_normalized:
                latents = _denormalize_latents(
                    latents,
                    components.vae.latents_mean,
                    components.vae.latents_std,
                    components.vae.config.scaling_factor,
                )
            block_state.latents = latents.to(device=device, dtype=torch.float32)
        elif video is not None:
            if isinstance(video, list):
                num_frames = len(video)
            if num_frames % vae_temporal_compression_ratio != 1:
                num_frames = num_frames // vae_temporal_compression_ratio * vae_temporal_compression_ratio + 1
                if isinstance(video, list):
                    video = video[:num_frames]

            video = components.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(device=device, dtype=torch.float32)

            init_latents = [
                retrieve_latents(components.vae.encode(vid.unsqueeze(0)), generator) for vid in video
            ]
            block_state.latents = torch.cat(init_latents, dim=0).to(torch.float32)
        else:
            raise ValueError("One of `video` or `latents` must be provided.")

        self.set_block_state(state, block_state)
        return components, state


class LTX2UpsampleStep(ModularPipelineBlocks):
    """Run the latent upsampler model with optional AdaIN and tone mapping."""

    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Run the latent upsampler model with optional AdaIN filtering and tone mapping"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("latent_upsampler", LTX2LatentUpsamplerModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("adain_factor", default=0.0, type_hint=float),
            InputParam("tone_map_compression_ratio", default=0.0, type_hint=float),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Upsampled latents"),
        ]

    @staticmethod
    def adain_filter_latent(latents: torch.Tensor, reference_latents: torch.Tensor, factor: float = 1.0):
        result = latents.clone()
        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)
                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean
        result = torch.lerp(latents, result, factor)
        return result

    @staticmethod
    def tone_map_latents(latents: torch.Tensor, compression: float) -> torch.Tensor:
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)
        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term
        return latents * scales

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents.to(components.latent_upsampler.dtype)
        reference_latents = latents

        latents_upsampled = components.latent_upsampler(latents)

        if block_state.adain_factor > 0.0:
            latents = self.adain_filter_latent(latents_upsampled, reference_latents, block_state.adain_factor)
        else:
            latents = latents_upsampled

        if block_state.tone_map_compression_ratio > 0.0:
            latents = self.tone_map_latents(latents, block_state.tone_map_compression_ratio)

        block_state.latents = latents

        self.set_block_state(state, block_state)
        return components, state


class LTX2UpsamplePostprocessStep(ModularPipelineBlocks):
    """Decode upsampled latents to video frames."""

    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Decode upsampled latents into video frames"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("output_type", default="pil", type_hint=str),
            InputParam("decode_timestep", default=0.0),
            InputParam("decode_noise_scale", default=None),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("videos", description="Decoded video frames"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents

        if block_state.output_type == "latent":
            block_state.videos = latents
        else:
            batch_size = latents.shape[0]
            device = latents.device

            if not components.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(
                    latents.shape, generator=block_state.generator, device=device, dtype=latents.dtype
                )
                decode_timestep = block_state.decode_timestep
                decode_noise_scale = block_state.decode_noise_scale

                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = components.vae.decode(latents, timestep, return_dict=False)[0]
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )

        self.set_block_state(state, block_state)
        return components, state


# ====================
# UPSAMPLE BLOCKS
# ====================


class LTX2UpsampleBlocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for LTX2 latent upsampling."""

    model_name = "ltx2"
    block_classes = [
        LTX2UpsamplePrepareStep,
        LTX2UpsampleStep,
        LTX2UpsamplePostprocessStep,
    ]
    block_names = ["prepare", "upsample", "postprocess"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX2 latent upsampling (stage1 -> upsample -> stage2)."

    @property
    def outputs(self):
        return [OutputParam("videos")]


class LTX2UpsampleCorePrepareStep(LTX2UpsamplePrepareStep):
    """Upsample prepare step for the full pipeline: latents_normalized defaults to True."""

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("video", description="Video frames to encode and upsample"),
            InputParam("latents", type_hint=torch.Tensor, description="Pre-computed latents to upsample"),
            InputParam("latents_normalized", default=True, type_hint=bool),
            InputParam("height", default=512, type_hint=int),
            InputParam("width", default=768, type_hint=int),
            InputParam("num_frames", default=121, type_hint=int),
            InputParam("spatial_patch_size", default=1, type_hint=int),
            InputParam("temporal_patch_size", default=1, type_hint=int),
            InputParam("generator"),
        ]


class LTX2UpsampleCoreBlocks(SequentialPipelineBlocks):
    """Upsample blocks for the full pipeline: prepare + upsample only (no decode).

    Outputs 5D latents (not decoded video), suitable for chaining into Stage2.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2UpsampleCorePrepareStep,
        LTX2UpsampleStep,
    ]
    block_names = ["prepare", "upsample"]

    @property
    def description(self):
        return "Latent upsample blocks (no decode) for use within the full pipeline."

    @property
    def outputs(self):
        return [OutputParam("latents")]
