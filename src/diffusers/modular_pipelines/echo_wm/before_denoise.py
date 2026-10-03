# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch

from ...models import EchoWMTransformer3DModel
from ...utils.torch_utils import randn_tensor
from ..ltx2.before_denoise import (
    LTX2Image2VideoPrepareLatentsStep,
    LTX2PrepareAudioLatentsStep,
    LTX2PrepareCoordsStep,
    LTX2PrepareLatentsStep,
    _create_noised_state,
    _normalize_audio_latents,
    _normalize_latents,
    _pack_audio_latents,
    _pack_latents,
)
from ..modular_pipeline_utils import ComponentSpec


class EchoWMPrepareLatentsStep(LTX2PrepareLatentsStep):
    """Draw video noise in the reference's packed layout and transformer dtype."""

    model_name = "echo-wm"

    @property
    def expected_components(self):
        return [ComponentSpec("transformer", EchoWMTransformer3DModel)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        device, dtype = components._execution_device, components.transformer.dtype
        spatial_patch = components.transformer_spatial_patch_size
        temporal_patch = components.transformer_temporal_patch_size
        if block_state.noise_scale is None:
            block_state.noise_scale = 0.0

        if block_state.latents is not None:
            latents = block_state.latents.to(device=device, dtype=dtype)
            if latents.ndim == 5:
                latents = _normalize_latents(
                    latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
                )
                latents = _pack_latents(latents, spatial_patch, temporal_patch)
            block_state.latents = _create_noised_state(latents, block_state.noise_scale, block_state.generator)
        else:
            frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
            height = block_state.height // components.vae_spatial_compression_ratio
            width = block_state.width // components.vae_spatial_compression_ratio
            shape = (
                block_state.batch_size * block_state.num_videos_per_prompt,
                (frames // temporal_patch) * (height // spatial_patch) * (width // spatial_patch),
                components.transformer.config.in_channels * temporal_patch * spatial_patch**2,
            )
            block_state.latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=dtype)
        self.set_block_state(state, block_state)
        return components, state


class EchoWMImage2VideoPrepareLatentsStep(LTX2Image2VideoPrepareLatentsStep):
    """Keep the image-conditioned sample in the reference model's working dtype."""

    model_name = "echo-wm"

    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        state.set("latents", state.get("latents").to(components.transformer.dtype))
        return components, state


class EchoWMPrepareAudioLatentsStep(LTX2PrepareAudioLatentsStep):
    """Draw audio noise directly in the packed time-by-channel/mel layout."""

    model_name = "echo-wm"

    @property
    def expected_components(self):
        return [*super().expected_components, ComponentSpec("transformer", EchoWMTransformer3DModel)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        device, dtype = components._execution_device, components.transformer.dtype
        frames_per_second = (
            components.audio_sampling_rate
            / components.audio_hop_length
            / components.audio_vae_temporal_compression_ratio
        )
        audio_num_frames = round(block_state.num_frames / block_state.frame_rate * frames_per_second)
        if block_state.audio_latents is not None:
            latents = block_state.audio_latents.to(device=device, dtype=dtype)
            if latents.ndim == 4:
                audio_num_frames = latents.shape[2]
                latents = _pack_audio_latents(latents)
            latents = _normalize_audio_latents(latents, components.audio_latents_mean, components.audio_latents_std)
            block_state.audio_latents = _create_noised_state(latents, block_state.noise_scale, block_state.generator)
        else:
            shape = (
                block_state.batch_size * block_state.num_videos_per_prompt,
                audio_num_frames,
                components.audio_vae.config.latent_channels
                * (components.audio_vae.config.mel_bins // components.audio_vae_mel_compression_ratio),
            )
            block_state.audio_latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=dtype
            )
        block_state.audio_num_frames = audio_num_frames
        self.set_block_state(state, block_state)
        return components, state


class EchoWMPrepareCoordsStep(LTX2PrepareCoordsStep):
    """Match VideoLatentTools' video-coordinate dtype; audio coordinates stay FP32."""

    model_name = "echo-wm"

    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        state.set("video_coords", state.get("video_coords").to(components.transformer.dtype))
        return components, state
