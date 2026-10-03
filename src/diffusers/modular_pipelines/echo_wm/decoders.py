# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch

from ..ltx2.decoders import LTX2VaeDecoderStep
from ..modular_pipeline import PipelineState
from ..modular_pipeline_utils import InputParam


class EchoWMVaeDecoderStep(LTX2VaeDecoderStep):
    """Decode video with Echo-WM's default spatial and temporal tiling parameters."""

    model_name = "echo-wm"

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam(
                "vae_tiling",
                type_hint=bool,
                default=True,
                description="Enable spatial and temporal VAE decoding tiles to reduce peak memory usage.",
            ),
            InputParam(
                "vae_tile_size",
                type_hint=int,
                default=512,
                description="Spatial tile long-side size in pixels; the short side follows the video aspect ratio.",
            ),
            InputParam(
                "vae_tile_overlap",
                type_hint=int,
                default=64,
                description="Spatial tile overlap in pixels.",
            ),
            InputParam(
                "vae_temporal_tile_size",
                type_hint=int,
                default=64,
                description="Temporal tile size in sample frames, excluding the causal boundary frame.",
            ),
            InputParam(
                "vae_temporal_tile_overlap",
                type_hint=int,
                default=24,
                description="Temporal tile overlap in sample frames.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae
        tiling_settings = {
            "use_tiling": block_state.vae_tiling,
            "use_framewise_decoding": block_state.vae_tiling,
        }
        if block_state.vae_tiling:
            spatial_scale = vae.spatial_compression_ratio
            temporal_scale = vae.temporal_compression_ratio
            for size, overlap, scale, name in (
                (block_state.vae_tile_size, block_state.vae_tile_overlap, spatial_scale, "spatial"),
                (
                    block_state.vae_temporal_tile_size,
                    block_state.vae_temporal_tile_overlap,
                    temporal_scale,
                    "temporal",
                ),
            ):
                if size < 2 * scale or overlap < 0 or overlap >= size or size % scale or overlap % scale:
                    raise ValueError(
                        f"VAE {name} tile size must be at least {2 * scale}, with 0 <= overlap < size; "
                        f"both must be multiples of {scale}. Got size={size}, overlap={overlap}."
                    )

            # The reference scales both spatial tile axes by the video's long side in latent space.
            long_side = max(block_state.height, block_state.width)
            min_latent_size = max(2, block_state.vae_tile_overlap // spatial_scale + 1)
            tile_height, tile_width = (
                max(min_latent_size, round(block_state.vae_tile_size // spatial_scale * length / long_side))
                * spatial_scale
                for length in (block_state.height, block_state.width)
            )
            tiling_settings.update(
                tile_sample_min_height=tile_height,
                tile_sample_min_width=tile_width,
                tile_sample_stride_height=tile_height - block_state.vae_tile_overlap,
                tile_sample_stride_width=tile_width - block_state.vae_tile_overlap,
                tile_sample_min_num_frames=block_state.vae_temporal_tile_size,
                tile_sample_stride_num_frames=block_state.vae_temporal_tile_size
                - block_state.vae_temporal_tile_overlap,
            )

        # ComponentsManager can share this VAE with another pipeline. Scope the decoding policy to this call,
        # including when decoding raises, so encoding and other pipelines retain their own settings.
        previous_settings = {name: getattr(vae, name) for name in tiling_settings}
        try:
            for name, value in tiling_settings.items():
                setattr(vae, name, value)
            return super().__call__(components, state)
        finally:
            for name, value in previous_settings.items():
                setattr(vae, name, value)
