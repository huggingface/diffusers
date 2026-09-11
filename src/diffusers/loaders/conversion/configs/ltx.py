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

"""Original configuration helpers and model presets for the ltx assembly recipe."""

from typing import Any


def get_transformer_config(version: str) -> dict[str, Any]:
    if version == "0.9.7":
        config = {
            "in_channels": 128,
            "out_channels": 128,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "cross_attention_dim": 4096,
            "num_layers": 48,
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 4096,
            "attention_bias": True,
            "attention_out_bias": True,
        }
    else:
        config = {
            "in_channels": 128,
            "out_channels": 128,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "cross_attention_dim": 2048,
            "num_layers": 28,
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 4096,
            "attention_bias": True,
            "attention_out_bias": True,
        }
    return config


def get_vae_config(version: str) -> dict[str, Any]:
    if version in ["0.9.0"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 512),
            "down_block_types": (
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
            ),
            "decoder_block_out_channels": (128, 256, 512, 512),
            "layers_per_block": (4, 3, 3, 3, 4),
            "decoder_layers_per_block": (4, 3, 3, 3, 4),
            "spatio_temporal_scaling": (True, True, True, False),
            "decoder_spatio_temporal_scaling": (True, True, True, False),
            "decoder_inject_noise": (False, False, False, False, False),
            "downsample_type": ("conv", "conv", "conv", "conv"),
            "upsample_residual": (False, False, False, False),
            "upsample_factor": (1, 1, 1, 1),
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
            "timestep_conditioning": False,
        }
    elif version in ["0.9.1"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 512),
            "down_block_types": (
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
            ),
            "decoder_block_out_channels": (256, 512, 1024),
            "layers_per_block": (4, 3, 3, 3, 4),
            "decoder_layers_per_block": (5, 6, 7, 8),
            "spatio_temporal_scaling": (True, True, True, False),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (True, True, True, False),
            "downsample_type": ("conv", "conv", "conv", "conv"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": True,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
        }
    elif version in ["0.9.5"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 1024, 2048),
            "down_block_types": (
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
            ),
            "decoder_block_out_channels": (256, 512, 1024),
            "layers_per_block": (4, 6, 6, 2, 2),
            "decoder_layers_per_block": (5, 5, 5, 5),
            "spatio_temporal_scaling": (True, True, True, True),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (False, False, False, False),
            "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": True,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        }
    elif version in ["0.9.7"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 1024, 2048),
            "down_block_types": (
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
            ),
            "decoder_block_out_channels": (256, 512, 1024),
            "layers_per_block": (4, 6, 6, 2, 2),
            "decoder_layers_per_block": (5, 5, 5, 5),
            "spatio_temporal_scaling": (True, True, True, True),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (False, False, False, False),
            "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": True,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        }
    return config


def get_spatial_latent_upsampler_config(version: str) -> dict[str, Any]:
    if version == "0.9.7":
        config = {
            "in_channels": 128,
            "mid_channels": 512,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
        }
    elif version == "0.9.8":
        config = {
            "in_channels": 128,
            "mid_channels": 512,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
        }
    else:
        raise ValueError(f"Unsupported version: {version}")
    return config


__all__ = ["get_spatial_latent_upsampler_config", "get_transformer_config", "get_vae_config"]
