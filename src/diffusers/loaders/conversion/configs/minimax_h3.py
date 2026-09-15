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

"""Original configuration helpers and model presets for the minimax_h3 assembly recipe."""

import json
import os
from typing import Any


MINIMAX_H3_TRANSFORMER_CONFIG = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,  # token_refiner_num_layers
    "ffn_dim": 14336,  # ffn_hidden_size
    "in_channels": 24,  # latents_dim
    "audio_in_channels": 32,  # audio_latents_dim
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,  # timestep_input_dim
    "time_embed_hidden_dim": 5376,  # time_embed_hidden_size
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,  # rope_inv_freq_len
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
}

MINIMAX_H3_TEST_TRANSFORMER_CONFIG = {
    **MINIMAX_H3_TRANSFORMER_CONFIG,
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 128,
    "text_dim": 48,
    "freq_dim": 16,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 32,
    "rope_freq_dim": 4,
}

MINIMAX_H3_VIDEO_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,  # out_ch
    "latent_channels": 24,  # z_channels == embed_dim
    "block_out_channels": [128, 256, 256, 512, 512, 1024],  # ch * ch_mult
    "layers_per_block": 2,  # num_res_blocks
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],  # space_down
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],  # time_down
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "spatial_padding_mode": "reflect",  # padding_mode
    "decoder_num_layers": 36,  # vit_decoder_kwargs.num_layers
    "decoder_num_attention_heads": 32,  # vit_decoder_kwargs.heads
    "decoder_attention_head_dim": 64,  # vit_decoder_kwargs.dim_head
    "decoder_num_register_tokens": 4,  # ViT3DDecoder default
    "decoder_ffn_mult": 4,  # FeedForward default
    "decoder_rope_theta": 100.0,  # vit_decoder_kwargs.rope_theta
    "decoder_rope_dim_ratio": 0.75,  # vit_decoder_kwargs.rope_dim_ratio
    "decoder_norm_eps": 1e-05,  # ViT3DDecoder eps
    "clip_length": 17,  # video_vae/config.json vae_clip_length
    "token_drop": 3,  # video_vae/config.json vae_token_drop
}

MINIMAX_H3_TEST_VIDEO_VAE_CONFIG = {
    **MINIMAX_H3_VIDEO_VAE_CONFIG,
    "block_out_channels": [32, 64],
    "layers_per_block": 1,
    "spatial_downsample_factors": [2, 2],
    "temporal_downsample_factors": [2, 2],
    "decoder_num_layers": 4,
    "decoder_num_attention_heads": 4,
    "decoder_attention_head_dim": 32,
}

MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG = {
    "num_attention_heads": 8,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}


def get_audio_vae_config(checkpoint_path: str) -> dict[str, Any]:
    """Build the `AutoencoderKLMiniMaxH3Audio` config from the original audio-VAE metadata.

    `audio_vae/metadata.json` carries the constructor kwargs the checkpoint was built with, and `audio_vae/config.json`
    carries the per-channel `latents_mean` / `latents_std` MiniMax-H3 normalizes with. The two are cross-checked here
    because they duplicate the latent width and sample rate.
    """
    source_dir = os.path.join(checkpoint_path, "audio_vae")
    with open(os.path.join(source_dir, "metadata.json")) as f:
        kwargs = json.load(f)["metadata"]["kwargs"]
    with open(os.path.join(source_dir, "config.json")) as f:
        wrapper_config = json.load(f)

    if kwargs["decoder_type"] != "bigvgan":
        raise ValueError(f"Only the BigVGAN decoder is supported, got {kwargs['decoder_type']!r}.")
    if not kwargs["attn_proj"]:
        raise ValueError("The audio VAE is expected to carry the causal-attention latent projection.")
    latent_channels = kwargs["vae_latent_channels"]
    if wrapper_config["latent_channels"] != latent_channels:
        raise ValueError(
            f"latent width disagreement: metadata.json says {latent_channels}, "
            f"config.json says {wrapper_config['latent_channels']}."
        )
    if wrapper_config["sample_rate"] != kwargs["sample_rate"]:
        raise ValueError(
            f"sample rate disagreement: metadata.json says {kwargs['sample_rate']}, "
            f"config.json says {wrapper_config['sample_rate']}."
        )
    for key in ("latents_mean", "latents_std"):
        if len(wrapper_config[key]) != latent_channels:
            raise KeyError(f"{source_dir}/config.json `{key}` does not have {latent_channels} entries.")

    return {
        "encoder_dim": kwargs["encoder_dim"],
        "encoder_rates": kwargs["encoder_rates"],
        "latent_dim": kwargs["latent_dim"],
        "latent_channels": latent_channels,
        "decoder_dim": kwargs["decoder_dim"],
        "decoder_rates": kwargs["decoder_rates"],
        # The reference's two hardcoded BigVGAN tables (16 kHz and 32 kHz) both pair rate `u` with kernel
        # `2u` for even `u` and `2u - 1` for odd `u`, i.e. [5, 5, 2, ...] -> [9, 9, 4, ...].
        "decoder_kernel_sizes": [2 * rate - (rate % 2) for rate in kwargs["decoder_rates"]],
        **MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG,
        # Renamed from the original `sample_rate` to the diffusers audio convention.
        "sampling_rate": kwargs["sample_rate"],
        "latents_mean": wrapper_config["latents_mean"],
        "latents_std": wrapper_config["latents_std"],
    }


__all__ = [
    "MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG",
    "MINIMAX_H3_TEST_TRANSFORMER_CONFIG",
    "MINIMAX_H3_TEST_VIDEO_VAE_CONFIG",
    "MINIMAX_H3_TRANSFORMER_CONFIG",
    "MINIMAX_H3_VIDEO_VAE_CONFIG",
    "get_audio_vae_config",
]
