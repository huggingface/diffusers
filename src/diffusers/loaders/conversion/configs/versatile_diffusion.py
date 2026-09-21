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

"""Original configuration helpers and model presets for the versatile_diffusion assembly recipe."""

from argparse import Namespace


SCHEDULER_CONFIG = Namespace(
    **{
        "beta_linear_start": 0.00085,
        "beta_linear_end": 0.012,
        "timesteps": 1000,
        "scale_factor": 0.18215,
    }
)

IMAGE_UNET_CONFIG = Namespace(
    **{
        "input_channels": 4,
        "model_channels": 320,
        "output_channels": 4,
        "num_noattn_blocks": [2, 2, 2, 2],
        "channel_mult": [1, 2, 4, 4],
        "with_attn": [True, True, True, False],
        "num_heads": 8,
        "context_dim": 768,
        "use_checkpoint": True,
    }
)

TEXT_UNET_CONFIG = Namespace(
    **{
        "input_channels": 768,
        "model_channels": 320,
        "output_channels": 768,
        "num_noattn_blocks": [2, 2, 2, 2],
        "channel_mult": [1, 2, 4, 4],
        "second_dim": [4, 4, 4, 4],
        "with_attn": [True, True, True, False],
        "num_heads": 8,
        "context_dim": 768,
        "use_checkpoint": True,
    }
)

AUTOENCODER_CONFIG = Namespace(
    **{
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
)


def create_image_unet_diffusers_config(unet_params):
    """
    Creates a config for the diffusers based on the config of the VD model.
    """

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if unet_params.with_attn[i] else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if unet_params.with_attn[-i - 1] else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    if not all(n == unet_params.num_noattn_blocks[0] for n in unet_params.num_noattn_blocks):
        raise ValueError("Not all num_res_blocks are equal, which is not supported in this script.")

    config = {
        "sample_size": None,
        "in_channels": unet_params.input_channels,
        "out_channels": unet_params.output_channels,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params.num_noattn_blocks[0],
        "cross_attention_dim": unet_params.context_dim,
        "attention_head_dim": unet_params.num_heads,
    }

    return config


def create_text_unet_diffusers_config(unet_params):
    """
    Creates a config for the diffusers based on the config of the VD model.
    """

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlockFlat" if unet_params.with_attn[i] else "DownBlockFlat"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlockFlat" if unet_params.with_attn[-i - 1] else "UpBlockFlat"
        up_block_types.append(block_type)
        resolution //= 2

    if not all(n == unet_params.num_noattn_blocks[0] for n in unet_params.num_noattn_blocks):
        raise ValueError("Not all num_res_blocks are equal, which is not supported in this script.")

    config = {
        "sample_size": None,
        "in_channels": (unet_params.input_channels, 1, 1),
        "out_channels": (unet_params.output_channels, 1, 1),
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params.num_noattn_blocks[0],
        "cross_attention_dim": unet_params.context_dim,
        "attention_head_dim": unet_params.num_heads,
    }

    return config


def create_vae_diffusers_config(vae_params):
    """
    Creates a config for the diffusers based on the config of the VD model.
    """

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": vae_params.resolution,
        "in_channels": vae_params.in_channels,
        "out_channels": vae_params.out_ch,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params.z_channels,
        "layers_per_block": vae_params.num_res_blocks,
    }
    return config


__all__ = [
    "AUTOENCODER_CONFIG",
    "IMAGE_UNET_CONFIG",
    "SCHEDULER_CONFIG",
    "TEXT_UNET_CONFIG",
    "create_image_unet_diffusers_config",
    "create_text_unet_diffusers_config",
    "create_vae_diffusers_config",
]
