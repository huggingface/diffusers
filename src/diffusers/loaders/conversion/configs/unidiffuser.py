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

"""Original configuration helpers and model presets for the unidiffuser assembly recipe."""

from argparse import Namespace


SCHEDULER_CONFIG = Namespace(
    **{
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "solver_order": 3,
    }
)


def create_vae_diffusers_config(config_type):
    # Hardcoded for now
    if config_type == "test":
        vae_config = create_vae_diffusers_config_test()
    elif config_type == "big":
        vae_config = create_vae_diffusers_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )
    return vae_config


def create_unidiffuser_unet_config(config_type, version):
    # Hardcoded for now
    if config_type == "test":
        unet_config = create_unidiffuser_unet_config_test()
    elif config_type == "big":
        unet_config = create_unidiffuser_unet_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )
    # Unidiffuser-v1 uses data type embeddings
    if version == 1:
        unet_config["use_data_type_embedding"] = True
    return unet_config


def create_text_decoder_config(config_type):
    # Hardcoded for now
    if config_type == "test":
        text_decoder_config = create_text_decoder_config_test()
    elif config_type == "big":
        text_decoder_config = create_text_decoder_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )
    return text_decoder_config


def create_vae_diffusers_config_test():
    vae_config = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [32, 64],
        "latent_channels": 4,
        "layers_per_block": 1,
    }
    return vae_config


def create_unidiffuser_unet_config_test():
    unet_config = {
        "text_dim": 32,
        "clip_img_dim": 32,
        "num_text_tokens": 77,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 2,
        "dropout": 0.0,
        "norm_num_groups": 32,
        "attention_bias": False,
        "sample_size": 16,
        "patch_size": 2,
        "activation_fn": "gelu",
        "num_embeds_ada_norm": 1000,
        "norm_type": "layer_norm",
        "block_type": "unidiffuser",
        "pre_layer_norm": False,
        "use_timestep_embedding": False,
        "norm_elementwise_affine": True,
        "use_patch_pos_embed": False,
        "ff_final_dropout": True,
        "use_data_type_embedding": False,
    }
    return unet_config


def create_text_decoder_config_test():
    text_decoder_config = {
        "prefix_length": 77,
        "prefix_inner_dim": 32,
        "prefix_hidden_dim": 32,
        "vocab_size": 1025,  # 1024 + 1 for new EOS token
        "n_positions": 1024,
        "n_embd": 32,
        "n_layer": 5,
        "n_head": 4,
        "n_inner": 37,
        "activation_function": "gelu",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
    }
    return text_decoder_config


def create_vae_diffusers_config_big():
    vae_config = {
        "sample_size": 256,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [128, 256, 512, 512],
        "latent_channels": 4,
        "layers_per_block": 2,
    }
    return vae_config


def create_unidiffuser_unet_config_big():
    unet_config = {
        "text_dim": 64,
        "clip_img_dim": 512,
        "num_text_tokens": 77,
        "num_attention_heads": 24,
        "attention_head_dim": 64,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 30,
        "dropout": 0.0,
        "norm_num_groups": 32,
        "attention_bias": False,
        "sample_size": 64,
        "patch_size": 2,
        "activation_fn": "gelu",
        "num_embeds_ada_norm": 1000,
        "norm_type": "layer_norm",
        "block_type": "unidiffuser",
        "pre_layer_norm": False,
        "use_timestep_embedding": False,
        "norm_elementwise_affine": True,
        "use_patch_pos_embed": False,
        "ff_final_dropout": True,
        "use_data_type_embedding": False,
    }
    return unet_config


def create_text_decoder_config_big():
    text_decoder_config = {
        "prefix_length": 77,
        "prefix_inner_dim": 768,
        "prefix_hidden_dim": 64,
        "vocab_size": 50258,  # 50257 + 1 for new EOS token
        "n_positions": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_inner": 3072,
        "activation_function": "gelu",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
    }
    return text_decoder_config


__all__ = [
    "SCHEDULER_CONFIG",
    "create_text_decoder_config",
    "create_text_decoder_config_big",
    "create_text_decoder_config_test",
    "create_unidiffuser_unet_config",
    "create_unidiffuser_unet_config_big",
    "create_unidiffuser_unet_config_test",
    "create_vae_diffusers_config",
    "create_vae_diffusers_config_big",
    "create_vae_diffusers_config_test",
]
