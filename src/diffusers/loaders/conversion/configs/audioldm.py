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

"""Original configuration helpers and model presets for the audioldm assembly recipe."""


def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a UNet config for diffusers based on the config of the original AudioLDM model.
    """
    unet_params = original_config["model"]["params"]["unet_config"]["params"]
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]

    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    cross_attention_dim = (
        unet_params["cross_attention_dim"] if "cross_attention_dim" in unet_params else block_out_channels
    )

    class_embed_type = "simple_projection" if "extra_film_condition_dim" in unet_params else None
    projection_class_embeddings_input_dim = (
        unet_params["extra_film_condition_dim"] if "extra_film_condition_dim" in unet_params else None
    )
    class_embeddings_concat = unet_params["extra_film_use_concat"] if "extra_film_use_concat" in unet_params else None

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params["in_channels"],
        "out_channels": unet_params["out_channels"],
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params["num_res_blocks"],
        "cross_attention_dim": cross_attention_dim,
        "class_embed_type": class_embed_type,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "class_embeddings_concat": class_embeddings_concat,
    }

    return config


def create_vae_diffusers_config(original_config, checkpoint, image_size: int):
    """
    Creates a VAE config for diffusers based on the config of the original AudioLDM model. Compared to the original
    Stable Diffusion conversion, this function passes a *learnt* VAE scaling factor to the diffusers VAE.
    """
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    _ = original_config["model"]["params"]["first_stage_config"]["params"]["embed_dim"]

    block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    scaling_factor = checkpoint["scale_factor"] if "scale_by_std" in original_config["model"]["params"] else 0.18215

    config = {
        "sample_size": image_size,
        "in_channels": vae_params["in_channels"],
        "out_channels": vae_params["out_ch"],
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params["z_channels"],
        "layers_per_block": vae_params["num_res_blocks"],
        "scaling_factor": float(scaling_factor),
    }
    return config


def create_transformers_vocoder_config(original_config):
    """
    Creates a config for transformers SpeechT5HifiGan based on the config of the vocoder model.
    """
    vocoder_params = original_config["model"]["params"]["vocoder_config"]["params"]

    config = {
        "model_in_dim": vocoder_params["num_mels"],
        "sampling_rate": vocoder_params["sampling_rate"],
        "upsample_initial_channel": vocoder_params["upsample_initial_channel"],
        "upsample_rates": list(vocoder_params["upsample_rates"]),
        "upsample_kernel_sizes": list(vocoder_params["upsample_kernel_sizes"]),
        "resblock_kernel_sizes": list(vocoder_params["resblock_kernel_sizes"]),
        "resblock_dilation_sizes": [
            list(resblock_dilation) for resblock_dilation in vocoder_params["resblock_dilation_sizes"]
        ],
        "normalize_before": False,
    }

    return config


DEFAULT_CONFIG = {
    "model": {
        "params": {
            "linear_start": 0.0015,
            "linear_end": 0.0195,
            "timesteps": 1000,
            "channels": 8,
            "scale_by_std": True,
            "unet_config": {
                "target": "audioldm.latent_diffusion.openaimodel.UNetModel",
                "params": {
                    "extra_film_condition_dim": 512,
                    "extra_film_use_concat": True,
                    "in_channels": 8,
                    "out_channels": 8,
                    "model_channels": 128,
                    "attention_resolutions": [8, 4, 2],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 3, 5],
                    "num_head_channels": 32,
                },
            },
            "first_stage_config": {
                "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 8,
                    "ddconfig": {
                        "z_channels": 8,
                        "resolution": 256,
                        "in_channels": 1,
                        "out_ch": 1,
                        "ch": 128,
                        "ch_mult": [1, 2, 4],
                        "num_res_blocks": 2,
                    },
                },
            },
            "vocoder_config": {
                "target": "audioldm.first_stage_model.vocoder",
                "params": {
                    "upsample_rates": [5, 4, 2, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 8, 4, 4],
                    "upsample_initial_channel": 1024,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "num_mels": 64,
                    "sampling_rate": 16000,
                },
            },
        },
    },
}

__all__ = [
    "DEFAULT_CONFIG",
    "create_transformers_vocoder_config",
    "create_unet_diffusers_config",
    "create_vae_diffusers_config",
]
