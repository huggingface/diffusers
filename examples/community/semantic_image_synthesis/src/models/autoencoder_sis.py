import math

import torch

from diffusers import AutoencoderKL


class AutoencoderSIS(AutoencoderKL):
    def forward(self, x: torch.Tensor, encode: bool = False, decode: bool = False):
        if encode:
            return self.encode(x)
        elif decode:
            return self.decode(x)
        else:
            return AutoencoderKL.forward(self, x)

    def get_config(sample_size: int, compression: 4, in_channels: int, out_channels: int, latent_channels: int = 4):
        # Inspired from here :
        # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/config.json
        n_blocks_minus_1 = math.log(compression) / math.log(2)
        assert n_blocks_minus_1 == int(n_blocks_minus_1), f"compression should be 2**n {compression} given"
        n_blocks = int(n_blocks_minus_1) + 1
        down_block_types = ["DownEncoderBlock2D"] * n_blocks
        up_block_types = ["UpDecoderBlock2D"] * n_blocks
        block_out_channels = [min(128 * (2**i), 512) for i in range(n_blocks)]
        SIS_CONFIG_1024 = {
            "sample_size": 1024,
            "scaling_factor": 0.18215,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "latent_channels": latent_channels,
            "layers_per_block": 2,
            "force_upcast":False
        }
        # Inspired from here :
        # https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/vae/config.json
        SIS_CONFIG_768 = {
            "sample_size": 768,
            "scaling_factor": 0.18215,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "latent_channels": latent_channels,
            "layers_per_block": 2,
            "force_upcast":False
        }
        # Inspired from here :
        # https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
        SIS_CONFIG_512 = {
            "sample_size": 512,
            "scaling_factor": 0.18215,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "latent_channels": latent_channels,
            "layers_per_block": 2,
            "force_upcast":False
        }
        SIS_CONFIG_384 = {
            "sample_size": 384,
            "scaling_factor": 0.18215,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "latent_channels": latent_channels,
            "layers_per_block": 2,
            "force_upcast":False
        }
        SIS_CONFIG_256 = {
            "sample_size": 256,
            "scaling_factor": 0.18215,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "latent_channels": latent_channels,
            "layers_per_block": 2,
            "force_upcast":False
        }
        SIS_CONFIG_192 = {
            "sample_size": 192,
            "scaling_factor": 0.18215,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "latent_channels": latent_channels,
            "layers_per_block": 2,
            "force_upcast":False
        }
        SIS_CONFIG_128 = {
            "sample_size": 128,
            "scaling_factor": 0.18215,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "latent_channels": latent_channels,
            "layers_per_block": 1,
            "force_upcast":False
        }
        CONFIG_DICT = {
            128: SIS_CONFIG_128,
            192: SIS_CONFIG_192,
            256: SIS_CONFIG_256,
            384: SIS_CONFIG_384,
            512: SIS_CONFIG_512,
            768: SIS_CONFIG_768,
            1024: SIS_CONFIG_1024,
        }
        assert sample_size in CONFIG_DICT, f"Sample size should be in {list(CONFIG_DICT.keys())}"
        return CONFIG_DICT[sample_size]
