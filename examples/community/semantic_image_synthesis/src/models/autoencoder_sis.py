from diffusers import AutoencoderKL
import torch

class AutoencoderSIS(AutoencoderKL):
    
    def forward(self,x:torch.Tensor,encode:bool=False,decode:bool=False):
        if encode:
            return self.encode(x)
        elif decode:
            return self.decode(x)
        else:
            return AutoencoderKL.forward(self,x)

    def get_config(
            sample_size:int,
            latent_size:int,
            in_channels:int,
            out_channels:int,
            latent_channels:int=4):
        # Inspired from here : 
        # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/config.json
        SIS_CONFIG_1024 = {
            "sample_size": 1024,
            "scaling_factor": latent_size/1024,
            "in_channels":in_channels,
            "out_channels":out_channels,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"],
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "block_out_channels": [
                128,
                256,
                512,
                512
            ],
            "latent_channels":latent_channels,          
            "layers_per_block": 2,
        }
        # Inspired from here : 
        # https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/vae/config.json
        SIS_CONFIG_768 = {
            "sample_size": 768,
            "scaling_factor": latent_size/768,
            "in_channels":in_channels,
            "out_channels":out_channels,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"],
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "block_out_channels": [
                128,
                256,
                512,
                512
            ],
            "latent_channels":latent_channels,          
            "layers_per_block": 2,
        }
        # Inspired from here : 
        # https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
        SIS_CONFIG_512 = {
            "sample_size": 512,
            "scaling_factor": latent_size/512,
            "in_channels":in_channels,
            "out_channels":out_channels,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"],
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "block_out_channels": [
                128,
                256,
                512,
                512
            ],
            "latent_channels":latent_channels,          
            "layers_per_block": 2,
        }
        SIS_CONFIG_256 = {
            "sample_size": 256,
            "scaling_factor": latent_size/256,
            "in_channels":in_channels,
            "out_channels":out_channels,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"],
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "block_out_channels": [
                128,
                256,
                512,
                512
            ],
            "latent_channels":latent_channels,          
            "layers_per_block": 2,
        }
        SIS_CONFIG_128 = {
            "sample_size": 128,
            "scaling_factor": latent_size/128,
            "in_channels":in_channels,
            "out_channels":out_channels,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"],
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "block_out_channels": [
                128,
                256,
                512
            ],
            "latent_channels":latent_channels,          
            "layers_per_block": 1,
        }
        CONFIG_DICT = {
            128:SIS_CONFIG_128,
            256:SIS_CONFIG_256,
            512:SIS_CONFIG_512,
            768:SIS_CONFIG_768,
            1024:SIS_CONFIG_1024
        }
        assert sample_size in CONFIG_DICT,f"Sample size should be in {list(CONFIG_DICT.keys())}"
        return CONFIG_DICT[sample_size]