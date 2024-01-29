import argparse

import torch
import yaml

from diffusers import DDIMScheduler, LDMPipeline, UNetLDMModel, VQModel


def convert_ldm_original(checkpoint_path, config_path, output_path):
    config = yaml.safe_load(config_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    keys = list(state_dict.keys())

    # extract state_dict for VQVAE
    first_stage_dict = {}
    first_stage_key = "first_stage_model."
    for key in keys:
        if key.startswith(first_stage_key):
            first_stage_dict[key.replace(first_stage_key, "")] = state_dict[key]

    # extract state_dict for UNetLDM
    unet_state_dict = {}
    unet_key = "model.diffusion_model."
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = state_dict[key]

    vqvae_init_args = config["model"]["params"]["first_stage_config"]["params"]
    unet_init_args = config["model"]["params"]["unet_config"]["params"]

    vqvae = VQModel(**vqvae_init_args).eval()
    vqvae.load_state_dict(first_stage_dict)

    unet = UNetLDMModel(**unet_init_args).eval()
    unet.load_state_dict(unet_state_dict)

    noise_scheduler = DDIMScheduler(
        timesteps=config["model"]["params"]["timesteps"],
        beta_schedule="scaled_linear",
        beta_start=config["model"]["params"]["linear_start"],
        beta_end=config["model"]["params"]["linear_end"],
        clip_sample=False,
    )

    pipeline = LDMPipeline(vqvae, unet, noise_scheduler)
    pipeline.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    convert_ldm_original(args.checkpoint_path, args.config_path, args.output_path)
