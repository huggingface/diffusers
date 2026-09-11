#!/usr/bin/env python3
import argparse
import math
import os
from copy import deepcopy

import requests
import torch
from audio_diffusion.models import DiffusionAttnUnet1D
from diffusion import sampling
from torch import nn

from diffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT


MODELS_MAP = {
    "gwf-440k": {
        "url": "https://model-server.zqevans2.workers.dev/gwf-440k.ckpt",
        "sample_rate": 48000,
        "sample_size": 65536,
    },
    "jmann-small-190k": {
        "url": "https://model-server.zqevans2.workers.dev/jmann-small-190k.ckpt",
        "sample_rate": 48000,
        "sample_size": 65536,
    },
    "jmann-large-580k": {
        "url": "https://model-server.zqevans2.workers.dev/jmann-large-580k.ckpt",
        "sample_rate": 48000,
        "sample_size": 131072,
    },
    "maestro-uncond-150k": {
        "url": "https://model-server.zqevans2.workers.dev/maestro-uncond-150k.ckpt",
        "sample_rate": 16000,
        "sample_size": 65536,
    },
    "unlocked-uncond-250k": {
        "url": "https://model-server.zqevans2.workers.dev/unlocked-uncond-250k.ckpt",
        "sample_rate": 16000,
        "sample_size": 65536,
    },
    "honk-140k": {
        "url": "https://model-server.zqevans2.workers.dev/honk-140k.ckpt",
        "sample_rate": 16000,
        "sample_size": 65536,
    },
}


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma**2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)


class Object(object):
    pass


class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers=4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


def download(model_name):
    url = MODELS_MAP[model_name]["url"]
    r = requests.get(url, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)

    local_filename = f"./{model_name}.ckpt"
    with open(local_filename, "wb") as fp:
        for chunk in r.iter_content(chunk_size=8192):
            fp.write(chunk)

    return local_filename


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_path.split("/")[-1].split(".")[0]
    if not os.path.isfile(args.model_path):
        assert model_name == args.model_path, (
            f"Make sure to provide one of the official model names {MODELS_MAP.keys()}"
        )
        args.model_path = download(model_name)

    sample_rate = MODELS_MAP[model_name]["sample_rate"]
    sample_size = MODELS_MAP[model_name]["sample_size"]

    config = Object()
    config.sample_size = sample_size
    config.sample_rate = sample_rate
    config.latent_dim = 0

    diffusers_model = UNet1DModel(sample_size=sample_size, sample_rate=sample_rate)

    orig_model = DiffusionUncond(config)
    orig_model.load_state_dict(torch.load(args.model_path, map_location=device)["state_dict"])
    orig_model = orig_model.diffusion_ema.eval()
    orig_model_state_dict = orig_model.state_dict()
    converted = convert_component_checkpoint(orig_model_state_dict, dict(diffusers_model.config), "UNet1DModel")
    diffusers_model.load_state_dict(converted, strict=True)

    steps = 100
    seed = 33

    diffusers_scheduler = IPNDMScheduler(num_train_timesteps=steps)

    generator = torch.manual_seed(seed)
    noise = torch.randn([1, 2, config.sample_size], generator=generator).to(device)

    t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
    step_list = get_crash_schedule(t)

    pipe = DanceDiffusionPipeline(unet=diffusers_model, scheduler=diffusers_scheduler)

    generator = torch.manual_seed(33)
    audio = pipe(num_inference_steps=steps, generator=generator).audios

    generated = sampling.iplms_sample(orig_model, noise, step_list, {})
    generated = generated.clamp(-1, 1)

    diff_sum = (generated - audio).abs().sum()
    diff_max = (generated - audio).abs().max()

    if args.save:
        pipe.save_pretrained(args.checkpoint_path)

    print("Diff sum", diff_sum)
    print("Diff max", diff_max)

    assert diff_max < 1e-3, f"Diff max: {diff_max} is too much :-/"

    print(f"Conversion for {model_name} successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted model or not."
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)
