#!/usr/bin/env python3
from torch import nn
from audio_diffusion.models import DiffusionAttnUnet1D
import argparse
from copy import deepcopy
import torch
import os
import math


MODELS_MAP = {
    "gwf-440k": {
                         'url': "https://model-server.zqevans2.workers.dev/gwf-440k.ckpt",
                         'sample_rate': 48000,
                         'sample_size': 65536
                         },
    "jmann-small-190k": {
                         'url': "https://model-server.zqevans2.workers.dev/jmann-small-190k.ckpt",
                         'sample_rate': 48000,
                         'sample_size': 65536
                         },
    "jmann-large-580k": {
                         'url': "https://model-server.zqevans2.workers.dev/jmann-large-580k.ckpt",
                         'sample_rate': 48000,
                         'sample_size': 131072
                         },
    "maestro-uncond-150k": {
                         'url': "https://model-server.zqevans2.workers.dev/maestro-uncond-150k.ckpt",
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
    "unlocked-uncond-250k": {
                         'url': "https://model-server.zqevans2.workers.dev/unlocked-uncond-250k.ckpt",
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
    "honk-140k": {'url': "https://model-server.zqevans2.workers.dev/honk-140k.ckpt", 'sample_rate': 16000, 'sample_size': 65536}
}


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
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
    pass


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model_path.split("/")[-1].split(".")[0]
    if not os.path.isfile(args.model_path):
        assert model_name == args.model_path, f"Make sure to provide one of the official model names {MODELS_MAP.keys()}"
        args.model_path = download(model_name)

    sample_rate = MODELS_MAP[model_name]["sample_rate"]
    sample_size = MODELS_MAP[model_name]["sample_size"]

    config = Object()
    config.sample_size = sample_size
    config.sample_rate = sample_rate
    config.latent_dim = 0

    diffusion_model = DiffusionUncond(config)
    diffusion_model.load_state_dict(torch.load(args.model_path, map_location=device)["state_dict"])
    model = diffusion_model.eval()

    steps = 100
    step_index = 2

    generator = torch.manual_seed(33)
    noise = torch.randn([1, 2, config.sample_size], generator=generator).to(device)
    t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
    step_list = get_crash_schedule(t)

    output = model.diffusion_ema(noise, step_list[step_index: step_index + 1])
    assert output.abs().sum() - 4550.5430 < 1e-3

    import ipdb; ipdb.set_trace()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)
