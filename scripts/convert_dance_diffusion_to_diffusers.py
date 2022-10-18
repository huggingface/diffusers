#!/usr/bin/env python3
import argparse
import math
import os
from copy import deepcopy

import torch
from torch import nn

from audio_diffusion.models import DiffusionAttnUnet1D
from diffusers import UNet1DModel


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
    pass


DOWN_NUM_TO_LAYER = {
    "1": "resnets.0",
    "2": "attentions.0",
    "3": "resnets.1",
    "4": "attentions.1",
    "5": "resnets.2",
    "6": "attentions.2",
}
UP_NUM_TO_LAYER = {
    "8": "resnets.0",
    "9": "attentions.0",
    "10": "resnets.1",
    "11": "attentions.1",
    "12": "resnets.2",
    "13": "attentions.2",
}
MID_NUM_TO_LAYER = {
    "1": "resnets.0",
    "2": "attentions.0",
    "3": "resnets.1",
    "4": "attentions.1",
    "5": "resnets.2",
    "6": "attentions.2",
    "8": "resnets.3",
    "9": "attentions.3",
    "10": "resnets.4",
    "11": "attentions.4",
    "12": "resnets.5",
    "13": "attentions.5",
}
DEPTH_0_TO_LAYER = {
    "0": "resnets.0",
    "1": "resnets.1",
    "2": "resnets.2",
    "4": "resnets.0",
    "5": "resnets.1",
    "6": "resnets.2",
}


def rename(input_string, max_depth=13):
    string = input_string

    if string.split(".")[0] == "timestep_embed":
        return string.replace("timestep_embed", "time_proj")

    depth = 0
    if string.startswith("net.3."):
        depth += 1
        string = string[6:]
    elif string.startswith("net."):
        string = string[4:]

    while string.startswith("main.7."):
        depth += 1
        string = string[7:]

    if string.startswith("main."):
        string = string[5:]

    # mid block
    if string[:2].isdigit():
        layer_num = string[:2]
        string_left = string[2:]
    else:
        layer_num = string[0]
        string_left = string[1:]

    if depth == max_depth:
        new_layer = MID_NUM_TO_LAYER[layer_num]
        prefix = "mid_block"
    elif depth > 0 and int(layer_num) < 7:
        new_layer = DOWN_NUM_TO_LAYER[layer_num]
        prefix = f"down_blocks.{depth}"
    elif depth > 0 and int(layer_num) > 7:
        new_layer = UP_NUM_TO_LAYER[layer_num]
        prefix = f"up_blocks.{max_depth - depth - 1}"
    elif depth == 0:
        new_layer = DEPTH_0_TO_LAYER[layer_num]
        prefix = f"up_blocks.{max_depth - 1}" if int(layer_num) > 3 else "down_blocks.0"

    new_string = prefix + "." + new_layer + string_left
    return new_string


def rename_orig_weights(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith("kernel"):
            # up- and downsample layers, don't have trainable weights
            continue

        new_k = rename(k)
        new_state_dict[new_k] = v

    return new_state_dict


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_path.split("/")[-1].split(".")[0]
    if not os.path.isfile(args.model_path):
        assert (
            model_name == args.model_path
        ), f"Make sure to provide one of the official model names {MODELS_MAP.keys()}"
        args.model_path = download(model_name)

    sample_rate = MODELS_MAP[model_name]["sample_rate"]
    sample_size = MODELS_MAP[model_name]["sample_size"]

    config = Object()
    config.sample_size = sample_size
    config.sample_rate = sample_rate
    config.latent_dim = 0

    diffusers_model = UNet1DModel()
    diffusers_state_dict = diffusers_model.state_dict()

    orig_model = DiffusionUncond(config)
    orig_model.load_state_dict(torch.load(args.model_path, map_location=device)["state_dict"])
    orig_model = orig_model.diffusion_ema.eval()
    orig_model_state_dict = orig_model.state_dict()
    renamed_state_dict = rename_orig_weights(orig_model_state_dict)

    renamed_minus_diffusers = set(renamed_state_dict.keys()) - set(diffusers_state_dict.keys())
    diffusers_minus_renamed = set(diffusers_state_dict.keys()) - set(renamed_state_dict.keys())

    assert len(renamed_minus_diffusers) == 0, f"Problem with {renamed_minus_diffusers}"
    assert all(k.endswith("kernel") for k in list(diffusers_minus_renamed)), f"Problem with {diffusers_minus_renamed}"

    for key, value in renamed_state_dict.items():
        assert (
            diffusers_state_dict[key].squeeze().shape == value.squeeze().shape
        ), f"Shape for {key} doesn't match. Diffusers: {diffusers_state_dict[key].shape} vs. {value.shape}"
        diffusers_state_dict[key] = value

    steps = 100
    step_index = 2

    generator = torch.manual_seed(33)
    noise = torch.randn([1, 2, config.sample_size], generator=generator).to(device)
    t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
    step_list = get_crash_schedule(t)

    output = orig_model(noise, step_list[step_index : step_index + 1])
    assert output.abs().sum() - 4550.5430 < 1e-3

    diffusers_output = diffusers_model(noise, step_list[step_index : step_index + 1])
    import ipdb; ipdb.set_trace()
    assert diffusers_output.abs().sum() - 4550.5430 < 1e-3
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)
