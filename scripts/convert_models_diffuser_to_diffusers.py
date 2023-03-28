import json
import os

import torch

from diffusers import UNet1DModel


os.makedirs("hub/hopper-medium-v2/unet/hor32", exist_ok=True)
os.makedirs("hub/hopper-medium-v2/unet/hor128", exist_ok=True)

os.makedirs("hub/hopper-medium-v2/value_function", exist_ok=True)


def unet(hor):
    if hor == 128:
        down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D")
        block_out_channels = (32, 128, 256)
        up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D")

    elif hor == 32:
        down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D")
        block_out_channels = (32, 64, 128, 256)
        up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D")
    model = torch.load(f"/Users/bglickenhaus/Documents/diffuser/temporal_unet-hopper-mediumv2-hor{hor}.torch")
    state_dict = model.state_dict()
    config = {
        "down_block_types": down_block_types,
        "block_out_channels": block_out_channels,
        "up_block_types": up_block_types,
        "layers_per_block": 1,
        "use_timestep_embedding": True,
        "out_block_type": "OutConv1DBlock",
        "norm_num_groups": 8,
        "downsample_each_block": False,
        "in_channels": 14,
        "out_channels": 14,
        "extra_in_channels": 0,
        "time_embedding_type": "positional",
        "flip_sin_to_cos": False,
        "freq_shift": 1,
        "sample_size": 65536,
        "mid_block_type": "MidResTemporalBlock1D",
        "act_fn": "mish",
    }
    hf_value_function = UNet1DModel(**config)
    print(f"length of state dict: {len(state_dict.keys())}")
    print(f"length of value function dict: {len(hf_value_function.state_dict().keys())}")
    mapping = dict(zip(model.state_dict().keys(), hf_value_function.state_dict().keys()))
    for k, v in mapping.items():
        state_dict[v] = state_dict.pop(k)
    hf_value_function.load_state_dict(state_dict)

    torch.save(hf_value_function.state_dict(), f"hub/hopper-medium-v2/unet/hor{hor}/diffusion_pytorch_model.bin")
    with open(f"hub/hopper-medium-v2/unet/hor{hor}/config.json", "w") as f:
        json.dump(config, f)


def value_function():
    config = {
        "in_channels": 14,
        "down_block_types": ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
        "up_block_types": (),
        "out_block_type": "ValueFunction",
        "mid_block_type": "ValueFunctionMidBlock1D",
        "block_out_channels": (32, 64, 128, 256),
        "layers_per_block": 1,
        "downsample_each_block": True,
        "sample_size": 65536,
        "out_channels": 14,
        "extra_in_channels": 0,
        "time_embedding_type": "positional",
        "use_timestep_embedding": True,
        "flip_sin_to_cos": False,
        "freq_shift": 1,
        "norm_num_groups": 8,
        "act_fn": "mish",
    }

    model = torch.load("/Users/bglickenhaus/Documents/diffuser/value_function-hopper-mediumv2-hor32.torch")
    state_dict = model
    hf_value_function = UNet1DModel(**config)
    print(f"length of state dict: {len(state_dict.keys())}")
    print(f"length of value function dict: {len(hf_value_function.state_dict().keys())}")

    mapping = dict(zip(state_dict.keys(), hf_value_function.state_dict().keys()))
    for k, v in mapping.items():
        state_dict[v] = state_dict.pop(k)

    hf_value_function.load_state_dict(state_dict)

    torch.save(hf_value_function.state_dict(), "hub/hopper-medium-v2/value_function/diffusion_pytorch_model.bin")
    with open("hub/hopper-medium-v2/value_function/config.json", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    unet(32)
    # unet(128)
    value_function()
