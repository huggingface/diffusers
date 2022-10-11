
import torch
from diffusers import DDPMScheduler, UNet1DModel, ValueFunction, ValueFunctionScheduler
import os
import json
os.makedirs("hub/hopper-medium-v2/unet", exist_ok=True)
os.makedirs("hub/hopper-medium-v2/value_function", exist_ok=True)

def unet():
    model = torch.load("/Users/bglickenhaus/Documents/diffuser/temporal_unet-hopper-hor32.torch")
    state_dict = model.state_dict()
    hf_value_function = UNet1DModel(dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14)
    mapping = dict((k, hfk) for k, hfk in zip(model.state_dict().keys(), hf_value_function.state_dict().keys()))
    for k, v in mapping.items():
        state_dict[v] = state_dict.pop(k)
    hf_value_function.load_state_dict(state_dict)

    torch.save(hf_value_function.state_dict(), "hub/hopper-medium-v2/unet/diffusion_pytorch_model.bin")
    config = dict(dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14)
    with open("hub/hopper-medium-v2/unet/config.json", "w") as f:
        json.dump(config, f)

def value_function():
    model = torch.load("/Users/bglickenhaus/Documents/diffuser/value_function-hopper-hor32.torch")
    state_dict = model.state_dict()
    hf_value_function = ValueFunction(dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14)
    print(f"length of state dict: {len(state_dict.keys())}")
    print(f"length of value function dict: {len(hf_value_function.state_dict().keys())}")

    mapping = dict((k, hfk) for k, hfk in zip(model.state_dict().keys(), hf_value_function.state_dict().keys()))
    for k, v in mapping.items():
        state_dict[v] = state_dict.pop(k)

    hf_value_function.load_state_dict(state_dict)

    torch.save(hf_value_function.state_dict(), "hub/hopper-medium-v2/value_function/diffusion_pytorch_model.bin")
    config = dict(dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14)
    with open("hub/hopper-medium-v2/value_function/config.json", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    unet()
    value_function()