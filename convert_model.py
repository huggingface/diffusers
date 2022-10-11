
import torch
from diffusers import DDPMScheduler, TemporalUNet, ValueFunction, ValueFunctionScheduler



model = torch.load("/Users/bglickenhaus/Documents/diffuser/temporal_unet-hopper-hor32.torch")
state_dict = model.state_dict()
hf_value_function = TemporalUNet(training_horizon=32, dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14, cond_dim=11)
mapping = dict((k, hfk) for k, hfk in zip(model.state_dict().keys(), hf_value_function.state_dict().keys()))
for k, v in mapping.items():
    state_dict[v] = state_dict.pop(k)
hf_value_function.load_state_dict(state_dict)

torch.save(hf_value_function.state_dict(), "hub/hopper-medium-v2-unet/diffusion_pytorch_model.bin")
