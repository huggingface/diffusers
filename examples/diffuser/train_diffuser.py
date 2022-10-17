import numpy as np
import torch

import d4rl  # noqa
import gym
from diffusers import DDPMScheduler, UNet1DModel


env_name = "hopper-medium-expert-v2"
env = gym.make(env_name)
data = env.get_dataset()  # dataset is only used for normalization in this colab

# Cuda settings for colab
# torch.cuda.get_device_name(0)
DEVICE = "cpu"
DTYPE = torch.float

# diffusion model settings
n_samples = 4  # number of trajectories planned via diffusion
horizon = 128  # length of sampled trajectories
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
num_inference_steps = 100  # number of difusion steps


def normalize(x_in, data, key):
    upper = np.max(data[key], axis=0)
    lower = np.min(data[key], axis=0)
    x_out = 2 * (x_in - lower) / (upper - lower) - 1
    return x_out


def de_normalize(x_in, data, key):
    upper = np.max(data[key], axis=0)
    lower = np.min(data[key], axis=0)
    x_out = lower + (upper - lower) * (1 + x_in) / 2
    return x_out


def to_torch(x_in, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x_in) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x_in.items()}
    elif torch.is_tensor(x_in):
        return x_in.to(device).type(dtype)
    return torch.tensor(x_in, dtype=dtype, device=device)


obs = env.reset()
obs_raw = obs

# normalize observations for forward passes
obs = normalize(obs, data, "observations")


# Two generators for different parts of the diffusion loop to work in colab
generator = torch.Generator(device="cuda")
generator_cpu = torch.Generator(device="cpu")
network = UNet1DModel.from_pretrained("fusing/ddpm-unet-rl-hopper-hor128").to(device=DEVICE)

scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
optimizer = torch.optim.AdamW(
    network.parameters(),
    lr=0.001,
    betas=(0.95, 0.99),
    weight_decay=1e-6,
    eps=1e-8,
)
# 3 different pretrained models are available for this task.
# The horizion represents the length of trajectories used in training.
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor256").to(device=DEVICE)
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor512").to(device=DEVICE)


def reset_x0(x_in, cond, act_dim):
    for key, val in cond.items():
        x_in[:, key, act_dim:] = val.clone()
    return x_in


# TODO: Flesh this out using accelerate library (a la other examples)
