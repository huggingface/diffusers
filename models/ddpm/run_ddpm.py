#!/usr/bin/env python3
import torch

from diffusers import GaussianDiffusion, UNetConfig, UNetModel


config = UNetConfig(dim=64, dim_mults=(1, 2, 4, 8))
model = UNetModel(config)
print(model.config)

model.save_pretrained("/home/patrick/diffusion_example")

import ipdb


ipdb.set_trace()

diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000, loss_type="l1")  # number of steps  # L1 or L2

training_images = torch.randn(8, 3, 128, 128)  # your images need to be normalized from a range of -1 to +1
loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.sample(batch_size=4)
sampled_images.shape  # (4, 3, 128, 128)
