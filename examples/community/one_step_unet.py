#!/usr/bin/env python3
import torch

from diffusers import DiffusionPipeline


class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(self):
        image = torch.randn(
            (1, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
        )
        timestep = 1

        model_output = self.unet(image, timestep).sample
        scheduler_output = self.scheduler.step(model_output, timestep, image).prev_sample

        result = scheduler_output - scheduler_output + torch.ones_like(scheduler_output)

        return result
