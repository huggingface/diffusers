#!/usr/bin/env python3
import torch

from diffusers import DiffusionPipeline


# TODO(Patrick, Anton, Suraj) - rename `x` to better variable names
class ScoreSdeVpPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    def __call__(self, num_inference_steps=1000, generator=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.image_size
        channels = self.model.config.num_channels
        shape = (1, channels, img_size, img_size)

        model = self.model.to(device)

        x = torch.randn(*shape).to(device)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            t = t * torch.ones(shape[0], device=device)
            scaled_t = t * (num_inference_steps - 1)

            with torch.no_grad():
                result = model(x, scaled_t)

            x, x_mean = self.scheduler.step_pred(result, x, t)

        x_mean = (x_mean + 1.0) / 2.0

        return x_mean
