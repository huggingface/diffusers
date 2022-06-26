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

        beta_min, beta_max = 0.1, 20

        model = self.model.to(device)

        x = torch.randn(*shape).to(device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            t = t * torch.ones(shape[0], device=device)
            sigma_t = t * (num_inference_steps - 1)

            with torch.no_grad():
                result = model(x, sigma_t)

            log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            result = -result / std[:, None, None, None]

            x, x_mean = self.scheduler.step_pred(result, x, t)

        x_mean = (x_mean + 1.) / 2.

        return x_mean
