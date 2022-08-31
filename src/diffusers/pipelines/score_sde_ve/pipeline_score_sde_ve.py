#!/usr/bin/env python3
import warnings

import torch

from diffusers import DiffusionPipeline


class ScoreSdeVePipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, num_inference_steps=2000, generator=None, output_type="pil", **kwargs):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        sample = torch.randn(*shape, generator=generator) * self.scheduler.config.sigma_max
        sample = sample.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=self.device)

            # correction step
            for _ in range(self.scheduler.correct_steps):
                model_output = self.unet(sample, sigma_t)["sample"]
                sample = self.scheduler.step_correct(model_output, sample, generator=generator)["prev_sample"]

            # prediction step
            model_output = model(sample, sigma_t)["sample"]
            output = self.scheduler.step_pred(model_output, t, sample, generator=generator)

            sample, sample_mean = output["prev_sample"], output["prev_sample_mean"]

        sample = sample_mean.clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        return {"sample": sample}
