#!/usr/bin/env python3
import torch

from diffusers import DiffusionPipeline
from tqdm.auto import tqdm


class ScoreSdeVePipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, num_inference_steps=2000, generator=None, torch_device=None, output_type="pil"):

        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet.to(torch_device)

        sample = torch.randn(*shape) * self.scheduler.config.sigma_max
        sample = sample.to(torch_device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=torch_device)

            # correction step
            for _ in range(self.scheduler.correct_steps):
                model_output = self.unet(sample, sigma_t)["sample"]
                sample = self.scheduler.step_correct(model_output, sample)["prev_sample"]

            # prediction step
            model_output = model(sample, sigma_t)["sample"]
            output = self.scheduler.step_pred(model_output, t, sample)

            sample, sample_mean = output["prev_sample"], output["prev_sample_mean"]

        sample = sample_mean.clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        return {"sample": sample}
