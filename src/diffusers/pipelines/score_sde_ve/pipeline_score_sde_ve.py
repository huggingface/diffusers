#!/usr/bin/env python3
import torch

from diffusers import DiffusionPipeline
from tqdm.auto import tqdm


class ScoreSdeVePipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, num_inference_steps=2000, generator=None, output_type="pil"):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.sample_size
        shape = (1, 3, img_size, img_size)

        model = self.model.to(device)

        sample = torch.randn(*shape) * self.scheduler.config.sigma_max
        sample = sample.to(device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=device)

            for _ in range(self.scheduler.correct_steps):
                model_output = self.model(sample, sigma_t)

                if isinstance(model_output, dict):
                    model_output = model_output["sample"]

                sample = self.scheduler.step_correct(model_output, sample)["prev_sample"]

            with torch.no_grad():
                model_output = model(sample, sigma_t)

                if isinstance(model_output, dict):
                    model_output = model_output["sample"]

            output = self.scheduler.step_pred(model_output, t, sample)
            sample, sample_mean = output["prev_sample"], output["prev_sample_mean"]

        sample = sample.clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        return {"sample": sample}
