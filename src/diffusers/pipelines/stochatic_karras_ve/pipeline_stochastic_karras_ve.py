#!/usr/bin/env python3
import torch

from tqdm.auto import tqdm

from ...models import UNet2DModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import KarrasVeScheduler


class KarrasVePipeline(DiffusionPipeline):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2].
    Use Algorithm 2 and the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364
    [2] Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." https://arxiv.org/abs/2011.13456
    """

    unet: UNet2DModel
    scheduler: KarrasVeScheduler

    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, num_inference_steps=50, generator=None, torch_device=None, output_type="pil"):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet.to(torch_device)

        # sample x_0 ~ N(0, sigma_0^2 * I)
        sample = torch.randn(*shape) * self.scheduler.config.sigma_max
        sample = sample.to(torch_device)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # here sigma_t == t_i from the paper
            sigma = self.scheduler.schedule[t]
            sigma_prev = self.scheduler.schedule[t - 1] if t > 0 else 0

            # 1. Select temporarily increased noise level sigma_hat
            # 2. Add new noise to move from sample_i to sample_hat
            sample_hat, sigma_hat = self.scheduler.add_noise_to_input(sample, sigma, generator=generator)

            # 3. Predict the noise residual given the noise magnitude `sigma_hat`
            # The model inputs and output are adjusted by following eq. (213) in [1].
            model_output = (sigma_hat / 2) * model((sample_hat + 1) / 2, sigma_hat / 2)["sample"]

            # 4. Evaluate dx/dt at sigma_hat
            # 5. Take Euler step from sigma to sigma_prev
            step_output = self.scheduler.step(model_output, sigma_hat, sigma_prev, sample_hat)

            if sigma_prev != 0:
                # 6. Apply 2nd order correction
                # The model inputs and output are adjusted by following eq. (213) in [1].
                model_output = (sigma_prev / 2) * model((step_output["prev_sample"] + 1) / 2, sigma_prev / 2)["sample"]
                step_output = self.scheduler.step_correct(
                    model_output,
                    sigma_hat,
                    sigma_prev,
                    sample_hat,
                    step_output["prev_sample"],
                    step_output["derivative"],
                )
            sample = step_output["prev_sample"]

        sample = (sample / 2 + 0.5).clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        return {"sample": sample}
