#!/usr/bin/env python3
import torch

from diffusers import DiffusionPipeline
from tqdm.auto import tqdm


class StochasticKarrasPipeline(DiffusionPipeline):
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

        sample = torch.randn(*shape) * self.scheduler.config.sigma_max
        sample = sample.to(torch_device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i in tqdm(self.scheduler.timesteps):
            t = self.scheduler.schedule[i]
            t_prev = self.scheduler.schedule[i - 1] if i > 0 else 0
            x_hat, t_hat = self.scheduler.get_model_inputs(sample, t, generator=generator)

            D = x_hat + t_hat * model(x_hat, torch.log(0.5 * t_hat))["sample"]
            d = (x_hat - D) / t_hat
            x_prev = x_hat + (t_prev - t_hat) * d

            if t_prev != 0:
                D_prev = x_prev + t_prev * model(x_prev, torch.log(0.5 * t_prev))["sample"]

                d_prev = (x_prev - D_prev) / t_prev
                x_prev = x_hat + (t_prev - t_hat) * (0.5 * d + 0.5 * d_prev)
            sample = x_prev
            print(i, sample.min().item(), sample.max().item())


        sample = (sample / 2 + 0.5).clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        return {"sample": sample}
