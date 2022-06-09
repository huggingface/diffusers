# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# limitations under the License.


import torch

import tqdm
from diffusers import DiffusionPipeline


class DDIM(DiffusionPipeline):
    def __init__(self, unet, noise_scheduler):
        super().__init__()
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler)

    def __call__(self, batch_size=1, generator=None, torch_device=None, eta=0.0, num_inference_steps=50):
        # eta corresponds to η in paper and should be between [0, 1]
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        num_trained_timesteps = self.noise_scheduler.num_timesteps
        inference_step_times = range(0, num_trained_timesteps, num_trained_timesteps // num_inference_steps)

        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop
        image = self.noise_scheduler.sample_noise(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            device=torch_device,
            generator=generator,
        )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_image -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_image_direction -> "direction pointingc to x_t"
        # - pred_prev_image -> "x_t-1"
        for t in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
            # 1. predict noise residual
            with torch.no_grad():
                pred_noise_t = self.unet(image, inference_step_times[t])

            # 2. get actual t and t-1
            train_step = inference_step_times[t]
            prev_train_step = inference_step_times[t - 1] if t > 0 else -1

            # 3. compute alphas, betas
            alpha_prod_t = self.noise_scheduler.get_alpha_prod(train_step)
            alpha_prod_t_prev = self.noise_scheduler.get_alpha_prod(prev_train_step)
            beta_prod_t = (1 - alpha_prod_t)
            beta_prod_t_prev = (1 - alpha_prod_t_prev)

            # 4. Compute predicted previous image from predicted noise
            # First: compute predicted original image from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_image = (image - beta_prod_t.sqrt() * pred_noise_t) / alpha_prod_t.sqrt()

            # Second: Clip "predicted x_0"
            pred_original_image = torch.clamp(pred_original_image, -1, 1)

            # Third: Compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            std_dev_t = (beta_prod_t_prev / beta_prod_t).sqrt() * (1 - alpha_prod_t / alpha_prod_t_prev).sqrt()
            std_dev_t = eta * std_dev_t

            # Fourth: Compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_image_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_noise_t

            # Fifth: Compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_prev_image = alpha_prod_t_prev.sqrt() * pred_original_image + pred_image_direction

            # 5. Sample x_t-1 image optionally if η > 0.0 by adding noise to pred_prev_image
            # Note: eta = 1.0 essentially corresponds to DDPM
            if eta > 0.0:
                noise = self.noise_scheduler.sample_noise(image.shape, device=image.device, generator=generator)
                prev_image = pred_prev_image + std_dev_t * noise
            else:
                prev_image = pred_prev_image

            # 6. Set current image to prev_image: x_t -> x_t-1
            image = prev_image

        return image
