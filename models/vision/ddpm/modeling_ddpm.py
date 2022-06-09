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


class DDPM(DiffusionPipeline):
    def __init__(self, unet, noise_scheduler):
        super().__init__()
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler)

    def __call__(self, batch_size=1, generator=None, torch_device=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop
        image = self.noise_scheduler.sample_noise(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            device=torch_device,
            generator=generator,
        )

        for t in tqdm.tqdm(reversed(range(len(self.noise_scheduler))), total=len(self.noise_scheduler)):
            # 1. predict noise residual
            with torch.no_grad():
                noise_residual = self.unet(image, t)

            # 2. compute alphas, betas
            alpha_prod_t = self.noise_scheduler.get_alpha_prod(t)
            alpha_prod_t_prev = self.noise_scheduler.get_alpha_prod(t - 1)
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 3. compute predicted image from residual
            # See 2nd formula at https://github.com/hojonathanho/diffusion/issues/5#issue-896554416 for comparison
            # First: Compute inner formula
            pred_mean = (1 / alpha_prod_t.sqrt()) * (image - beta_prod_t.sqrt() * noise_residual)
            # Second: Clip
            pred_mean = torch.clamp(pred_mean, -1, 1)
            # Third: Compute outer coefficients
            pred_mean_coeff = (alpha_prod_t_prev.sqrt() * self.noise_scheduler.get_beta(t)) / beta_prod_t
            image_coeff = (beta_prod_t_prev * self.noise_scheduler.get_alpha(t).sqrt()) / beta_prod_t 
            # Fourth: Compute outer formula
            prev_image = pred_mean_coeff * pred_mean + image_coeff * image

            # 4. sample variance
            prev_variance = self.noise_scheduler.sample_variance(
                t, prev_image.shape, device=torch_device, generator=generator
            )

            # 5. sample  x_{t-1} ~ N(prev_image, prev_variance) = add variance to predicted image
            sampled_prev_image = prev_image + prev_variance
            image = sampled_prev_image

        return image
