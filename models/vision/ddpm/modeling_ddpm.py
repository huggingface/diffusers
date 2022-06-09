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
                pred_noise_t = self.unet(image, t)

            # 2. compute alphas, betas
            alpha_prod_t = self.noise_scheduler.get_alpha_prod(t)
            alpha_prod_t_prev = self.noise_scheduler.get_alpha_prod(t - 1)
            beta_prod_t = (1 - alpha_prod_t)
            beta_prod_t_prev = (1 - alpha_prod_t_prev)

            # 3. compute predicted image from residual
            # First: compute predicted original image from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_image = (image - beta_prod_t.sqrt() * pred_noise_t) / alpha_prod_t.sqrt()

            # Second: Clip "predicted x_0"
            pred_original_image = torch.clamp(pred_original_image, -1, 1)

            # Third: Compute coefficients for pred_original_image x_0 and current image x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_image_coeff = (alpha_prod_t_prev.sqrt() * self.noise_scheduler.get_beta(t)) / beta_prod_t
            current_image_coeff = self.noise_scheduler.get_alpha(t).sqrt() * beta_prod_t_prev / beta_prod_t
            # Fourth: Compute predicted previous image µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_prev_image = pred_original_image_coeff * pred_original_image + current_image_coeff * image

            # 5. For t > 0, compute predicted variance βt (see formala (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
            # and sample from it to get previous image
            # x_{t-1} ~ N(pred_prev_image, variance) == add variane to pred_image
            if t > 0:
                variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.noise_scheduler.get_beta(t)).sqrt()
                noise = self.noise_scheduler.sample_noise(image.shape, device=image.device, generator=generator)
                sampled_variance = variance * noise
#                sampled_variance = self.noise_scheduler.sample_variance(
#                    t, pred_prev_image.shape, device=torch_device, generator=generator
#                )
                prev_image = pred_prev_image + sampled_variance
            else:
                prev_image = pred_prev_image

            # 6. Set current image to prev_image: x_t -> x_t-1
            image = prev_image

        return image
