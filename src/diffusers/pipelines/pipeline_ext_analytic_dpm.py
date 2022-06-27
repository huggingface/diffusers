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
import torch

import tqdm

from ..pipeline_utils import DiffusionPipeline


class SNDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, sn_net, noise_scheduler):
        super().__init__()
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler, sn_net=sn_net)

    def __call__(self, batch_size=1, generator=None, torch_device=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        self.sn_net.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            generator=generator,
        )
        image = image.to(torch_device)

        num_prediction_steps = len(self.noise_scheduler)
        for t in tqdm.tqdm(reversed(range(num_prediction_steps)), total=num_prediction_steps):
            # 1. predict noise residual
            with torch.no_grad():
                residual = self.unet(image, t)
                noise_square = self.sn_net(image, t)
                noise_residual_square = noise_square - residual**2

            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.noise_scheduler.step(residual, image, t)

            # 3. optionally sample variance via equation (10)
            variance = 0
            if t > 0:
                gamma = (torch.sqrt(self.noise_scheduler.alphas_cumprod[t - 1]) * self.noise_scheduler.betas[t]) / (
                    1 - self.noise_scheduler.alphas_cumprod[t]
                )
                beta_bar_div_alpha_bar = (
                    1 - self.noise_scheduler.alphas_cumprod[t]
                ) / self.noise_scheduler.alphas_cumprod[t]
                noise = torch.randn(image.shape, generator=generator).to(image.device)
                variance = (
                    self.noise_scheduler.get_variance(t) + gamma**2 * beta_bar_div_alpha_bar * noise_residual_square
                ).sqrt() * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        return image


class NPRDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, npr_net, noise_scheduler):
        super().__init__()
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler, npr_net=npr_net)

    def __call__(self, batch_size=1, generator=None, torch_device=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        self.npr_net.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            generator=generator,
        )
        image = image.to(torch_device)

        num_prediction_steps = len(self.noise_scheduler)
        for t in tqdm.tqdm(reversed(range(num_prediction_steps)), total=num_prediction_steps):
            # 1. predict noise residual
            with torch.no_grad():
                residual = self.unet(image, t)
                noise_pred_residual_square = self.npr_net(image, t)

            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.noise_scheduler.step(residual, image, t)

            # 3. optionally sample variance via equation (16)
            variance = 0
            if t > 0:
                gamma = (torch.sqrt(self.noise_scheduler.alphas_cumprod[t - 1]) * self.noise_scheduler.betas[t]) / (
                    1 - self.noise_scheduler.alphas_cumprod[t]
                )
                beta_bar_div_alpha_bar = (
                    1 - self.noise_scheduler.alphas_cumprod[t]
                ) / self.noise_scheduler.alphas_cumprod[t]
                noise = torch.randn(image.shape, generator=generator).to(image.device)
                variance = (
                    self.noise_scheduler.get_variance(t)
                    + gamma**2 * beta_bar_div_alpha_bar * noise_pred_residual_square
                ).sqrt() * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        return image
