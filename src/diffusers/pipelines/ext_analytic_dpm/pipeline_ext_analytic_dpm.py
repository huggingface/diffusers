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

from ...pipeline_utils import DiffusionPipeline


class SNDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, sn_net, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler, sn_net=sn_net)

    def __call__(self, batch_size=1, generator=None, torch_device=None, output_type="pil"):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        self.sn_net.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(torch_device)

        # set step values
        self.scheduler.set_timesteps(1000)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise residual
            model_output = self.unet(image, t)["sample"]
            noise_square = self.sn_net(image, t)["sample"]
            noise_residual_square = noise_square - model_output**2

            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.scheduler.step(model_output, t, image)["prev_sample"]

            # 3. optionally sample variance via equation (10)
            variance = 0
            if t > 0:
                gamma = (torch.sqrt(self.scheduler.alphas_cumprod[t - 1]) * self.scheduler.betas[t]) / (
                    1 - self.scheduler.alphas_cumprod[t]
                )
                beta_bar_div_alpha_bar = (1 - self.scheduler.alphas_cumprod[t]) / self.scheduler.alphas_cumprod[t]
                noise = torch.randn(model_output.shape, generator=generator).to(model_output.device)
                variance = (
                    self.scheduler.get_variance(t) + gamma**2 * beta_bar_div_alpha_bar * noise_residual_square
                ).sqrt() * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}


class NPRDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, npr_net, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler, npr_net=npr_net)

    def __call__(self, batch_size=1, generator=None, torch_device=None, output_type="pil"):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        self.npr_net.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(torch_device)

        # set step values
        self.scheduler.set_timesteps(1000)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise and residual
            model_output = self.unet(image, t)["sample"]
            noise_pred_residual_square = self.npr_net(image, t)["sample"]

            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.scheduler.step(model_output, t, image)["prev_sample"]

            # 3. optionally sample variance via equation (16)
            variance = 0
            if t > 0:
                gamma = (torch.sqrt(self.scheduler.alphas_cumprod[t - 1]) * self.scheduler.betas[t]) / (
                    1 - self.scheduler.alphas_cumprod[t]
                )
                beta_bar_div_alpha_bar = (1 - self.scheduler.alphas_cumprod[t]) / self.scheduler.alphas_cumprod[t]
                noise = torch.randn(model_output.shape, generator=generator).to(model_output.device)
                variance = (
                    self.scheduler.get_variance(t) + gamma**2 * beta_bar_div_alpha_bar * noise_pred_residual_square
                ).sqrt() * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
