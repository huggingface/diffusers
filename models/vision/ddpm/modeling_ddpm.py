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


from diffusers import DiffusionPipeline
import tqdm
import torch


class DDPM(DiffusionPipeline):

    modeling_file = "modeling_ddpm.py"

    def __init__(self, unet, noise_scheduler):
        super().__init__()
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler)

    def __call__(self, batch_size=1, generator=None, torch_device=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        # 1. Sample gaussian noise
        image = self.noise_scheduler.sample_noise((batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution), device=torch_device, generator=generator)
        for t in tqdm.tqdm(reversed(range(len(self.noise_scheduler))), total=len(self.noise_scheduler)):
            # i) define coefficients for time step t
            clipped_image_coeff = 1 / torch.sqrt(self.noise_scheduler.get_alpha_prod(t))
            clipped_noise_coeff = torch.sqrt(1 / self.noise_scheduler.get_alpha_prod(t) - 1)
            image_coeff = (1 - self.noise_scheduler.get_alpha_prod(t - 1)) * torch.sqrt(self.noise_scheduler.get_alpha(t)) / (1 - self.noise_scheduler.get_alpha_prod(t))
            clipped_coeff = torch.sqrt(self.noise_scheduler.get_alpha_prod(t - 1)) * self.noise_scheduler.get_beta(t) / (1 - self.noise_scheduler.get_alpha_prod(t))

            # ii) predict noise residual
            with torch.no_grad():
                noise_residual = self.unet(image, t)

            # iii) compute predicted image from residual
            # See 2nd formula at https://github.com/hojonathanho/diffusion/issues/5#issue-896554416 for comparison
            pred_mean = clipped_image_coeff * image - clipped_noise_coeff * noise_residual
            pred_mean = torch.clamp(pred_mean, -1, 1)
            prev_image = clipped_coeff * pred_mean + image_coeff * image

            # iv) sample variance
            prev_variance = self.noise_scheduler.sample_variance(t, prev_image.shape, device=torch_device, generator=generator)

            # v) sample  x_{t-1} ~ N(prev_image, prev_variance)
            sampled_prev_image = prev_image + prev_variance
            image = sampled_prev_image

        return image
