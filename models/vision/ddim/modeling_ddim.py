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
        image = self.noise_scheduler.sample_noise((batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution), device=torch_device, generator=generator)

        for t in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
            # get actual t and t-1
            train_step = inference_step_times[t]
            prev_train_step = inference_step_times[t - 1] if t > 0 else - 1

            # compute alphas
            alpha_prod_t = self.noise_scheduler.get_alpha_prod(train_step)
            alpha_prod_t_prev = self.noise_scheduler.get_alpha_prod(prev_train_step)
            alpha_prod_t_rsqrt = 1 / alpha_prod_t.sqrt()
            alpha_prod_t_prev_rsqrt = 1 / alpha_prod_t_prev.sqrt()
            beta_prod_t_sqrt = (1 - alpha_prod_t).sqrt()
            beta_prod_t_prev_sqrt = (1 - alpha_prod_t_prev).sqrt()

            # compute relevant coefficients
            coeff_1 = (alpha_prod_t_prev - alpha_prod_t).sqrt() * alpha_prod_t_prev_rsqrt * beta_prod_t_prev_sqrt / beta_prod_t_sqrt * eta
            coeff_2 = ((1 - alpha_prod_t_prev) - coeff_1 ** 2).sqrt()

            # model forward
            with torch.no_grad():
                noise_residual = self.unet(image, train_step)

            # predict mean of prev image
            pred_mean = alpha_prod_t_rsqrt * (image - beta_prod_t_sqrt * noise_residual)
            pred_mean = torch.clamp(pred_mean, -1, 1)
            pred_mean = (1 / alpha_prod_t_prev_rsqrt) * pred_mean + coeff_2 * noise_residual

            # if eta > 0.0 add noise. Note eta = 1.0 essentially corresponds to DDPM
            if eta > 0.0:
                noise = self.noise_scheduler.sample_noise(image.shape, device=image.device, generator=generator)
                image = pred_mean + coeff_1 * noise
            else:
                image = pred_mean

        return image
