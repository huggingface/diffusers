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

from ..pipeline_utils import DiffusionPipeline


class PNDM(DiffusionPipeline):
    def __init__(self, unet, noise_scheduler):
        super().__init__()
        noise_scheduler = noise_scheduler.set_format("pt")
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler)

    def __call__(self, batch_size=1, generator=None, torch_device=None, num_inference_steps=50):
        # eta corresponds to η in paper and should be between [0, 1]
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        num_trained_timesteps = self.noise_scheduler.timesteps
        inference_step_times = range(0, num_trained_timesteps, num_trained_timesteps // num_inference_steps)

        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            generator=generator,
        )
        image = image.to(torch_device)

        seq = inference_step_times
        seq_next = [-1] + list(seq[:-1])
        model = self.unet

        ets = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(image.shape[0]) * i)
            t_next = (torch.ones(image.shape[0]) * j)

            residual = model(image.to("cuda"), t.to("cuda"))
            residual = residual.to("cpu")

            t_list = [t, (t+t_next)/2, t_next]

            if len(ets) <= 2:
                ets.append(residual)
                image = image.to("cpu")
                x_2 = self.noise_scheduler.transfer(image, t_list[0], t_list[1], residual)
                e_2 = model(x_2.to("cuda"), t_list[1].to("cuda")).to("cpu")
                x_3 = self.noise_scheduler.transfer(image, t_list[0], t_list[1], e_2)
                e_3 = model(x_3.to("cuda"), t_list[1].to("cuda")).to("cpu")
                x_4 = self.noise_scheduler.transfer(image, t_list[0], t_list[2], e_3)
                e_4 = model(x_4.to("cuda"), t_list[2].to("cuda")).to("cpu")
                residual = (1 / 6) * (residual + 2 * e_2 + 2 * e_3 + e_4)
            else:
                ets.append(residual)
                residual = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])

            img_next = self.noise_scheduler.transfer(image.to("cpu"), t, t_next, residual)

#            with torch.no_grad():
#                t_start, t_end = t_next, t
#                img_next, ets = self.noise_scheduler.step(image, t_start, t_end, model, ets)

            image = img_next

        return image

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_image -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_image_direction -> "direction pointingc to x_t"
        # - pred_prev_image -> "x_t-1"
#        for t in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
            # 1. predict noise residual
#            with torch.no_grad():
#                residual = self.unet(image, inference_step_times[t])
#
            # 2. predict previous mean of image x_t-1
#            pred_prev_image = self.noise_scheduler.step(residual, image, t, num_inference_steps, eta)
#
            # 3. optionally sample variance
#            variance = 0
#            if eta > 0:
#                noise = torch.randn(image.shape, generator=generator).to(image.device)
#                variance = self.noise_scheduler.get_variance(t, num_inference_steps).sqrt() * eta * noise
#
            # 4. set current image to prev_image: x_t -> x_t-1
#            image = pred_prev_image + variance
