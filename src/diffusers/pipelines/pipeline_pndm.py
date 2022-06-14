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

        seq = list(inference_step_times)
        seq_next = [-1] + list(seq[:-1])
        model = self.unet

        warmup_time_steps = list(reversed([(t + 5) // 10 * 10 for t in range(seq[-4], seq[-1], 5)]))

        cur_residual = 0
        prev_image = image
        ets = []
        for i in range(len(warmup_time_steps)):
            t = warmup_time_steps[i] * torch.ones(image.shape[0])
            t_next = (warmup_time_steps[i + 1] if i < len(warmup_time_steps) - 1 else warmup_time_steps[-1]) * torch.ones(image.shape[0])

            residual = model(image.to("cuda"), t.to("cuda"))
            residual = residual.to("cpu")

            if i % 4 == 0:
                cur_residual += 1 / 6 * residual
                ets.append(residual)
                prev_image = image
            elif (i - 1) % 4 == 0:
                cur_residual += 1 / 3 * residual
            elif (i - 2) % 4 == 0:
                cur_residual += 1 / 3 * residual
            elif (i - 3) % 4 == 0:
                cur_residual += 1 / 6 * residual
                residual = cur_residual
                cur_residual = 0

            image = image.to("cpu")
            t_2 = warmup_time_steps[4 * (i // 4)] * torch.ones(image.shape[0])
            image = self.noise_scheduler.transfer(prev_image.to("cpu"), t_2, t_next, residual)

        step_idx = len(seq) - 4
        while step_idx >= 0:
            i = seq[step_idx]
            j = seq_next[step_idx]

            t = (torch.ones(image.shape[0]) * i)
            t_next = (torch.ones(image.shape[0]) * j)

            residual = model(image.to("cuda"), t.to("cuda"))
            residual = residual.to("cpu")
            ets.append(residual)
            residual = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])

            img_next = self.noise_scheduler.transfer(image.to("cpu"), t, t_next, residual)
            image = img_next

            step_idx = step_idx - 1

        return image
