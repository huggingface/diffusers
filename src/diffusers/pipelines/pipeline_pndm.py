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

        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            generator=generator,
        )
        image = image.to(torch_device)

        warmup_time_steps = self.noise_scheduler.get_warmup_time_steps(num_inference_steps)
        prev_image = image
        for t in tqdm.tqdm(range(len(warmup_time_steps))):
            t_orig = warmup_time_steps[t]
            residual = self.unet(image, t_orig)

            if t % 4 == 0:
                prev_image = image

            image = self.noise_scheduler.step_warm_up(residual, prev_image, t, num_inference_steps)

        timesteps = self.noise_scheduler.get_time_steps(num_inference_steps)
        for t in tqdm.tqdm(range(len(timesteps))):
            t_orig = timesteps[t]
            residual = self.unet(image, t_orig)

            image = self.noise_scheduler.step(residual, image, t, num_inference_steps)

        return image
