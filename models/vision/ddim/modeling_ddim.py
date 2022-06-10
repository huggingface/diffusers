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

        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop
        image = self.noise_scheduler.sample_noise(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            device=torch_device,
            generator=generator,
        )

        for t in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
            # 1. predict noise residual
            orig_t = self.noise_scheduler.get_orig_t(t, num_inference_steps)
            with torch.no_grad():
                residual = self.unet(image, orig_t)

            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.noise_scheduler.compute_prev_image_step(residual, image, t, num_inference_steps, eta)

            # 3. optionally sample variance
            variance = 0
            if eta > 0:
                noise = self.noise_scheduler.sample_noise(image.shape, device=image.device, generator=generator)
                variance = self.noise_scheduler.get_variance(t).sqrt() * eta * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        return image
