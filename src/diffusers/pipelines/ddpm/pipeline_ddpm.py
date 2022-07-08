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

from ...models import UNetModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DiscreteScheduler


class DDPMPipeline(DiffusionPipeline):
    unet: UNetModel
    noise_scheduler: DiscreteScheduler

    def __init__(self, unet: UNetModel, noise_scheduler: DiscreteScheduler):
        super().__init__()
        noise_scheduler = noise_scheduler.set_format("pt")
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler)

    @torch.no_grad()
    def __call__(self, batch_size: int = 1, seed: int = None, device: int = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        random_generator = torch.manual_seed(seed)
        self.unet.to(device)

        sample = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            generator=random_generator,
        )
        sample = sample.to(device)

        for t in reversed(range(self.noise_scheduler.num_timesteps)):
            noise_prediction = self.unet(sample, t)

            noise = torch.randn(sample.shape, generator=random_generator).to(device)
            sample = self.noise_scheduler.step(noise_prediction, sample, t, noise)

        image = (sample / 2 + 0.5).cpu().permute(0, 2, 3, 1).numpy()

        return image
