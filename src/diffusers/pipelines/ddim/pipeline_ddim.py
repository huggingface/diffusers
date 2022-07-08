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


class DDIMPipeline(DiffusionPipeline):
    unet: UNetModel
    noise_scheduler: DiscreteScheduler

    def __init__(self, unet, noise_scheduler):
        super().__init__()
        noise_scheduler = noise_scheduler.set_format("pt")
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        eta: float = 0.0,
        num_inference_steps: int = None,
        seed: int = None,
        device: str = None,
    ):
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        random_generator = torch.manual_seed(seed)
        self.unet.to(device)

        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution),
            generator=random_generator,
        )
        image = image.to(device)

        self.noise_scheduler.set_num_inference_steps(num_inference_steps)

        for t in reversed(range(num_inference_steps)):
            # adjust the reduced timestep to the number of training timesteps
            t = t * (self.noise_scheduler.num_timesteps // num_inference_steps)
            noise_prediction = self.unet(image, t)
            noise = torch.randn(image.shape, generator=random_generator).to(device)
            image = self.noise_scheduler.step(noise_prediction, image, t, eta=eta, noise=noise)

        image = (image / 2 + 0.5).cpu().permute(0, 2, 3, 1).numpy()

        return image
