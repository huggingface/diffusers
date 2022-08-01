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

from tqdm.auto import tqdm

from ...models import UNet2DModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import RePaintScheduler


class RePaintPipeline(DiffusionPipeline):
    unet: UNet2DModel
    scheduler: RePaintScheduler

    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        original_image: torch.Tensor,
        mask: torch.Tensor,
        num_inference_steps=250,
        jump_length=10,
        jump_n_sample=10,
        generator=None,
        torch_device=None,
        output_type="pil",
    ):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        original_image = original_image.to(torch_device)
        mask = mask.to(torch_device)

        # sample gaussian noise to begin the loop
        sample = torch.randn(
            (original_image.shape[0], self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        sample = sample.to(torch_device)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps, jump_length, jump_n_sample)

        t_last = self.scheduler.timesteps[-1] + 1
        for t in tqdm(self.scheduler.timesteps):
            if t < t_last:
                # predict the noise residual
                model_output = self.unet(sample, t)["sample"]
                # compute previous image: x_t -> x_t-1
                sample = self.scheduler.step(model_output, t, sample, original_image, mask, generator)["prev_sample"]
            else:
                # compute the reverse: x_t-1 -> x_t
                sample = self.scheduler.undo_step(sample, t, generator)
            t_last = t

        sample = (sample / 2 + 0.5).clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        return {"sample": sample}
