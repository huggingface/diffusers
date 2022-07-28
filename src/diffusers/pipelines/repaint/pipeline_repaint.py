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

from ...pipeline_utils import DiffusionPipeline


class RePaintPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size=1,
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

        # sample gaussian noise to begin the loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(torch_device)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps, jump_length, jump_n_sample)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict the noise residual
            model_output = self.unet(image, t)["sample"]

            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image)["prev_sample"]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
