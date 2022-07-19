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


class DDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(self, batch_size=1, generator=None, torch_device=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.image_size, self.unet.image_size),
            generator=generator,
        )
        image = image.to(torch_device)

        num_prediction_steps = len(self.scheduler)
        for t in tqdm(reversed(range(num_prediction_steps)), total=num_prediction_steps):
            # 1. predict noise model_output
            with torch.no_grad():
                model_output = self.unet(image, t)

                if isinstance(model_output, dict):
                    model_output = model_output["sample"]

            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.scheduler.step(model_output, t, image)["prev_sample"]

            # 3. optionally sample variance
            variance = 0
            if t > 0:
                noise = torch.randn(image.shape, generator=generator).to(image.device)
                variance = self.scheduler.get_variance(t).sqrt() * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        return image
