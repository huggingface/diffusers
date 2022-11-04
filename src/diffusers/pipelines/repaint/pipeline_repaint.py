# Copyright 2022 ETH Zurich Computer Vision Lab and The HuggingFace Team. All rights reserved.
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


from typing import Optional, Tuple, Union

import numpy as np
import torch

import PIL
from tqdm.auto import tqdm

from ...models import UNet2DModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import RePaintScheduler


def _preprocess_image(image: PIL.Image.Image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


def _preprocess_mask(mask: PIL.Image.Image):
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    return mask


class RePaintPipeline(DiffusionPipeline):
    unet: UNet2DModel
    scheduler: RePaintScheduler

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        original_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        num_inference_steps: int = 250,
        eta: float = 0.0,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            original_image (`torch.FloatTensor` or `PIL.Image.Image`):
                The original image to inpaint on.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                The mask_image where 0.0 values define which part of the original image to inpaint (change).
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`):
                The weight of noise for added noise in a diffusion step. Its value is between 0.0 and 1.0 - 0.0 is DDIM
                and 1.0 is DDPM scheduler respectively.
            jump_length (`int`, *optional*, defaults to 10):
                The number of steps taken forward in time before going backward in time for a single jump ("j" in
                RePaint paper). Take a look at Figure 9 and 10 in https://arxiv.org/pdf/2201.09865.pdf.
            jump_n_sample (`int`, *optional*, defaults to 10):
                The number of times we will make forward time jump for a given chosen time sample. Take a look at
                Figure 9 and 10 in https://arxiv.org/pdf/2201.09865.pdf.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if not isinstance(original_image, torch.FloatTensor):
            original_image = _preprocess_image(original_image)
        original_image = original_image.to(self.device)
        if not isinstance(mask_image, torch.FloatTensor):
            mask_image = _preprocess_mask(mask_image)
        mask_image = mask_image.to(self.device)

        # sample gaussian noise to begin the loop
        image = torch.randn(
            original_image.shape,
            generator=generator,
            device=self.device,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, jump_length, jump_n_sample, self.device)
        self.scheduler.eta = eta

        t_last = self.scheduler.timesteps[0] + 1
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            if t < t_last:
                # predict the noise residual
                model_output = self.unet(image, t).sample
                # compute previous image: x_t -> x_t-1
                image = self.scheduler.step(model_output, t, image, original_image, mask_image, generator).prev_sample

            else:
                # compute the reverse: x_t-1 -> x_t
                image = self.scheduler.undo_step(image, t_last, generator)
            t_last = t

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
