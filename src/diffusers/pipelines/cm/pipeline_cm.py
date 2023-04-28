# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Tuple, Union

import torch

from ...models import UNet2DModel
from ...schedulers import CMScheduler
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class CMPipeline(DiffusionPipeline):
    r"""
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://arxiv.org/abs/2011.13456
    
    https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`CMScheduler`]):
            Scheduler for the diffusion process to be used in combination with `unet` to denoise the encoded image.
    """

    # add type hints for linting
    unet: UNet2DModel
    scheduler: CMScheduler

    def __init__(self, unet: UNet2DModel, scheduler: CMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        # sample x_0 ~ N(0, sigma_0^2 * I)
        sample = randn_tensor(shape, generator=generator, device=self.device) * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # One step sampling
        step_output = self.scheduler.step(model, sample)
        sample = step_output.prev_sample

        # Multi-step sampling (stochastic_iterative_sampler), if len(self.scheduler.timesteps[:-1]) > 0
        for i, t in self.progress_bar(enumerate(self.scheduler.timesteps[:-1])):

            # 1. Add new noise to move from sample_i to sample_hat
            sample_hat, sigma_hat, sigma = self.scheduler.add_noise_to_input(sample, generator=generator, i=i)

            # 2. Make prediction following consistency model x = f(x_\tau, \tau)
            step_output = self.scheduler.step(model, sample_hat, sigma)

            sample = step_output.prev_sample

        sample = (sample / 2 + 0.5).clamp(0, 1)
        image = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
