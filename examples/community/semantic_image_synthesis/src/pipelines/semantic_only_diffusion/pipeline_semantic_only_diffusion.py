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
import torch.nn as nn

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

from ...models import AutoencoderKL, UNet2DSISModel


class SemanticOnlyDDMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation conditionned by a Semantic Mask

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image.
        scheduler ([`ScoreSdeVeScheduler`]):
            A `ScoreSdeVeScheduler` to be used in combination with `unet` to denoise the encoded image.
    """
    unet: UNet2DSISModel
    scheduler: DDPMScheduler

    def __init__(self, unet: UNet2DSISModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        segmap: torch.Tensor = None,
        num_inference_steps: int = 2000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        guidance_scale: float = 7.5,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            guidance_scale (`float`):
                The guidance scale for the image generation.
                default = 7.5,
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        img_size = self.unet.config.img_size
        img_depth = self.unet.config.in_channels
        shape = (batch_size, img_depth, img_size, img_size)

        # We reshape the segmentation map to match the diffusion size.
        while len(segmap.shape) < 4:
            segmap = segmap.unsqueeze(1)
        segmap = nn.UpsamplingNearest2d(size=shape[-2:])(segmap)
        model = self.unet

        sample = randn_tensor(shape, generator=generator, device=generator.device) * self.scheduler.init_noise_sigma
        sample = sample.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # prediction step (with segmentation map as input)
            model_input = torch.cat([sample] * 2)
            # Like in the documentation example, we scale model_input
            # https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#denoise-the-image
            model_input = self.scheduler.scale_model_input(model_input, timestep=t)
            model_cond = torch.cat([segmap, torch.zeros_like(segmap)])
            with torch.no_grad():
                noise_pred = model(model_input, t, model_cond).sample
            noise_pred_cond, noise_pred_ucond = noise_pred.chunk(2)
            noise_pred = noise_pred_ucond + guidance_scale * (noise_pred_cond - noise_pred_ucond)
            # We finally use this noise to sample
            sample = self.scheduler.step(noise_pred, t, sample, generator=generator).prev_sample
        # We finally convert it to an image...
        sample = 0.5 * (sample + 1).squeeze()
        sample = sample.permute(1, 2, 0).cpu().numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)


class SemanticOnlyLDMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation conditionned by a Semantic Mask

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image.
        scheduler ([`ScoreSdeVeScheduler`]):
            A `ScoreSdeVeScheduler` to be used in combination with `unet` to denoise the encoded image.
    """
    unet: UNet2DSISModel
    scheduler: DDPMScheduler
    vae: AutoencoderKL

    def __init__(self, unet: UNet2DSISModel, scheduler: DDPMScheduler, vae: AutoencoderKL):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        segmap: torch.Tensor = None,
        num_inference_steps: int = 2000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        guidance_scale: float = 7.5,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            guidance_scale (`float`):
                The guidance scale for the image generation.
                default = 7.5,
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        img_size = self.unet.config.img_size
        img_depth = self.unet.config.in_channels
        shape = (batch_size, img_depth, img_size, img_size)

        # We reshape the segmentation map to match the diffusion size.
        while len(segmap.shape) < 4:
            segmap = segmap.unsqueeze(1)
        segmap = nn.UpsamplingNearest2d(size=shape[-2:])(segmap)
        model = self.unet

        sample = randn_tensor(shape, generator=generator, device=generator.device) * self.scheduler.init_noise_sigma
        sample = sample.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # prediction step (with segmentation map as input)
            model_input = torch.cat([sample] * 2)
            # Like in the documentation example, we scale model_input
            # https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#denoise-the-image
            model_input = self.scheduler.scale_model_input(model_input, timestep=t)
            model_cond = torch.cat([segmap, torch.zeros_like(segmap)])
            with torch.no_grad():
                noise_pred = model(model_input, t, model_cond).sample
            noise_pred_cond, noise_pred_ucond = noise_pred.chunk(2)
            noise_pred = noise_pred_ucond + guidance_scale * (noise_pred_cond - noise_pred_ucond)
            # We finally use this noise to sample
            sample = self.scheduler.step(noise_pred, t, sample, generator=generator).prev_sample
        # If we are in the case of a LDM
        sample = self.vae.decode(sample).sample
        # We finally convert it to an image...
        sample = 0.5 * (sample + 1).squeeze()
        sample = sample.permute(1, 2, 0).cpu().numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)
