# Copyright (c) 2021 OpenAI
# MIT License

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

from typing import List, Optional, Tuple, Union

import torch

from ...models import AutoencoderKL, DiT
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDIMScheduler


class DiTPipeline(DiffusionPipeline):
    r"""
    This pipeline inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        dit ([`DiT`]):
            Class conditioned Transformer in Diffusion model to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `dit` to denoise the encoded image latents.
    """

    def __init__(self, dit: DiT, vae: AutoencoderKL, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(dit=dit, vae=vae, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 250,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of imagenet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Scale of the guidance signal.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.
        """

        batch_size = len(class_labels)
        latent_size = self.dit.config.input_size
        latent_channels = self.dit.config.in_channels

        latents = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=self.device)
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=self.device)
        class_null = torch.tensor([1000] * batch_size, device=self.device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.dit(latent_model_input, t, class_labels_input)

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :3], noise_pred[:, 3:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.dit.config.learn_sigma:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(
                model_output, t, latent_model_input, generator=generator
            ).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)

        latents = 1 / 0.18215 * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
