# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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

# Author: lucasruan1618

from typing import Any, Callable

import torch

from ...image_processor import PipelineImageInput
from ...utils import replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_ideogram4 import (
    PROMPT_UPSAMPLE_TEMPERATURE,
    Ideogram4Pipeline,
    _expand_tensor_to_effective_batch,
    _logit_normal_sigmas,
    _resolution_aware_mu,
)
from .pipeline_output import Ideogram4PipelineOutput


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import os
        >>> import torch
        >>> from diffusers import Ideogram4Img2ImgPipeline, Ideogram4PromptEnhancerHead
        >>> from diffusers.utils import load_image

        >>> prompt_enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(
        ...     "diffusers/qwen3-vl-8b-instruct-lm-head", torch_dtype=torch.bfloat16
        ... )
        >>> pipe = Ideogram4Img2ImgPipeline.from_pretrained(
        ...     "ideogram-ai/ideogram-4-nf4-diffusers",
        ...     prompt_enhancer_head=prompt_enhancer_head,
        ...     torch_dtype=torch.bfloat16,
        ...     token=os.environ["HF_TOKEN"],
        ... ).to("cuda")
        >>> init_image = load_image(
        ...     "https://github.com/lucasruan1618/Image_storage/blob/main/Input/cute_cat.png?raw=true"
        ... ).convert("RGB")
        >>> text_prompt = "wizard dog, Gandalf-inspired, Lord of the Rings aesthetic, majestic yet cute, Studio Ghibli style"

        >>> image = pipe(
        ...     prompt=text_prompt,
        ...     image=init_image,
        ...     height=init_image.height,
        ...     width=init_image.width,
        ...     strength=0.8,
        ...     prompt_upsampling=True,
        ...     generator=torch.Generator(device="cuda").manual_seed(42),
        ... ).images[0]
        >>> image.save("ideogram4_img2img_0.8.png")
        ```
"""


class Ideogram4Img2ImgPipeline(Ideogram4Pipeline):
    r"""Image-to-image pipeline for Ideogram4."""

    def get_timesteps(self, num_inference_steps: int, strength: float) -> tuple[torch.Tensor, int, int]:
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        begin_index = t_start * self.scheduler.order
        timesteps = self.scheduler.timesteps[begin_index:]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(begin_index)
        return timesteps, num_inference_steps - t_start, t_start

    def _encode_vae_image(
        self, image: torch.Tensor, generator: torch.Generator | list[torch.Generator] | None
    ) -> torch.Tensor:
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(image.shape[0])
            ]
            return torch.cat(image_latents)
        return self.vae.encode(image).latent_dist.sample(generator)

    def prepare_latents(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        num_image_tokens: int,
        latent_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effective_batch_size = batch_size * num_images_per_prompt
        shape = (effective_batch_size, num_image_tokens, latent_dim)

        if isinstance(generator, list) and len(generator) != effective_batch_size:
            raise ValueError(
                f"You passed {len(generator)} generators, but the effective batch size is {effective_batch_size}."
            )
        if latents is not None:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            return latents.to(device=device, dtype=dtype)

        if image.shape[0] not in (1, batch_size, effective_batch_size):
            raise ValueError(
                f"`image` batch size must be 1, batch size ({batch_size}), or effective batch size "
                f"({effective_batch_size}), but got {image.shape[0]}."
            )
        image = image.to(device=device, dtype=self.vae.dtype)
        image_generator = generator[: image.shape[0]] if isinstance(generator, list) else generator
        image_latents = self._encode_vae_image(image, image_generator).float()

        patch = self.patch_size
        latent_height, latent_width = image_latents.shape[-2:]
        if latent_height % patch != 0 or latent_width % patch != 0:
            raise ValueError(
                f"Encoded image dimensions ({latent_height}, {latent_width}) must be divisible by patch size {patch}."
            )

        grid_h, grid_w = latent_height // patch, latent_width // patch
        image_latents = image_latents.view(
            image_latents.shape[0], image_latents.shape[1], grid_h, patch, grid_w, patch
        )
        image_latents = image_latents.permute(0, 2, 4, 3, 5, 1).reshape(image_latents.shape[0], -1, latent_dim)

        bn_mean = self.vae.bn.running_mean.view(1, 1, -1).to(device=device, dtype=image_latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps).view(1, 1, -1)
        image_latents = (image_latents - bn_mean) / bn_std.to(device=device, dtype=image_latents.dtype)
        image_latents = _expand_tensor_to_effective_batch(
            image_latents, batch_size, num_images_per_prompt, tensor_name="image"
        )
        if image_latents.shape != shape:
            raise ValueError(f"Unexpected encoded image shape, got {image_latents.shape}, expected {shape}")

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return self.scheduler.scale_noise(image_latents.to(dtype), timestep, noise)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: PipelineImageInput | None = None,
        strength: float = 0.6,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 48,
        guidance_scale: float | None = None,
        guidance_schedule: list[float] | torch.Tensor | None = (7.0,) * 45 + (3.0,) * 3,
        mu: float = 0.0,
        std: float = 1.5,
        prompt_upsampling: bool = False,
        prompt_upsampling_temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
        max_sequence_length: int = 2048,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[["Ideogram4Img2ImgPipeline", int, int, dict[str, Any]], dict[str, Any]]
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> Ideogram4PipelineOutput | tuple[Any]:
        r"""
        Run image-to-image generation.

        Args:
            prompt (`str` or `list[str]`):
                Prompt(s) to guide image generation.
            image (`PipelineImageInput`):
                Image or batch of images used to initialize generation.
            strength (`float`, *optional*, defaults to 0.6):
                Amount of transformation applied to `image`. Must be in `[0, 1]`.
            height (`int`, *optional*):
                Output height. By default, uses the input image height.
            width (`int`, *optional*):
                Output width. By default, uses the input image width.
            num_inference_steps (`int`, *optional*, defaults to 48):
                Number of flow-matching steps before applying `strength`.
            guidance_scale (`float`, *optional*):
                Constant guidance scale. Mutually exclusive with `guidance_schedule`.
            guidance_schedule (`list[float]` or `torch.Tensor`, *optional*):
                Per-step guidance schedule for the full denoising schedule.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) used for VAE sampling and noise generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.

        Examples:

        Returns:
            [`~pipelines.ideogram4.Ideogram4PipelineOutput`] or `tuple`.
        """
        if image is None:
            raise ValueError("`image` must be provided.")
        if strength < 0 or strength > 1:
            raise ValueError(f"`strength` must be in [0.0, 1.0], but got {strength}.")

        init_image = self.image_processor.preprocess(image, height=height, width=width).float()
        height, width = init_image.shape[-2:]
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        effective_batch_size = batch_size * num_images_per_prompt
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._interrupt = False

        if prompt_upsampling:
            prompt = self.upsample_prompt(
                prompt,
                height=height,
                width=width,
                temperature=prompt_upsampling_temperature,
                max_new_tokens=max_sequence_length,
                generator=generator,
                device=device,
            )

        grid_h = height // (self.vae_scale_factor * self.patch_size)
        grid_w = width // (self.vae_scale_factor * self.patch_size)
        num_image_tokens = grid_h * grid_w
        llm_features, position_ids, segment_ids, indicator = self.encode_prompt(
            prompt, grid_h, grid_w, max_sequence_length, device
        )
        llm_features = _expand_tensor_to_effective_batch(llm_features, batch_size, num_images_per_prompt)
        position_ids = _expand_tensor_to_effective_batch(position_ids, batch_size, num_images_per_prompt)
        segment_ids = _expand_tensor_to_effective_batch(segment_ids, batch_size, num_images_per_prompt)
        indicator = _expand_tensor_to_effective_batch(indicator, batch_size, num_images_per_prompt)

        neg_llm_features = torch.zeros(
            effective_batch_size,
            num_image_tokens,
            llm_features.shape[-1],
            dtype=llm_features.dtype,
            device=device,
        )
        neg_position_ids = position_ids[:, max_sequence_length:]
        neg_segment_ids = segment_ids[:, max_sequence_length:]
        neg_indicator = indicator[:, max_sequence_length:]

        schedule_mu = _resolution_aware_mu(height, width, mu)
        sigmas = _logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        timesteps, adjusted_steps, t_start = self.get_timesteps(num_inference_steps, strength)
        if adjusted_steps < 1:
            raise ValueError(f"After applying strength={strength}, the number of denoising steps is {adjusted_steps}.")
        self._num_timesteps = len(timesteps)

        if guidance_scale is not None:
            guidance_schedule = [float(guidance_scale)] * adjusted_steps
        else:
            guidance_schedule = list(guidance_schedule)[t_start:]
        guidance_weights = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)

        latent_dim = self.transformer.config.in_channels
        latents = self.prepare_latents(
            image=init_image,
            timestep=timesteps[:1].repeat(effective_batch_size),
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_image_tokens=num_image_tokens,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        text_z_padding = torch.zeros(
            effective_batch_size, max_sequence_length, latent_dim, dtype=torch.float32, device=device
        )
        llm_features = llm_features.to(self.transformer.dtype)
        neg_llm_features = neg_llm_features.to(self.unconditional_transformer.dtype)

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        with self.progress_bar(total=adjusted_steps) as progress_bar:
            for i, timestep in enumerate(timesteps):
                if self.interrupt:
                    continue

                model_timestep = 1.0 - timestep.float() / num_train_timesteps
                model_timestep = model_timestep.expand(effective_batch_size).to(self.transformer.dtype)
                pos_out = self.transformer(
                    hidden_states=torch.cat([text_z_padding, latents], dim=1).to(self.transformer.dtype),
                    timestep=model_timestep,
                    encoder_hidden_states=llm_features,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    return_dict=False,
                )[0]
                pos_velocity = pos_out[:, max_sequence_length:].float()
                neg_velocity = self.unconditional_transformer(
                    hidden_states=latents.to(self.unconditional_transformer.dtype),
                    timestep=model_timestep,
                    encoder_hidden_states=neg_llm_features,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                    return_dict=False,
                )[0].float()

                self._guidance_scale = guidance_schedule[i]
                velocity = guidance_weights[i] * pos_velocity + (1.0 - guidance_weights[i]) * neg_velocity
                latents = self.scheduler.step(-velocity, timestep, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {name: locals()[name] for name in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, timestep, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            bn_mean = self.vae.bn.running_mean.view(1, 1, -1).to(device=latents.device, dtype=latents.dtype)
            bn_std = torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps).view(1, 1, -1)
            z = latents * bn_std.to(device=latents.device, dtype=latents.dtype) + bn_mean
            patch = self.patch_size
            ae_channels = z.shape[-1] // (patch * patch)
            z = z.view(effective_batch_size, grid_h, grid_w, patch, patch, ae_channels)
            z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
            z = z.view(effective_batch_size, ae_channels, grid_h * patch, grid_w * patch)
            decoded = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(decoded.float(), output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Ideogram4PipelineOutput(images=image)
