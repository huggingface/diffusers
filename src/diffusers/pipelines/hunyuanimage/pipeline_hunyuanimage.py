# Copyright 2025 Tencent Hunyuan Team and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLHunyuanImage, HunyuanImage2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import HunyuanImagePipeline

        >>> pipe = HunyuanImagePipeline.from_pretrained(
        ...     "tencent/HunyuanImage-2.1", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> prompt = "A cute cartoon penguin wearing a red scarf"
        >>> image = pipe(
        ...     prompt=prompt,
        ...     height=2048,
        ...     width=2048,
        ...     num_inference_steps=50,
        ...     guidance_scale=3.5,
        ... ).images[0]
        >>> image.save("penguin.png")
        ```
"""


# Resolutions supported by the model
STANDARD_RATIO = [1.0, 4.0/3.0, 3.0/4.0, 16.0/9.0, 9.0/16.0]
STANDARD_SHAPE = [
    [(2048, 2048)],  # 1:1
    [(2304, 1792)],  # 4:3
    [(1792, 2304)],  # 3:4
    [(2560, 1536)],  # 16:9
    [(1536, 2560)],  # 9:16
]


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Retrieve timesteps for the scheduler.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class HunyuanImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using HunyuanImage 2.1.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLHunyuanImage`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
            HunyuanImage uses a custom VAE with 32x spatial compression.
        text_encoder ([`T5EncoderModel`]):
            Text encoder model to encode prompts.
        tokenizer ([`T5Tokenizer`]):
            Tokenizer for the text encoder.
        transformer ([`HunyuanImage2DModel`]):
            The HunyuanImage transformer model.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLHunyuanImage,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        transformer: HunyuanImage2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae.config, "block_out_channels") else 32
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                The prompt to encode.
            device (`torch.device`):
                The device to use.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate per prompt.
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier-free guidance.
            negative_prompt (`str` or `List[str]`, *optional*):
                The negative prompt to use for classifier-free guidance.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            max_sequence_length (`int`, *optional*, defaults to 256):
                Maximum sequence length for the text encoder.
        """
        if prompt_embeds is None:
            # Convert prompt to list
            if isinstance(prompt, str):
                prompt = [prompt]
            batch_size = len(prompt)

            # Tokenize
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            # Encode
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )[0]

        # Duplicate for num_images_per_prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Get attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(1, num_images_per_prompt)
            attention_mask = attention_mask.view(bs_embed * num_images_per_prompt, -1)
        else:
            attention_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.long)

        # Handle negative prompt for CFG
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            # Tokenize negative prompt
            uncond_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids
            negative_attention_mask = uncond_input.attention_mask

            # Encode negative prompt
            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids.to(device),
                attention_mask=negative_attention_mask.to(device),
            )[0]

            # Duplicate for num_images_per_prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            
            negative_attention_mask = negative_attention_mask.repeat(1, num_images_per_prompt)
            negative_attention_mask = negative_attention_mask.view(bs_embed * num_images_per_prompt, -1)

        # For classifier-free guidance, concatenate unconditional and conditional
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([negative_attention_mask, attention_mask])

        return prompt_embeds, attention_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        """Prepare initial latents for the diffusion process."""
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale the initial noise by the scheduler's init noise sigma
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 2048,
        width: int = 2048,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        r"""
        Generate images from text prompts using HunyuanImage 2.1.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to 2048):
                The height in pixels of the generated image. Should be 2048 for best results.
            width (`int`, *optional*, defaults to 2048):
                The width in pixels of the generated image. Should be 2048 for best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More steps usually lead to higher quality images.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for denoising.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for denoising.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Higher guidance scale encourages images closely linked to `prompt`, at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt(s) to guide what to not include in image generation.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A torch.Generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable[[int, int, Dict], None]`, *optional*):
                A callback function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                List of tensor inputs to pass to the callback function.
            max_sequence_length (`int`, *optional*, defaults to 256):
                Maximum sequence length for the text encoder.

        Returns:
            [`ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`ImagePipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images.
        """
        # 0. Default height and width to unet config
        height = height or 2048
        width = width or 2048

        # 1. Check inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        self._guidance_scale = guidance_scale
        
        # 2. Encode prompt
        prompt_embeds, attention_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        self._num_timesteps = len(timesteps)

        # 4. Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents if doing classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # Broadcast timestep to batch dimension
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=attention_mask,
                    return_dict=False,
                )[0]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Call callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 6. Decode latents
        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
