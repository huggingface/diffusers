# Copyright 2025 Nucleus-Image Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable

import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLQwenImage, NucleusMoEImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import NucleusMoEImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are an image generation assistant. Follow the user's prompt literally. Pay careful attention to spatial layout: objects described as on the left must appear on the left, on the right on the right. Match exact object counts and assign colors to the correct objects."

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import NucleusMoEImagePipeline

        >>> pipe = NucleusMoEImagePipeline.from_pretrained("NucleusAI/NucleusMoE-Image", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt, num_inference_steps=50).images[0]
        >>> image.save("nucleus_moe.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class NucleusMoEImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using NucleusMoE.

    This pipeline uses a single-stream DiT with Mixture-of-Experts feed-forward layers, cross-attention to a Qwen3-VL
    text encoder, and a flow-matching Euler discrete scheduler.

    Args:
        transformer ([`NucleusMoEImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLQwenImage`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            Text encoder for computing prompt embeddings.
        processor ([`Qwen3VLProcessor`]):
            Processor for tokenizing text inputs.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        transformer: NucleusMoEImageTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
        )
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.default_sample_size = 128
        self.default_max_sequence_length = 1024
        self.default_return_index = -8

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode_prompt(
        self,
        prompt: str | list[str] = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int | None = None,
        return_index: int | None = None,
    ):
        r"""
        Encode text prompt(s) into embeddings using the Qwen3-VL text encoder.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to encode.
            device (`torch.device`, *optional*):
                Torch device for the resulting tensors.
            num_images_per_prompt (`int`, defaults to 1):
                Number of images to generate per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Skips encoding when provided.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for pre-generated embeddings.
            max_sequence_length (`int`, defaults to 1024):
                Maximum token length for the encoded prompt.
        """
        device = device or self._execution_device
        return_index = return_index or self.default_return_index

        if prompt_embeds is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            formatted = [self._format_prompt(p) for p in prompt]

            inputs = self.processor(
                text=formatted,
                padding="longest",
                pad_to_multiple_of=8,
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device=device)

            prompt_embeds_mask = inputs.attention_mask

            outputs = self.text_encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.hidden_states[return_index]
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(device=device)
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.to(device=device)

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        return_index=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both undefined.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )

        if return_index is not None and abs(return_index) >= self.text_encoder.config.text_config.num_hidden_layers:
            raise ValueError(
                f"absolute value of `return_index` cannot be >= {self.text_encoder.config.text_config.num_hidden_layers} "
                f"but is {abs(return_index)}"
            )

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, patch_size):
        latents = latents.view(
            batch_size, num_channels_latents, height // patch_size, patch_size, width // patch_size, patch_size
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // patch_size) * (width // patch_size), num_channels_latents * patch_size * patch_size
        )
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, patch_size, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = patch_size * (int(height) // (vae_scale_factor * patch_size))
        width = patch_size * (int(width) // (vae_scale_factor * patch_size))
        latents = latents.view(
            batch_size,
            height // patch_size,
            width // patch_size,
            channels // (patch_size * patch_size),
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (patch_size * patch_size), 1, height, width)
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        patch_size,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = patch_size * (int(height) // (self.vae_scale_factor * patch_size))
        width = patch_size * (int(width) // (self.vae_scale_factor * patch_size))
        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width, patch_size)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        guidance_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        return_index: int | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, an empty string is used when
                `true_cfg_scale > 1`.
            true_cfg_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Values greater than 1 enable CFG.
            height (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            sigmas (`list[float]`, *optional*):
                Custom sigmas for the denoising schedule. If not defined, a linear schedule is used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of torch generators to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents to be used as inputs for image generation.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `"pil"`, `"np"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`NucleusMoEImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Kwargs passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                Tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, defaults to 512):
                Maximum sequence length for the text prompt.

        Examples:

        Returns:
            [`NucleusMoEImagePipelineOutput`] or `tuple`:
                [`NucleusMoEImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple` where the first element
                is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        max_sequence_length = max_sequence_length or self.default_max_sequence_length

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            return_index=return_index,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_cfg = guidance_scale > 1

        if do_cfg and not has_neg_prompt:
            negative_prompt = [""] * batch_size

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            return_index=return_index,
        )
        if do_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                return_index=return_index,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        patch_size = self.transformer.config.patch_size

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            patch_size,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        img_shapes = [
            (1, height // self.vae_scale_factor // patch_size, width // self.vae_scale_factor // patch_size)
        ] * (batch_size * num_images_per_prompt)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        self.scheduler.set_begin_index(0)

        if self.transformer.is_cache_enabled:
            self.transformer._reset_stateful_cache()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / self.scheduler.config.num_train_timesteps,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    attention_kwargs=self._attention_kwargs,
                    return_dict=False,
                )[0]

                if do_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / self.scheduler.config.num_train_timesteps,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        img_shapes=img_shapes,
                        attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]

                    comb_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                noise_pred = -noise_pred

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, patch_size, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return NucleusMoEImagePipelineOutput(images=image)
