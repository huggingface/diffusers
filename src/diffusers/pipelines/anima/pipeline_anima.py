# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from typing import Callable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, T5TokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...loaders import AnimaLoraLoaderMixin
from ...models import AutoencoderKLQwenImage, CosmosTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .modeling_anima import AnimaTextConditioner


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import AnimaPipeline

        >>> pipe = AnimaPipeline.from_pretrained("path/to/anima-diffusers", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image = pipe("A cinematic portrait of a woman in a rain-soaked city street").images[0]
        >>> image.save("anima.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
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


class AnimaPipeline(DiffusionPipeline, AnimaLoraLoaderMixin):
    r"""
    Pipeline for text-to-image generation using Anima.

    Anima uses a Qwen3 text encoder, a T5-token LLM adapter, the Cosmos Predict2 DiT, and the Qwen-Image VAE.
    Supports loading LoRA weights with [`~loaders.AnimaLoraLoaderMixin.load_lora_weights`].

    Args:
        text_encoder (`~transformers.PreTrainedModel`):
            Qwen3 text encoder used to produce source hidden states for the Anima text conditioner.
        tokenizer (`~transformers.PreTrainedTokenizer`):
            Qwen tokenizer paired with `text_encoder`.
        t5_tokenizer (`~transformers.T5TokenizerFast`):
            T5 tokenizer used to produce target token ids for the Anima text conditioner.
        text_conditioner ([`AnimaTextConditioner`]):
            Adapter that maps Qwen3 hidden states and T5 token ids to Cosmos text embeddings.
        transformer ([`CosmosTransformer3DModel`]):
            Cosmos Predict2 transformer used to denoise image latents.
        vae ([`AutoencoderKLQwenImage`]):
            Qwen-Image VAE used to decode latents into images.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler used for denoising.
    """

    model_cpu_offload_seq = "text_encoder->text_conditioner->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        t5_tokenizer: T5TokenizerFast,
        text_conditioner: AnimaTextConditioner,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLQwenImage,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            t5_tokenizer=t5_tokenizer,
            text_conditioner=text_conditioner,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = 128

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = prompt_embeds * prompt_attention_mask.to(prompt_embeds).unsqueeze(-1)

        return prompt_embeds, prompt_attention_mask

    def _get_t5_prompt_ids(
        self,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.t5_tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or self._execution_device
        dtype = dtype or self.text_conditioner.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            qwen_prompt_embeds, qwen_attention_mask = self._get_qwen_prompt_embeds(
                prompt=prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )
            t5_input_ids, t5_attention_mask = self._get_t5_prompt_ids(
                prompt=prompt, max_sequence_length=max_sequence_length, device=device
            )
            prompt_embeds = self.text_conditioner(
                source_hidden_states=qwen_prompt_embeds,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attention_mask,
                source_attention_mask=qwen_attention_mask,
            )
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_qwen_prompt_embeds, negative_qwen_attention_mask = self._get_qwen_prompt_embeds(
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )
            negative_t5_input_ids, negative_t5_attention_mask = self._get_t5_prompt_ids(
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device
            )
            negative_prompt_embeds = self.text_conditioner(
                source_hidden_states=negative_qwen_prompt_embeds,
                target_input_ids=negative_t5_input_ids,
                target_attention_mask=negative_t5_attention_mask,
                source_attention_mask=negative_qwen_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and"
                f" {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if max_sequence_length is not None and max_sequence_length > 4096:
            raise ValueError(f"`max_sequence_length` cannot be greater than 4096 but is {max_sequence_length}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor + 1
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @property
    def guidance_scale(self):
        return self._guidance_scale

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
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, pass `prompt_embeds`.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide image generation. Used when `guidance_scale > 1`.
            height (`int`, *optional*, defaults to `1024`):
                Height in pixels of the generated image.
            width (`int`, *optional*, defaults to `1024`):
                Width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps.
            sigmas (`list[float]`, *optional*):
                Custom sigma schedule to use for schedulers that support `sigmas`.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated conditioned prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated conditioned negative prompt embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format, one of `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`~pipelines.ImagePipelineOutput`].
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                Function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                Tensor inputs available to `callback_on_step_end`.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length used by both text tokenizers.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                Generated images if `return_dict` is `True`; otherwise a tuple whose first item is the images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        num_frames = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.transformer.dtype,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        self.scheduler.set_begin_index(0)

        transformer_dtype = self.transformer.dtype
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(transformer_dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                latent_model_input = latents.to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    negative_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = negative_noise_pred + self.guidance_scale * (noise_pred - negative_noise_pred)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents[:, :, 0]
        else:
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

        return ImagePipelineOutput(images=image)
