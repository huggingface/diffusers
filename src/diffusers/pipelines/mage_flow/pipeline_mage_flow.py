# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from ...image_processor import VaeImageProcessor
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import MageFlowPipelineOutput


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
        >>> from diffusers import MageFlowPipeline

        >>> pipe = MageFlowPipeline.from_pretrained("microsoft/Mage-Flow-4B", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt, num_inference_steps=30, guidance_scale=5.0).images[0]
        >>> image.save("mage_flow.png")
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


class MageFlowPipeline(DiffusionPipeline):
    r"""
    The Mage-Flow pipeline for text-to-image generation.

    Args:
        transformer ([`MageFlowTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderMageVAE`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            Qwen3-VL text encoder for producing text conditioning embeddings.
        tokenizer (`AutoTokenizer`):
            Tokenizer for the Qwen3-VL text encoder.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        transformer,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 16  # MageVAE downsample factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 2048
        self.default_sample_size = 64  # 1024 / 16 = 64
        # ChatML prompt template (same as QwenImage)
        self.prompt_template = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_start_idx = 34  # number of system-prompt tokens to skip

    def _get_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        if self.text_encoder is None:
            raise ValueError(
                "Text encoder is not available. Please provide `prompt_embeds` directly "
                "when the pipeline is initialized without a text encoder."
            )
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template
        drop_idx = self.prompt_template_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoder_out = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_out.hidden_states[-1]

        # Extract valid tokens per sample, drop system prompt prefix, then re-pad to uniform length.
        bool_mask = txt_tokens.attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_hidden_states = torch.split(selected, valid_lengths.tolist(), dim=0)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_embeds_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 2048,
    ):
        r"""
        Encode the text prompt into embeddings for the transformer.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt to be encoded.
            device (`torch.device`):
                Torch device.
            num_images_per_prompt (`int`):
                Number of images that should be generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for `prompt_embeds`.
            max_sequence_length (`int`):
                Maximum sequence length for the text embeddings.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_prompt_embeds(prompt, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

            if prompt_embeds_mask.all():
                prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}. "
                "Dimensions will be resized accordingly"
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
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            logger.warning(
                "`prompt_embeds` is provided and `prompt_embeds_mask` is not provided, so the model will treat all"
                " prompt tokens as valid. If `prompt_embeds` contains padding, you should provide the padding mask as"
                " `prompt_embeds_mask`. Make sure to generate `prompt_embeds_mask` from the same text encoder that was"
                " used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            logger.warning(
                "`negative_prompt_embeds` is provided and `negative_prompt_embeds_mask` is not provided, so the model"
                " will treat all negative prompt tokens as valid. If `negative_prompt_embeds` contains padding, you"
                " should provide the padding mask as `negative_prompt_embeds_mask`. Make sure to generate"
                " `negative_prompt_embeds_mask` from the same text encoder that was used to generate"
                " `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 2048:
            raise ValueError(f"`max_sequence_length` cannot be greater than 2048 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # MageVAE: 16x downsample, no patch packing
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # Flatten to sequence: [B, C, H, W] -> [B, H*W, C]
        latents = latents.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels_latents)

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
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 2048,
        sigmas: list[float] | None = None,
    ) -> MageFlowPipelineOutput | tuple:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                not greater than `1`).
            height (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Classifier-free guidance scale. Enabled by setting `guidance_scale > 1`. Higher guidance scale
                encourages images closely linked to the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for `negative_prompt_embeds`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.mage_flow.MageFlowPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to 2048):
                Maximum sequence length to use with the `prompt`.
            sigmas (`list[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.

        Examples:

        Returns:
            [`~pipelines.mage_flow.MageFlowPipelineOutput`] or `tuple`:
            [`~pipelines.mage_flow.MageFlowPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None and self.text_encoder is None:
                # text_encoder unavailable and no negative_prompt_embeds provided, skip CFG
                do_classifier_free_guidance = False
            else:
                negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                    prompt=negative_prompt if negative_prompt is not None else [""] * batch_size,
                    prompt_embeds=negative_prompt_embeds,
                    prompt_embeds_mask=negative_prompt_embeds_mask,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )

        # 4. Prepare latent variables
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

        # 5. Prepare image position ids for RoPE
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        img_ids = self._prepare_latent_image_ids(latent_h, latent_w, device, prompt_embeds.dtype)

        # 6. Prepare timesteps
        # Mage-Flow uses base_sigmas = linspace(1, 1/N, N) which differs from the scheduler's
        # default sigma computation. The scheduler's shift=6.0 is applied on top of these.
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # 7. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if do_classifier_free_guidance:
                    noise_pred_cond = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep / 1000,
                        img_ids=img_ids,
                        latent_height=latent_h,
                        latent_width=latent_w,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=timestep / 1000,
                        img_ids=img_ids,
                        latent_height=latent_h,
                        latent_width=latent_w,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep / 1000,
                        img_ids=img_ids,
                        latent_height=latent_h,
                        latent_width=latent_w,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 8. VAE decode
        if output_type == "latent":
            image = latents
        else:
            # Unflatten: [B, H*W, C] -> [B, C, H, W]
            latents = latents.reshape(batch_size * num_images_per_prompt, latent_h, latent_w, num_channels_latents)
            latents = latents.permute(0, 3, 1, 2)
            latents = latents.to(self.vae.dtype)
            image = self.vae(latents, return_dict=False)[0]
            image = image.clamp(-1, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return MageFlowPipelineOutput(images=image)
