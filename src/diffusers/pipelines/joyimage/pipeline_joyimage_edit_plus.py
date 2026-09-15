# Copyright 2025 The JoyImage Team and The HuggingFace Team. All rights reserved.
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

import torch
from PIL import Image
from transformers import (
    Qwen2Tokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models import AutoencoderKLWan
from ...models.transformers.transformer_joyimage_edit_plus import JoyImageEditPlusTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import JoyImageEditImageProcessor
from .pipeline_output import JoyImageEditPlusPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
Examples:
    ```python
    >>> import torch
    >>> from diffusers import JoyImageEditPlusPipeline
    >>> from diffusers.utils import load_image

    >>> model_id = "jdopensource/JoyAI-Image-Edit-Plus-Diffusers"
    >>> pipe = JoyImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> images = [
    ...     load_image("dog.png"),
    ...     load_image("person.png"),
    ... ]
    >>> output = pipe(
    ...     images=images,
    ...     prompt="Let the person lovingly play with the dog.",
    ...     height=1024,
    ...     width=1024,
    ...     num_inference_steps=30,
    ...     guidance_scale=4.0,
    ...     generator=torch.manual_seed(42),
    ... )
    >>> output.images[0].save("output.png")
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


class JoyImageEditPlusPipeline(DiffusionPipeline):
    r"""
    Diffusion pipeline for multi-image instruction-guided editing using JoyImage Edit Plus.

    Supports multiple reference images with different resolutions. Each reference image is independently VAE-encoded
    and patchified, then concatenated with the target noise patches for joint denoising.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            Multimodal text encoder for prompt encoding with inline image understanding.
        tokenizer ([`Qwen2Tokenizer`]):
            Tokenizer for text processing.
        transformer ([`JoyImageEditPlusTransformer3DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        processor ([`Qwen3VLProcessor`]):
            Processor for multimodal inputs (text + images).
        text_token_max_length (`int`, defaults to `2048`):
            Maximum token length for text encoding.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLWan,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: JoyImageEditPlusTransformer3DModel,
        processor: Qwen3VLProcessor,
        text_token_max_length: int = 2048,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            processor=processor,
        )

        self.text_token_max_length = text_token_max_length

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.image_processor = JoyImageEditImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.prompt_template_encode = {
            "multiple_images": (
                "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
                "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                "{}<|im_start|>assistant\n"
            ),
        }
        self.prompt_template_encode_start_idx = {
            "multiple_images": 34,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_last_decoder_hidden_states(self, forward_fn, **kwargs):
        """
        Run ``forward_fn(**kwargs)`` while capturing the **pre-norm** output of the last decoder layer via a forward
        hook.

        This model was trained on transformers 4.57, where ``Qwen3VLForConditionalGeneration``'s
        ``@check_model_inputs`` decorator monkey-patched each decoder layer to collect ``hidden_states``. Because
        ``Qwen3VLCausalLMOutputWithPast`` has no ``last_hidden_state`` field, ``tie_last_hidden_states`` had no effect
        and ``hidden_states[-1]`` was the **pre-norm** output of the last decoder layer.

        Starting from https://github.com/huggingface/transformers/pull/42609 the CausalLM forward explicitly returns
        ``hidden_states=outputs.hidden_states`` from the inner model. Combined with the subsequent
        ``@check_model_inputs`` → ``@capture_outputs`` migration (transformers 5.x), ``hidden_states`` is now captured
        at the ``Qwen3VLTextModel`` level where ``tie_last_hidden_states=True`` replaces ``hidden_states[-1]`` with the
        **post-norm** ``last_hidden_state``. The CausalLM simply passes this through, so ``hidden_states[-1]`` becomes
        post-norm – a ~10x scale difference (std ~2 vs ~21) that breaks inference.

        This helper bypasses both mechanisms by hooking the last decoder layer directly, returning the raw pre-norm
        output regardless of the transformers version.
        """
        captured = {}

        def _hook(_module, _input, output):
            captured["hidden_states"] = output[0] if isinstance(output, tuple) else output

        handle = self.text_encoder.model.language_model.layers[-1].register_forward_hook(_hook)
        try:
            forward_fn(**kwargs)
        finally:
            handle.remove()
        return captured["hidden_states"]

    def encode_prompt_multiple_images(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        images: list[Image.Image] | None = None,
        max_sequence_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with inline <image> tokens via the Qwen3-VL processor."""
        device = device or self._execution_device
        template = self.prompt_template_encode["multiple_images"]
        drop_idx = self.prompt_template_encode_start_idx["multiple_images"]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [p.replace("<image>\n", "<|vision_start|><|image_pad|><|vision_end|>") for p in prompt]
        prompt = [template.format(p) for p in prompt]

        inputs = self.processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        last_hidden_states = self._get_last_decoder_hidden_states(self.text_encoder, **inputs)

        prompt_embeds = last_hidden_states[:, drop_idx:]
        prompt_embeds_mask = inputs["attention_mask"][:, drop_idx:]

        if max_sequence_length is not None and prompt_embeds.shape[1] > max_sequence_length:
            prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
            prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]

        return prompt_embeds, prompt_embeds_mask

    def _pad_sequence(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = x.shape[1]
        if current_length >= target_length:
            return x[:, -target_length:]
        padding_length = target_length - current_length
        if x.ndim >= 3:
            padding = torch.zeros((x.shape[0], padding_length, *x.shape[2:]), dtype=x.dtype, device=x.device)
        else:
            padding = torch.zeros((x.shape[0], padding_length), dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=1)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        reference_images: list[list[Image.Image]] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[int, int, int]]]]:
        """Prepare 6D padded latent tensor with target noise + reference image latents.

        Args:
            latents: Optional pre-computed noise for the target slot. Shape ``(B, C, 1, H', W')`` where
                ``H'`` and ``W'`` are the latent-space dimensions. When ``None``, random noise is sampled.

        Returns:
            padded_latents: [B, max_patches, C, pt, ph, pw] target_mask: [B, max_patches] (True for target patches)
            shape_list: per-sample list of (t, h, w) tuples for each component
        """
        pt, ph, pw = self.transformer.config.patch_size

        all_patches = []
        all_target_masks = []
        all_shape_lists = []
        max_patches = 0

        for i in range(batch_size):
            sample_gen = generator[i] if isinstance(generator, list) else generator

            # Target noise
            t_target = 1
            h_target = int(height) // self.vae_scale_factor_spatial
            w_target = int(width) // self.vae_scale_factor_spatial
            if latents is None:
                noise_shape = (num_channels_latents, t_target, h_target, w_target)
                noise_block = randn_tensor(noise_shape, generator=sample_gen, device=device, dtype=dtype)
            else:
                noise_block = latents[i].to(device=device, dtype=dtype)

            sample_items = [noise_block]

            # Reference images
            if reference_images is not None and reference_images[i]:
                for ref_img_pil in reference_images[i]:
                    ref_tensor = self.image_processor.preprocess(ref_img_pil).to(device=device, dtype=dtype)
                    ref_tensor = ref_tensor.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

                    ref_latent = self.vae.encode(ref_tensor.to(self.vae.dtype)).latent_dist.mode()
                    ref_latent = ref_latent.to(dtype)
                    latents_mean = (
                        torch.tensor(self.vae.config.latents_mean)
                        .view(1, -1, 1, 1, 1)
                        .to(ref_latent.device, ref_latent.dtype)
                    )
                    latents_std = (
                        torch.tensor(self.vae.config.latents_std)
                        .view(1, -1, 1, 1, 1)
                        .to(ref_latent.device, ref_latent.dtype)
                    )
                    ref_latent = (ref_latent - latents_mean) / latents_std
                    ref_latent = ref_latent.squeeze(0)  # [C, 1, H', W']
                    sample_items.append(ref_latent)

            # Patchify each item and build shape_list
            sample_patches = []
            sample_masks = []
            sample_shapes = []

            for j, item in enumerate(sample_items):
                c, t, h, w = item.shape
                l_t, l_h, l_w = t // pt, h // ph, w // pw
                sample_shapes.append((l_t, l_h, l_w))

                patches = item.reshape(c, l_t, pt, l_h, ph, l_w, pw)
                patches = patches.permute(1, 3, 5, 0, 2, 4, 6).reshape(-1, c, pt, ph, pw)
                sample_patches.append(patches)
                sample_masks.append(torch.full((patches.shape[0],), j == 0, device=device, dtype=torch.bool))

            combined_patches = torch.cat(sample_patches, dim=0)
            combined_masks = torch.cat(sample_masks, dim=0)

            all_patches.append(combined_patches)
            all_target_masks.append(combined_masks)
            all_shape_lists.append(sample_shapes)
            max_patches = max(max_patches, combined_patches.shape[0])

        # Pad to uniform size
        padded_latents = torch.zeros(
            (batch_size, max_patches, num_channels_latents, pt, ph, pw), device=device, dtype=dtype
        )
        target_mask = torch.zeros((batch_size, max_patches), device=device, dtype=torch.bool)

        for i in range(batch_size):
            n = all_patches[i].shape[0]
            padded_latents[i, :n] = all_patches[i]
            target_mask[i, :n] = all_target_masks[i]

        return padded_latents, target_mask, all_shape_lists

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height is not None and height % self.vae_scale_factor_spatial != 0:
            raise ValueError(f"`height` must be divisible by {self.vae_scale_factor_spatial} but is {height}.")
        if width is not None and width % self.vae_scale_factor_spatial != 0:
            raise ValueError(f"`width` must be divisible by {self.vae_scale_factor_spatial} but is {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found"
                f" {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        images: list[Image.Image] | list[list[Image.Image]] | None = None,
        prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        timesteps: list[int] = None,
        sigmas: list[float] = None,
        guidance_scale: float = 4.0,
        negative_prompt: str | list[str] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 4096,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            images (`list[Image.Image]` or `list[list[Image.Image]]`, *optional*):
                Reference images for editing. Each image can have a different resolution. If a flat list is provided,
                it is treated as one sample with multiple references.
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`
                instead.
            height (`int`, *optional*):
                The height in pixels of the generated image. If `None`, determined from the last reference image.
            width (`int`, *optional*):
                The width in pixels of the generated image. If `None`, determined from the last reference image.
            num_inference_steps (`int`, *optional*, defaults to `30`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`list[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spacing is used.
            sigmas (`list[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to `4.0`):
                Classifier-free guidance scale. Higher values encourage the model to generate images more aligned with
                the `prompt` at the expense of lower image quality.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, a blank prompt is used for
                classifier-free guidance.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents to be used as inputs for image generation.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.ndarray`), `"pt"` (`torch.Tensor`), or `"latent"` for raw latent output.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`JoyImageEditPlusPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step with arguments: the pipeline, step index, timestep,
                and a dict of callback tensor inputs.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*, defaults to `["latents"]`):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, *optional*, defaults to `4096`):
                Maximum sequence length for the text encoder.

        Examples:

        Returns:
            [`JoyImageEditPlusPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`JoyImageEditPlusPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list of generated images.
        """
        # Normalize images input to List[List[Image]]
        if images is not None:
            if isinstance(images[0], Image.Image):
                images = [images]  # single sample

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Determine output resolution from last reference image if not specified
        if height is None or width is None:
            if images is not None and len(images[0]) > 0:
                last_img = images[0][-1]
                height, width = self.image_processor.get_default_height_width(last_img)
            else:
                height = height or 1024
                width = width or 1024

        device = self._execution_device

        # Pre-process images: bucket-resize each reference image (matching original pipeline)
        if images is not None:
            processed_images = []
            for sample_imgs in images:
                processed_sample = []
                for img in sample_imgs:
                    ref_h, ref_w = self.image_processor.get_default_height_width(img)
                    resize_img = self.image_processor.resize_center_crop(img, (ref_h, ref_w))
                    processed_sample.append(resize_img)
                processed_images.append(processed_sample)
            images = processed_images

        # Construct prompts with <image> tokens
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if images is not None:
            formatted_prompts = []
            for i in range(batch_size):
                num_refs = len(images[i]) if i < len(images) else 0
                image_tags = "".join(["<image>\n" for _ in range(num_refs)])
                p = prompt[i] if i < len(prompt) else prompt[0]
                formatted_prompts.append(f"<|im_start|>user\n{image_tags}{p}<|im_end|>\n")
        else:
            formatted_prompts = [f"<|im_start|>user\n{p}<|im_end|>\n" for p in prompt]

        # Flatten all images for the processor
        flattened_images = None
        if images is not None:
            flattened_images = [img for sublist in images for img in sublist]

        # Encode prompt
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt_multiple_images(
                prompt=formatted_prompts,
                images=flattened_images,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # Encode negative prompt for CFG
        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                neg_prompts = []
                for i in range(batch_size):
                    num_refs = len(images[i]) if images is not None and i < len(images) else 0
                    image_tags = "".join(["<image>\n" for _ in range(num_refs)])
                    neg_prompts.append(f"<|im_start|>user\n{image_tags} <|im_end|>\n")
                negative_prompt = neg_prompts
            elif negative_prompt is not None and negative_prompt_embeds is None:
                neg_list = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                neg_prompts = []
                for i in range(batch_size):
                    num_refs = len(images[i]) if images is not None and i < len(images) else 0
                    image_tags = "".join(["<image>\n" for _ in range(num_refs)])
                    n = neg_list[i] if i < len(neg_list) else neg_list[0]
                    neg_prompts.append(f"<|im_start|>user\n{image_tags}{n}<|im_end|>\n")
                negative_prompt = neg_prompts

            if negative_prompt_embeds is None:
                neg_prompt_list = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt_multiple_images(
                    prompt=neg_prompt_list,
                    images=flattened_images,
                    device=device,
                    max_sequence_length=max_sequence_length,
                )

            # Pad and concatenate [negative, positive]
            max_seq_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
            prompt_embeds = torch.cat(
                [
                    self._pad_sequence(negative_prompt_embeds, max_seq_len),
                    self._pad_sequence(prompt_embeds, max_seq_len),
                ]
            )
            if prompt_embeds_mask is not None and negative_prompt_embeds_mask is not None:
                prompt_embeds_mask = torch.cat(
                    [
                        self._pad_sequence(negative_prompt_embeds_mask, max_seq_len),
                        self._pad_sequence(prompt_embeds_mask, max_seq_len),
                    ]
                )

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # Prepare latents (patchified)
        num_channels_latents = self.transformer.config.in_channels
        latents, target_mask, shape_list = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            reference_images=images,
            latents=latents,
        )

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        clean_reference_backup = latents.clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Restore reference patches
                latents[~target_mask] = clean_reference_backup[~target_mask]

                model_input = latents

                # CFG expansion
                if self.do_classifier_free_guidance:
                    model_input_cfg = torch.cat([model_input] * 2)
                    t_expand = t.repeat(model_input_cfg.shape[0])
                    cfg_shape_list = shape_list * 2
                else:
                    model_input_cfg = model_input
                    t_expand = t.repeat(batch_size)
                    cfg_shape_list = shape_list

                # Transformer forward
                noise_pred = self.transformer(
                    hidden_states=model_input_cfg,
                    timestep=t_expand,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    shape_list=cfg_shape_list,
                    return_dict=False,
                )[0]

                # CFG combination with norm rescaling
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    comb_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    cond_norm = torch.norm(noise_pred_text, dim=2, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=2, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm.clamp_min(1e-6))

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(
                    dtype=prompt_embeds.dtype
                )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        # Post-processing: decode target latents
        if output_type != "latent":
            latents[~target_mask] = clean_reference_backup[~target_mask]
            pt, ph, pw = self.transformer.config.patch_size

            image_list = []
            for b_idx in range(batch_size):
                l_t, l_h, l_w = shape_list[b_idx][0]
                target_len = l_t * l_h * l_w

                target_patches = latents[b_idx, :target_len]
                c_lat = target_patches.shape[1]
                video_latent = target_patches.reshape(l_t, l_h, l_w, c_lat, pt, ph, pw)
                video_latent = video_latent.permute(3, 0, 4, 1, 5, 2, 6).reshape(
                    1, c_lat, l_t * pt, l_h * ph, l_w * pw
                )

                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, -1, 1, 1, 1)
                    .to(video_latent.device, video_latent.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, -1, 1, 1, 1)
                    .to(video_latent.device, video_latent.dtype)
                )
                video_latent = video_latent * latents_std + latents_mean

                sample_image = self.vae.decode(video_latent.to(self.vae.dtype), return_dict=False)[0]
                # [1, C, T=1, H, W] -> [C, H, W]
                sample_image = sample_image.float().squeeze(0).squeeze(1)
                image_list.append(sample_image)

            image = torch.stack(image_list)  # [B, C, H, W]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return JoyImageEditPlusPipelineOutput(images=image)
