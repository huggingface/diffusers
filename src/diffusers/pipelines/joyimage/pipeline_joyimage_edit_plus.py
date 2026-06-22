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
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from transformers import (
    Qwen2Tokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLWan
from ...models.transformers.transformer_joyimage_edit_plus import JoyImageEditPlusTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import JoyImageEditImageProcessor, find_best_bucket
from .pipeline_output import JoyImageEditPlusPipelineOutput


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


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        if "timesteps" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class JoyImageEditPlusPipeline(DiffusionPipeline):
    """Diffusion pipeline for multi-image editing using JoyImage Edit Plus.

    Supports multiple reference images with different resolutions. Each reference image is independently
    VAE-encoded and patchified, then concatenated with the target noise patches for joint denoising.

    Model offloading order: text_encoder -> transformer -> vae.
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
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.vae_image_processor = JoyImageEditImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
        )

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
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        images: Optional[List[Image.Image]] = None,
        max_sequence_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            padding = torch.zeros(
                (x.shape[0], padding_length, *x.shape[2:]), dtype=x.dtype, device=x.device
            )
        else:
            padding = torch.zeros((x.shape[0], padding_length), dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=1)

    def normalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
        else:
            latent = latent * self.vae.config.scaling_factor
        return latent

    def denormalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
            )
            latent = latent * latents_std + latents_mean
        else:
            latent = latent / self.vae.config.scaling_factor
        return latent

    def _resize_center_crop(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        w, h = img.size
        bh, bw = target_size
        scale = max(bh / h, bw / w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)
        img = img.resize((resize_w, resize_h), Image.LANCZOS)
        left = (resize_w - bw) // 2
        top = (resize_h - bh) // 2
        img = img.crop((left, top, left + bw, top + bh))
        return img

    def _get_bucket_size(self, img: Image.Image) -> Tuple[int, int]:
        return find_best_bucket(img.size[1], img.size[0], self.vae_image_processor.config.basesize)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        reference_images: Optional[List[List[Image.Image]]] = None,
        enable_denormalization: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int, int]]]]:
        """Prepare 6D padded latent tensor with target noise + reference image latents.

        Returns:
            padded_latents: [B, max_patches, C, pt, ph, pw]
            target_mask: [B, max_patches] (True for target patches)
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
            noise_shape = (num_channels_latents, t_target, h_target, w_target)
            noise_block = randn_tensor(noise_shape, generator=sample_gen, device=device, dtype=dtype)

            sample_items = [noise_block]

            # Reference images
            if reference_images is not None and reference_images[i]:
                for ref_img_pil in reference_images[i]:
                    ref_h, ref_w = self._get_bucket_size(ref_img_pil)
                    ref_img_pil = self._resize_center_crop(ref_img_pil, (ref_h, ref_w))

                    ref_tensor = torch.from_numpy(np.array(ref_img_pil.convert("RGB"))).to(device=device, dtype=dtype)
                    ref_tensor = (ref_tensor / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)

                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        ref_latent = self.vae.encode(ref_tensor.float()).latent_dist.mode()
                    ref_latent = ref_latent.to(dtype)
                    ref_latent = self.normalize_latents(ref_latent)
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

                patches = rearrange(item, "c (t pt) (h ph) (w pw) -> (t h w) c pt ph pw", pt=pt, ph=ph, pw=pw)
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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        images: List[Image.Image] | List[List[Image.Image]] | None = None,
        prompt: str | List[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 4096,
        enable_denormalization: bool = True,
    ):
        r"""
        Generate an edited image from multiple reference images and a text prompt.

        Args:
            images (`List[Image.Image]` or `List[List[Image.Image]]`):
                Reference images for editing. Each image can have a different resolution.
                If a flat list is provided, it's treated as one sample with multiple references.
            prompt (`str` or `List[str]`):
                Text prompt describing the desired edit.
            height (`int`, *optional*):
                Output height in pixels. If None, determined from the last reference image's bucket.
            width (`int`, *optional*):
                Output width in pixels. If None, determined from the last reference image's bucket.
            num_inference_steps (`int`, defaults to 30):
                Number of denoising steps.
            guidance_scale (`float`, defaults to 4.0):
                Classifier-free guidance scale.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt for CFG.
            generator (`torch.Generator`, *optional*):
                RNG generator for reproducibility.
            enable_denormalization (`bool`, defaults to True):
                Whether to denormalize latents before VAE decoding.

        Examples:

        Returns:
            [`JoyImageEditPlusPipelineOutput`] or `tuple`.
        """
        # Normalize images input to List[List[Image]]
        if images is not None:
            if isinstance(images[0], Image.Image):
                images = [images]  # single sample

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
                height, width = self._get_bucket_size(last_img)
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
                    ref_h, ref_w = self._get_bucket_size(img)
                    resize_img = self._resize_center_crop(img, (ref_h, ref_w))
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
            prompt_embeds = torch.cat([
                self._pad_sequence(negative_prompt_embeds, max_seq_len),
                self._pad_sequence(prompt_embeds, max_seq_len),
            ])
            if prompt_embeds_mask is not None and negative_prompt_embeds_mask is not None:
                prompt_embeds_mask = torch.cat([
                    self._pad_sequence(negative_prompt_embeds_mask, max_seq_len),
                    self._pad_sequence(prompt_embeds_mask, max_seq_len),
                ])

        # Prepare timesteps — compute sigmas with single shift to match original scheduler
        if timesteps is None and sigmas is None:
            shift = getattr(self.scheduler.config, "shift", 1.0)
            raw_sigmas = torch.linspace(1, 0, num_inference_steps + 1)
            shifted_sigmas = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)
            sigmas = shifted_sigmas[:-1].tolist()
            original_shift = self.scheduler.shift
            self.scheduler.set_shift(1.0)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas
            )
            self.scheduler.set_shift(original_shift)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas
            )

        # Prepare latents (patchified)
        num_channels_latents = self.transformer.config.in_channels
        padded_latents, target_mask, shape_list = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            reference_images=images,
            enable_denormalization=enable_denormalization,
        )

        # Zero out padding text tokens to prevent them from corrupting attention
        # (original uses explicit attention masking; here we neutralize padding values)
        if prompt_embeds_mask is not None:
            prompt_embeds = prompt_embeds * prompt_embeds_mask.unsqueeze(-1)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        clean_reference_backup = padded_latents.clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Restore reference patches
                padded_latents[~target_mask] = clean_reference_backup[~target_mask]

                model_input = padded_latents

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
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # Scheduler step
                padded_latents = self.scheduler.step(noise_pred, t, padded_latents, return_dict=False)[0].to(
                    dtype=prompt_embeds.dtype
                )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    padded_latents = callback_outputs.pop("latents", padded_latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        # Post-processing: decode target latents
        if output_type != "latent":
            padded_latents[~target_mask] = clean_reference_backup[~target_mask]
            pt, ph, pw = self.transformer.config.patch_size

            image_list = []
            for b_idx in range(batch_size):
                l_t, l_h, l_w = shape_list[b_idx][0]
                target_len = l_t * l_h * l_w

                target_patches = padded_latents[b_idx, :target_len]
                video_latent = rearrange(
                    target_patches,
                    "(t h w) c pt ph pw -> 1 c (t pt) (h ph) (w pw)",
                    t=l_t, h=l_h, w=l_w,
                )

                video_latent = self.denormalize_latents(video_latent)

                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    sample_image = self.vae.decode(video_latent.float(), return_dict=False)[0]
                sample_image = (sample_image / 2 + 0.5).clamp(0, 1).squeeze(0).cpu().float()
                image_list.append(sample_image)

            # Convert to output format
            output_images = []
            for img_tensor in image_list:
                # img_tensor: [C, T, H, W] -> [C, H, W] (T=1)
                img_tensor = img_tensor[:, 0]
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                if output_type == "pil":
                    output_images.append(Image.fromarray(img_np))
                elif output_type == "np":
                    output_images.append(img_np)
                else:
                    output_images.append(img_tensor)

            image = output_images
        else:
            image = padded_latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return JoyImageEditPlusPipelineOutput(images=image)
