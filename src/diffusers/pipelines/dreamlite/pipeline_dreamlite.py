# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
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
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin, TextualInversionLoaderMixin
from ...models import AutoencoderTiny
from ...models.unets.unet_dreamlite import DreamLiteUNetModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from ..stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from .pipeline_output import DreamLitePipelineOutput


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
        >>> from PIL import Image
        >>> from diffusers import DreamLitePipeline

        >>> pipe = DreamLitePipeline.from_pretrained(
        ...     "carlofkl/DreamLite-base", revision="diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> # Text-to-image
        >>> image = pipe(prompt="A serene mountain lake at sunrise").images[0]

        >>> # Image-to-image (instruction-based edit)
        >>> init_image = Image.open("input.png").convert("RGB")
        >>> edited = pipe(prompt="make it snowy", image=init_image).images[0]
        ```
"""


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.flux.pipeline_flux.retrieve_timesteps
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


class DreamLitePipeline(DiffusionPipeline, FromSingleFileMixin, TextualInversionLoaderMixin):
    r"""DreamLite pipeline for text-to-image and instruction-based image editing.

    The same pipeline supports both modes; the operating mode is auto-detected from the inputs:

    * ``image is None`` -> text-to-image (single CFG on text).
    * ``image is not None`` -> image-to-image / instruction edit (dual CFG: text + image).

    Components:
        text_encoder ([`~transformers.Qwen3VLForConditionalGeneration`]):
            Multimodal text/vision encoder used to produce conditioning embeddings.
        tokenizer ([`~transformers.AutoTokenizer`]):
            Tokenizer for text-only (generate) mode.
        processor ([`~transformers.Qwen3VLProcessor`]):
            Multimodal processor for edit mode (text + image template).
        vae ([`~diffusers.AutoencoderTiny`]):
            Mobile-friendly tiny VAE for latent encode/decode.
        unet ([`~diffusers.DreamLiteUNetModel`]):
            DreamLite UNet (GQA + qk_norm + depthwise-separable convs).
        scheduler ([`~diffusers.FlowMatchEulerDiscreteScheduler`]):
            Flow-matching Euler scheduler with dynamic shift.

    Note:
        ``batch_size`` is currently forced to ``1``; ``num_images_per_prompt`` is supported.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        processor: Qwen3VLProcessor,
        vae: AutoencoderTiny,
        unet: DreamLiteUNetModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )

        # Safe VAE scale factor: AutoencoderTiny exposes `encoder_block_out_channels`; fall back to 8.
        if self.vae is not None and hasattr(self.vae.config, "encoder_block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128

        # ----- Prompt encoding templates -----
        # ``prompt_template_encode_*`` is the chat template wrapped around the user prompt before tokenisation.
        # ``prompt_template_encode_*_start_idx`` is the number of tokens occupied by the template prefix
        # (system + chat-template scaffolding) that must be dropped from the encoder hidden states so the cross-
        # attention only attends to the **user prompt** content. The values come from running each template (with
        # an empty prompt) through the matching tokenizer / processor and recording the resulting prefix length;
        # they are pinned here for reproducibility, mirroring the pattern used by Qwen-Image pipelines.
        self.prompt_template_encode_generate = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_generate_start_idx = 34
        self.prompt_template_encode_edit = (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
            "texture, objects, background), then explain how the user's text instruction should alter "
            "or modify the image. Generate a new image that meets the user's requirements while maintaining "
            "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_edit_start_idx = 64

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1).tolist()
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths, dim=0)

    def encode_prompt(
        self,
        mode: str,
        prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
        image: Optional[Image.Image] = None,
        max_sequence_length: int = 500,
        text_pad_embedding: Optional[torch.Tensor] = None,
    ):
        if mode == "edit":
            template = self.prompt_template_encode_edit
            drop_idx = self.prompt_template_encode_edit_start_idx

            txts = [template.format(p) for p in prompts]
            # ``VaeImageProcessor.resize`` defaults to LANCZOS resampling, matching the reference preprocessing
            # exactly while avoiding a bespoke ``Image.resize`` call.
            cond_image = self.image_processor.resize(image, height=512, width=512)
            images = [cond_image] * len(prompts)

            tk_out = self.processor(text=txts, images=images, padding=True, return_tensors="pt").to(device)

            outputs = self.text_encoder(
                input_ids=tk_out.input_ids,
                attention_mask=tk_out.attention_mask,
                pixel_values=tk_out.pixel_values,
                image_grid_thw=tk_out.image_grid_thw,
                output_hidden_states=True,
            )

        elif mode == "generate":
            template = self.prompt_template_encode_generate
            drop_idx = self.prompt_template_encode_generate_start_idx

            txts = [template.format(p) for p in prompts]
            tk_out = self.tokenizer(
                text=txts,
                max_length=max_sequence_length + drop_idx,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = self.text_encoder(
                input_ids=tk_out.input_ids,
                attention_mask=tk_out.attention_mask,
                output_hidden_states=True,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}; expected 'generate' or 'edit'.")

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, tk_out.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        prompt_embeds = pad_sequence(split_hidden_states, batch_first=True, padding_value=0).to(
            dtype=dtype, device=device
        )

        B, L, _ = prompt_embeds.shape
        prompt_embeds_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, seq in enumerate(split_hidden_states):
            prompt_embeds_mask[i, : seq.shape[0]] = 1

        if text_pad_embedding is not None:
            pad_emb = text_pad_embedding.to(dtype=dtype, device=device)
            if pad_emb.ndim == 1:
                pad_emb = pad_emb.unsqueeze(0).unsqueeze(0)
            elif pad_emb.ndim == 2:
                pad_emb = pad_emb.unsqueeze(0)

            mask_expanded = prompt_embeds_mask.unsqueeze(-1).to(dtype=dtype)
            prompt_embeds = prompt_embeds * mask_expanded + pad_emb * (1 - mask_expanded)

        return prompt_embeds, prompt_embeds_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Generator list length must match batch size.")

        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def prepare_image_latents(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(f"`image` must be of type `torch.Tensor`, `PIL.Image.Image` or `list`, got {type(image)}")

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")

        return image_latents

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    # ---------------------------------------------------------------------
    # Main entry
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.0,
        num_inference_steps: int = 30,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 200,
        text_pad_embedding: Optional[torch.Tensor] = None,
    ):
        r"""Run the DreamLite pipeline.

        Args:
            prompt: Text prompt.
            negative_prompt: Negative text prompt (defaults to empty string).
            image: Optional input image. If provided, the pipeline runs in **edit / image-to-image** mode
                with dual classifier-free guidance; otherwise it runs in **text-to-image** mode.
            height: Output resolution (height). Defaults to ``default_sample_size * vae_scale_factor`` (1024).
                The same default applies in both T2I and I2I; pass an explicit value to override.
            width: Output resolution (width). Defaults to ``default_sample_size * vae_scale_factor`` (1024).
                The same default applies in both T2I and I2I; pass an explicit value to override.
            guidance_scale: CFG scale on the text branch (both modes).
            image_guidance_scale: Additional CFG scale on the image branch (edit mode only).
            num_inference_steps: Number of denoising steps.
            sigmas: Optional explicit FlowMatch sigmas; defaults to a uniform linspace.
            num_images_per_prompt: Output images per prompt (note: ``batch_size`` is forced to 1).
            generator: Random generator(s).
            output_type: ``"pil"``, ``"np"``, ``"pt"`` or ``"latent"``.
            return_dict: If True, returns a :class:`DreamLitePipelineOutput`; else a tuple ``(images,)``.
            max_sequence_length: Maximum number of user-prompt tokens kept after dropping the chat-template
                prefix. Only applies to ``generate`` mode (the ``edit`` mode uses the multimodal processor's native
                padding).
            text_pad_embedding: Optional learned pad embedding for masked positions.

        Returns:
            :class:`DreamLitePipelineOutput` or ``tuple``.
        """
        # 1. Init pipeline parameters
        if height is None and width is None and image is not None:
            w, h = image.size
            width = (w // self.vae_scale_factor) * self.vae_scale_factor
            height = (h // self.vae_scale_factor) * self.vae_scale_factor
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        task = "generate" if image is None else "edit"
        device = self._execution_device
        dtype = self.text_encoder.dtype
        batch_size = 1  # Note: pipeline currently forces batch_size = 1.
        negative_prompt = negative_prompt or ""

        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        # 2. Prepare Time IDs (carries original H,W as additional conditioning)
        original_size = (width, height)
        add_time_ids = torch.tensor([list(original_size)], device=device, dtype=dtype)

        # 3. Prepare Noise Latents (x_t)
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )

        # 4. Prepare Timesteps (FlowMatch with dynamic shift)
        image_seq_len = latents.shape[2] * latents.shape[3] // 4
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 5. Prepare Conditions (Text & Image)
        if task == "generate":
            prompt_str = f"[Generate]: {prompt}"
            prompt_embeds, text_attention_mask = self.encode_prompt(
                mode="generate",
                prompts=[negative_prompt, prompt_str],
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
                text_pad_embedding=text_pad_embedding,
            )
            if num_images_per_prompt > 1:
                prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                text_attention_mask = text_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
            image_latents = torch.zeros_like(latents)
        else:
            prompt_str = (
                f"[Edit]: A diptych with two side-by-side images of the same scene. "
                f"Compared to the right side, the left one has {prompt}"
            )
            prompt_embeds, text_attention_mask = self.encode_prompt(
                mode="edit",
                prompts=[negative_prompt, negative_prompt, prompt_str],
                image=image,
                device=device,
                dtype=dtype,
            )
            if num_images_per_prompt > 1:
                prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                text_attention_mask = text_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
            image_processed = self.image_processor.preprocess(image, height=height, width=width)
            image_latents = self.prepare_image_latents(
                image_processed,
                dtype=dtype,
                device=device,
            )
            uncond_image_latents = torch.zeros_like(latents)

        # 6. Denoising Loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                if task == "generate":
                    latents_in = torch.cat([latents] * 2)
                    cond_img_in = torch.cat([image_latents] * 2)
                    model_input = torch.cat([latents_in, cond_img_in], dim=3)
                    time_ids_in = torch.cat([add_time_ids] * 2)
                else:  # edit
                    latents_in = torch.cat([latents] * 3)
                    cond_img_in = torch.cat([uncond_image_latents, image_latents, image_latents])
                    model_input = torch.cat([latents_in, cond_img_in], dim=3)
                    time_ids_in = torch.cat([add_time_ids] * 3)

                # UNet Forward
                noise_pred = self.unet(
                    model_input,
                    timestep=t.expand(model_input.shape[0]).to(latents.dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=text_attention_mask,
                    added_cond_kwargs={"time_ids": time_ids_in},
                    return_dict=False,
                )[0]

                # Classifier-Free Guidance (single for T2I, dual for I2I)
                noise_pred = noise_pred[..., : latents.shape[-1]]
                if task == "generate":
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:  # edit
                    noise_pred_uncond, noise_pred_image, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # Scheduler Step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 7. Decode Latents
        if output_type == "latent":
            image_out = latents
        else:
            shift_factor = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
            latents = (latents / self.vae.config.scaling_factor) + shift_factor
            image_out = self.vae.decode(latents, return_dict=False)[0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)

        return DreamLitePipelineOutput(images=image_out)
