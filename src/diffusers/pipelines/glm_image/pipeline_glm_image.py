# Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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
import re
from typing import Any, Callable

import numpy as np
import PIL
import torch
from transformers import ByT5Tokenizer, PreTrainedModel, ProcessorMixin, T5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, GlmImageTransformer2DModel
from ...models.transformers.transformer_glm_image import GlmImageKVCache
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, is_transformers_version, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import GlmImagePipelineOutput


# Because it's not released in stable as of 13/01/2026. So this is just a proxy.
GlmImageProcessor = ProcessorMixin
GlmImageForConditionalGeneration = PreTrainedModel
if is_transformers_version(">=", "5.0.0.dev0"):
    from transformers import GlmImageForConditionalGeneration, GlmImageProcessor


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
        >>> from diffusers import GlmImagePipeline

        >>> pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        >>> image.save("output.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    base_shift: float = 0.25,
    max_shift: float = 0.75,
) -> float:
    m = (image_seq_len / base_seq_len) ** 0.5
    mu = m * max_shift + base_shift
    return mu


# Copied from diffusers.pipelines.cogview4.pipeline_cogview4.retrieve_timesteps
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
    accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

    if timesteps is not None and sigmas is not None:
        if not accepts_timesteps and not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep or sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None and sigmas is None:
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is None and sigmas is not None:
        if not accepts_sigmas:
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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class GlmImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using GLM-Image.

    This pipeline integrates both the AR (autoregressive) model for token generation and the DiT (diffusion
    transformer) model for image decoding.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer for the text encoder.
        processor (`AutoProcessor`):
            Processor for the AR model to handle chat templates and tokenization.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder for glyph embeddings.
        vision_language_encoder ([`GlmImageForConditionalGeneration`]):
            The AR model that generates image tokens from text prompts.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        transformer ([`GlmImageTransformer2DModel`]):
            A text conditioned transformer to denoise the encoded image latents (DiT).
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "vision_language_encoder->text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        tokenizer: ByT5Tokenizer,
        processor: GlmImageProcessor,
        text_encoder: T5EncoderModel,
        vision_language_encoder: GlmImageForConditionalGeneration,
        vae: AutoencoderKL,
        transformer: GlmImageTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            processor=processor,
            text_encoder=text_encoder,
            vision_language_encoder=vision_language_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer")
            and self.transformer is not None
            and hasattr(self.transformer.config, "sample_size")
            else 128
        )

    @staticmethod
    def _compute_generation_params(
        image_grid_thw,
        is_text_to_image: bool,
    ):
        grid_sizes = []
        grid_hw = []

        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            grid_sizes.append(int(h * w))
            grid_hw.append((int(h), int(w)))

        if not is_text_to_image:
            max_new_tokens = grid_sizes[-1] + 1
            large_image_start_offset = 0
            target_grid_h, target_grid_w = grid_hw[-1]
        else:
            total_tokens = sum(grid_sizes)
            max_new_tokens = total_tokens + 1
            large_image_start_offset = sum(grid_sizes[1:])
            target_grid_h, target_grid_w = grid_hw[0]
        return max_new_tokens, large_image_start_offset, target_grid_h, target_grid_w

    @staticmethod
    def _extract_large_image_tokens(
        outputs: torch.Tensor, input_length: int, large_image_start_offset: int, large_image_tokens: int
    ) -> torch.Tensor:
        generated_tokens = outputs[0][input_length:]
        large_image_start = large_image_start_offset
        large_image_end = large_image_start + large_image_tokens
        return generated_tokens[large_image_start:large_image_end]

    @staticmethod
    def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        token_ids = token_ids.view(1, 1, token_h, token_w)
        token_ids = torch.nn.functional.interpolate(token_ids.float(), scale_factor=2, mode="nearest").to(
            dtype=torch.long
        )
        token_ids = token_ids.view(1, -1)
        return token_ids

    @staticmethod
    def _validate_and_normalize_images(
        image: list[PIL.Image.Image] | list[list[PIL.Image.Image]],
        batch_size: int,
    ) -> list[list[PIL.Image.Image]]:
        """
        Validate and normalize image inputs to List[List[PIL.Image]].

        Rules:
        - batch_size > 1: Only accepts List[List[PIL.Image]], each sublist must have equal length
        - batch_size == 1: Accepts List[PIL.Image] for legacy compatibility (converted to [[img1, img2, ...]])
        - Other formats raise ValueError

        Args:
            image: Input images in various formats
            batch_size: Number of prompts in the batch

        Returns:
            Normalized images as List[List[PIL.Image]], or None if no images provided
        """
        if image is None or len(image) == 0:
            return None

        first_element = image[0]

        if batch_size == 1:
            # Legacy format: List[PIL.Image] -> [[img1, img2, ...]]
            if not isinstance(first_element, (list, tuple)):
                return [list(image)]
            # Already in List[List[PIL.Image]] format
            if len(image) != 1:
                raise ValueError(
                    f"For batch_size=1 with List[List[PIL.Image]] format, expected 1 image list, got {len(image)}."
                )
            return [list(image[0])]

        # batch_size > 1: must be List[List[PIL.Image]]
        if not isinstance(first_element, (list, tuple)):
            raise ValueError(
                f"For batch_size > 1, images must be List[List[PIL.Image]] format. "
                f"Got List[{type(first_element).__name__}] instead. "
                f"Each prompt requires its own list of condition images."
            )

        if len(image) != batch_size:
            raise ValueError(f"Number of image lists ({len(image)}) must match batch size ({batch_size}).")

        # Validate homogeneous: all sublists must have same length
        num_input_images_per_prompt = len(image[0])
        for idx, imgs in enumerate(image):
            if len(imgs) != num_input_images_per_prompt:
                raise ValueError(
                    f"All prompts must have the same number of condition images. "
                    f"Prompt 0 has {num_input_images_per_prompt} images, but prompt {idx} has {len(imgs)} images."
                )

        return [list(imgs) for imgs in image]

    def generate_prior_tokens(
        self,
        prompt: str | list[str],
        height: int,
        width: int,
        image: list[list[PIL.Image.Image]] | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ):
        """
        Generate prior tokens for the DiT model using the AR model.

        Args:
            prompt: Single prompt or list of prompts
            height: Target image height
            width: Target image width
            image: Normalized image input as List[List[PIL.Image]]. Should be pre-validated
                   using _validate_and_normalize_images() before calling this method.
            device: Target device
            generator: Random generator for reproducibility

        Returns:
            Tuple of:
                - prior_token_ids: Tensor of shape (batch_size, num_tokens) with upsampled prior tokens
                - prior_token_image_ids_per_sample: List of tensors, one per sample. Each tensor contains
                    the upsampled prior token ids for all condition images in that sample. None for t2i.
                - source_image_grid_thw_per_sample: List of tensors, one per sample. Each tensor has shape
                    (num_condition_images, 3) with upsampled grid info. None for t2i.
        """
        device = device or self._execution_device

        # Normalize prompt to list format
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)

        # Image is already normalized by _validate_and_normalize_images(): None or List[List[PIL.Image]]
        is_text_to_image = image is None
        # Build messages for each sample in the batch
        all_messages = []
        for idx, p in enumerate(prompt_list):
            content = []
            if not is_text_to_image:
                for img in image[idx]:
                    content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": p})
            all_messages.append([{"role": "user", "content": content}])
        # Process with the processor (supports batch with left padding)
        inputs = self.processor.apply_chat_template(
            all_messages,
            tokenize=True,
            padding=True if batch_size > 1 else False,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        image_grid_thw = inputs.get("image_grid_thw")
        images_per_sample = inputs.get("images_per_sample")

        # Determine number of condition images and grids per sample
        num_condition_images = 0 if is_text_to_image else len(image[0])
        if images_per_sample is not None:
            num_grids_per_sample = images_per_sample[0].item()
        else:
            # Fallback for batch_size=1: total grids is for single sample
            num_grids_per_sample = image_grid_thw.shape[0]

        # Compute generation params (same for all samples in homogeneous batch)
        first_sample_grids = image_grid_thw[:num_grids_per_sample]
        max_new_tokens, large_image_offset, token_h, token_w = self._compute_generation_params(
            image_grid_thw=first_sample_grids, is_text_to_image=is_text_to_image
        )

        # Generate source image tokens (prior_token_image_ids) for i2i mode
        prior_token_image_ids = None
        source_image_grid_thw = None
        if not is_text_to_image:
            # Extract source grids by selecting condition image indices (skip target grids)
            # Grid order from processor: [s0_cond1, s0_cond2, ..., s0_target, s1_cond1, s1_cond2, ..., s1_target, ...]
            # We need indices: [0, 1, ..., num_condition_images-1, num_grids_per_sample, num_grids_per_sample+1, ...]
            source_indices = []
            for sample_idx in range(batch_size):
                base = sample_idx * num_grids_per_sample
                source_indices.extend(range(base, base + num_condition_images))
            source_grids = image_grid_thw[source_indices]

            if len(source_grids) > 0:
                prior_token_image_embed = self.vision_language_encoder.get_image_features(
                    inputs["pixel_values"], source_grids
                ).pooler_output
                prior_token_image_embed = torch.cat(prior_token_image_embed, dim=0)
                prior_token_image_ids_d32 = self.vision_language_encoder.get_image_tokens(
                    prior_token_image_embed, source_grids
                )
                # Upsample each source image's prior tokens to match VAE/DiT resolution
                split_sizes = source_grids.prod(dim=-1).tolist()
                prior_ids_per_source = torch.split(prior_token_image_ids_d32, split_sizes)
                upsampled_prior_ids = []
                for i, prior_ids in enumerate(prior_ids_per_source):
                    t, h, w = source_grids[i].tolist()
                    upsampled = self._upsample_token_ids(prior_ids, int(h), int(w))
                    upsampled_prior_ids.append(upsampled.squeeze(0))
                prior_token_image_ids = torch.cat(upsampled_prior_ids, dim=0)
                # Upsample grid dimensions for later splitting
                upsampled_grids = source_grids.clone()
                upsampled_grids[:, 1] = upsampled_grids[:, 1] * 2
                upsampled_grids[:, 2] = upsampled_grids[:, 2] * 2
                source_image_grid_thw = upsampled_grids

        # Generate with AR model
        # Set torch random seed from generator for reproducibility
        # (transformers generate() doesn't accept generator parameter)
        if generator is not None:
            seed = generator.initial_seed()
            torch.manual_seed(seed)
            if device is not None and device.type == "cuda":
                torch.cuda.manual_seed(seed)
        outputs = self.vision_language_encoder.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        # Extract and upsample prior tokens for each sample
        # For left-padded inputs, generated tokens start after the padded input sequence
        all_prior_token_ids = []
        max_input_length = inputs["input_ids"].shape[-1]
        for idx in range(batch_size):
            # For left-padded sequences, generated tokens start at max_input_length
            # (padding is on the left, so all sequences end at the same position)
            prior_token_ids_d32 = self._extract_large_image_tokens(
                outputs[idx : idx + 1], max_input_length, large_image_offset, token_h * token_w
            )
            prior_token_ids = self._upsample_token_ids(prior_token_ids_d32, token_h, token_w)
            all_prior_token_ids.append(prior_token_ids)
        prior_token_ids = torch.cat(all_prior_token_ids, dim=0)

        # Split prior_token_image_ids and source_image_grid_thw into per-sample lists for easier consumption
        prior_token_image_ids_per_sample = None
        source_image_grid_thw_per_sample = None
        if prior_token_image_ids is not None and source_image_grid_thw is not None:
            # Split grids: each sample has num_condition_images grids
            source_image_grid_thw_per_sample = list(torch.split(source_image_grid_thw, num_condition_images))
            # Split prior_token_image_ids: tokens per sample may vary due to different image sizes
            tokens_per_image = source_image_grid_thw.prod(dim=-1).tolist()
            tokens_per_sample = []
            for i in range(batch_size):
                start_idx = i * num_condition_images
                end_idx = start_idx + num_condition_images
                tokens_per_sample.append(sum(tokens_per_image[start_idx:end_idx]))
            prior_token_image_ids_per_sample = list(torch.split(prior_token_image_ids, tokens_per_sample))

        return prior_token_ids, prior_token_image_ids_per_sample, source_image_grid_thw_per_sample

    def get_glyph_texts(self, prompt):
        """Extract glyph texts from prompt(s). Returns a list of lists for batch processing."""
        if isinstance(prompt, str):
            prompt = [prompt]
        all_ocr_texts = []
        for p in prompt:
            ocr_texts = (
                re.findall(r"'([^']*)'", p)
                + re.findall(r"\u201c([^\u201c\u201d]*)\u201d", p)
                + re.findall(r'"([^"]*)"', p)
                + re.findall(r"「([^「」]*)」", p)
            )
            all_ocr_texts.append(ocr_texts)
        return all_ocr_texts

    def _get_glyph_embeds(
        self,
        prompt: str | list[str] = None,
        max_sequence_length: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Get glyph embeddings for each prompt in the batch."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        # get_glyph_texts now returns a list of lists (one per prompt)
        all_glyph_texts = self.get_glyph_texts(prompt)

        all_glyph_embeds = []
        for glyph_texts in all_glyph_texts:
            if len(glyph_texts) == 0:
                glyph_texts = [""]
            input_ids = self.tokenizer(
                glyph_texts,
                max_length=max_sequence_length,
                truncation=True,
            ).input_ids
            input_ids = [
                [self.tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + input_ids_ for input_ids_ in input_ids
            ]
            max_length = max(len(input_ids_) for input_ids_ in input_ids)
            attention_mask = torch.tensor(
                [[1] * len(input_ids_) + [0] * (max_length - len(input_ids_)) for input_ids_ in input_ids],
                device=device,
            )
            input_ids = torch.tensor(
                [
                    input_ids_ + [self.tokenizer.pad_token_id] * (max_length - len(input_ids_))
                    for input_ids_ in input_ids
                ],
                device=device,
            )
            outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
            glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)
            all_glyph_embeds.append(glyph_embeds)

        # Pad to same sequence length and stack (use left padding to match transformers)
        max_seq_len = max(emb.size(1) for emb in all_glyph_embeds)
        padded_embeds = []
        for emb in all_glyph_embeds:
            if emb.size(1) < max_seq_len:
                pad = torch.zeros(emb.size(0), max_seq_len - emb.size(1), emb.size(2), device=device, dtype=emb.dtype)
                emb = torch.cat([pad, emb], dim=1)  # left padding
            padded_embeds.append(emb)

        glyph_embeds = torch.cat(padded_embeds, dim=0)
        return glyph_embeds.to(device=device, dtype=dtype)

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int = 2048,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
            max_sequence_length (`int`, defaults to `2048`):
                Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_glyph_embeds(prompt, max_sequence_length, device, dtype)

        # Repeat embeddings for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # For GLM-Image, negative_prompt must be "" instead of None
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._get_glyph_embeds(negative_prompt, max_sequence_length, device, dtype)

            if num_images_per_prompt > 1:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prior_token_ids=None,
        prior_token_image_ids=None,
        source_image_grid_thw=None,
        image=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * self.transformer.config.patch_size * 2) != 0
            or width is not None
            and width % (self.transformer.config.patch_size * 2) != 0
        ):
            # GLM-Image uses 32× downsampling, so the image dimensions must be multiples of 32.
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 4} but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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
        if prompt is None and prior_token_ids is None:
            raise ValueError(
                "Provide either `prompt` or `prior_token_ids`. Cannot leave both `prompt` and `prior_token_ids` undefined."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        # Validate prior token inputs: for i2i mode, all three must be provided together
        # For t2i mode, only prior_token_ids is needed (prior_token_image_ids and source_image_grid_thw should be None)
        prior_image_inputs = [prior_token_image_ids, source_image_grid_thw]
        num_prior_image_inputs = sum(x is not None for x in prior_image_inputs)
        if num_prior_image_inputs > 0 and num_prior_image_inputs < len(prior_image_inputs):
            raise ValueError(
                "`prior_token_image_ids` and `source_image_grid_thw` must be provided together for i2i mode. "
                f"Got prior_token_image_ids={prior_token_image_ids is not None}, "
                f"source_image_grid_thw={source_image_grid_thw is not None}."
            )
        if num_prior_image_inputs > 0 and prior_token_ids is None:
            raise ValueError(
                "`prior_token_ids` must be provided when `prior_token_image_ids` and `source_image_grid_thw` are provided."
            )
        if num_prior_image_inputs > 0 and image is None:
            raise ValueError(
                "`image` must be provided when `prior_token_image_ids` and `source_image_grid_thw` are provided "
                "for i2i mode, as the images are needed for VAE encoding to build the KV cache."
            )

        if prior_token_ids is not None and prompt_embeds is None and prompt is None:
            raise ValueError("`prompt_embeds` or `prompt` must also be provided with `prior_token_ids`.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

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
        prompt: str | list[str] | None = None,
        image: torch.Tensor
        | PIL.Image.Image
        | np.ndarray
        | list[torch.Tensor]
        | list[PIL.Image.Image]
        | list[np.ndarray]
        | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prior_token_ids: torch.Tensor | None = None,
        prior_token_image_ids: list[torch.Tensor] | None = None,
        source_image_grid_thw: list[torch.Tensor] | None = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 2048,
    ) -> GlmImagePipelineOutput | tuple:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. Must contain shape info in the format '<sop>H
                W<eop>' where H and W are token dimensions (d32). Example: "A beautiful sunset<sop>36 24<eop>"
                generates a 1152x768 image.
            image: Optional condition images for image-to-image generation.
            height (`int`, *optional*):
                The height in pixels. If not provided, derived from prompt shape info.
            width (`int`, *optional*):
                The width in pixels. If not provided, derived from prompt shape info.
            num_inference_steps (`int`, *optional*, defaults to `50`):
                The number of denoising steps for DiT.
            guidance_scale (`float`, *optional*, defaults to `1.5`):
                Guidance scale for classifier-free guidance.
            num_images_per_prompt (`int`, *optional*, defaults to `1`):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                Random generator for reproducibility.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format: "pil", "np", or "latent".

        Examples:

        Returns:
            [`GlmImagePipelineOutput`] or `tuple`: Generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
            prior_token_ids,
            prior_token_image_ids,
            source_image_grid_thw,
            image,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Validate and normalize image format
        normalized_image = self._validate_and_normalize_images(image, batch_size)

        # 3. Generate prior tokens (batch mode)
        # Get a single generator for AR model (use first if list provided)
        ar_generator = generator[0] if isinstance(generator, list) else generator
        if prior_token_ids is None:
            prior_token_ids, prior_token_image_ids_per_sample, source_image_grid_thw_per_sample = (
                self.generate_prior_tokens(
                    prompt=prompt,
                    image=normalized_image,
                    height=height,
                    width=width,
                    device=device,
                    generator=ar_generator,
                )
            )
        else:
            # User provided prior_token_ids directly (from generate_prior_tokens)
            prior_token_image_ids_per_sample = prior_token_image_ids
            source_image_grid_thw_per_sample = source_image_grid_thw

        # 4. Preprocess images for VAE encoding
        preprocessed_images = None
        if normalized_image is not None:
            preprocessed_images = []
            for prompt_images in normalized_image:
                prompt_preprocessed = []
                for img in prompt_images:
                    image_height, image_width = img.size[::-1] if isinstance(img, PIL.Image.Image) else img.shape[:2]
                    multiple_of = self.vae_scale_factor * self.transformer.config.patch_size
                    image_height = (image_height // multiple_of) * multiple_of
                    image_width = (image_width // multiple_of) * multiple_of
                    img = self.image_processor.preprocess(img, height=image_height, width=image_width)
                    prompt_preprocessed.append(img)
                    height = height or image_height
                    width = width or image_width
                preprocessed_images.append(prompt_preprocessed)

        # 5. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.dtype,
        )

        # 6. Prepare latents and (optional) image kv cache
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        kv_caches = GlmImageKVCache(num_layers=self.transformer.config.num_layers)

        if normalized_image is not None:
            kv_caches.set_mode("write")
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.latent_channels, 1, 1)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.latent_channels, 1, 1)

            latents_mean = latents_mean.to(device=device, dtype=prompt_embeds.dtype)
            latents_std = latents_std.to(device=device, dtype=prompt_embeds.dtype)

            # Process each sample's condition images
            for prompt_idx in range(batch_size):
                prompt_images = preprocessed_images[prompt_idx]
                prompt_prior_ids = prior_token_image_ids_per_sample[prompt_idx]
                prompt_grid_thw = source_image_grid_thw_per_sample[prompt_idx]

                # Split this sample's prior_token_image_ids by each image's token count
                split_sizes = prompt_grid_thw.prod(dim=-1).tolist()
                prior_ids_per_image = torch.split(prompt_prior_ids, split_sizes)
                # Process each condition image for this sample
                for condition_image, condition_image_prior_token_id in zip(prompt_images, prior_ids_per_image):
                    condition_image = condition_image.to(device=device, dtype=prompt_embeds.dtype)
                    condition_latent = retrieve_latents(
                        self.vae.encode(condition_image), generator=generator, sample_mode="argmax"
                    )
                    condition_latent = (condition_latent - latents_mean) / latents_std

                    _ = self.transformer(
                        hidden_states=condition_latent,
                        encoder_hidden_states=torch.zeros_like(prompt_embeds)[:1, :0, ...],
                        prior_token_id=condition_image_prior_token_id,
                        prior_token_drop=torch.full_like(condition_image_prior_token_id, False, dtype=torch.bool),
                        timestep=torch.zeros((1,), device=device),
                        target_size=torch.tensor([condition_image.shape[-2:]], device=device),
                        crop_coords=torch.zeros((1, 2), device=device),
                        attention_kwargs=attention_kwargs,
                        kv_caches=kv_caches,
                    )
                # Move to next sample's cache slot
                kv_caches.next_sample()

        # 7. Prepare additional timestep conditions
        target_size = (height, width)
        target_size = torch.tensor([target_size], dtype=prompt_embeds.dtype, device=device)
        crops_coords_top_left = torch.tensor([crops_coords_top_left], dtype=prompt_embeds.dtype, device=device)

        target_size = target_size.repeat(batch_size * num_images_per_prompt, 1)
        crops_coords_top_left = crops_coords_top_left.repeat(batch_size * num_images_per_prompt, 1)

        # Prepare timesteps
        image_seq_len = ((height // self.vae_scale_factor) * (width // self.vae_scale_factor)) // (
            self.transformer.config.patch_size**2
        )
        timesteps = (
            np.linspace(self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1)[:-1]
            if timesteps is None
            else np.array(timesteps)
        )
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("base_shift", 0.25),
            self.scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # 8. Denoising loop
        transformer_dtype = self.transformer.dtype
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # Repeat prior_token_ids for num_images_per_prompt
        if num_images_per_prompt > 1:
            prior_token_ids = prior_token_ids.repeat_interleave(num_images_per_prompt, dim=0)
        prior_token_drop_cond = torch.full_like(prior_token_ids, False, dtype=torch.bool)
        prior_token_drop_uncond = torch.full_like(prior_token_ids, True, dtype=torch.bool)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)

                timestep = t.expand(latents.shape[0]) - 1

                if prior_token_image_ids_per_sample is not None:
                    kv_caches.set_mode("read")

                noise_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    prior_token_id=prior_token_ids,
                    prior_token_drop=prior_token_drop_cond,
                    timestep=timestep,
                    target_size=target_size,
                    crop_coords=crops_coords_top_left,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    kv_caches=kv_caches,
                )[0].float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    if prior_token_image_ids_per_sample is not None:
                        kv_caches.set_mode("skip")
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=negative_prompt_embeds,
                        prior_token_id=prior_token_ids,
                        prior_token_drop=prior_token_drop_uncond,
                        timestep=timestep,
                        target_size=target_size,
                        crop_coords=crops_coords_top_left,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        kv_caches=kv_caches,
                    )[0].float()

                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, self.scheduler.sigmas[i], callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        kv_caches.clear()

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return GlmImagePipelineOutput(images=image)
