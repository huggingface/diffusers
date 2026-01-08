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
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import ByT5Tokenizer, GlmImageForConditionalGeneration, GlmImageProcessor, T5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...loaders import CogView4LoraLoaderMixin
from ...models import AutoencoderKL, GlmImageTransformer2DModel
from ...models.transformers.transformer_glm_image import GlmImageKVCache
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import GlmImagePipelineOutput


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

        >>> prompt = "A photo of an astronaut riding a horse on mars<sop>36 24<eop>"
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


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
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
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class GlmImagePipeline(DiffusionPipeline, CogView4LoraLoaderMixin):
    r"""
    Pipeline for text-to-image generation using GLM-Image.

    This pipeline integrates both the AR (autoregressive) model for token generation and the DiT (diffusion
    transformer) model for image decoding.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder for glyph embeddings.
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer for the text encoder.
        processor (`AutoProcessor`):
            Processor for the AR model to handle chat templates and tokenization.
        vision_language_encoder ([`GlmImageForConditionalGeneration`]):
            The AR model that generates image tokens from text prompts.
        transformer ([`GlmImageTransformer2DModel`]):
            A text conditioned transformer to denoise the encoded image latents (DiT).
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "transformer->vae"
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

    def _build_image_grid_thw(
        self,
        token_h: int,
        token_w: int,
        prev_token_h: int,
        prev_token_w: int,
        existing_grid: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if existing_grid is None or existing_grid.numel() == 0:
            return torch.tensor(
                [
                    [1, token_h, token_w],
                    [1, prev_token_h, prev_token_w],
                ],
                device=device,
            )
        else:
            return torch.cat([existing_grid.to(device), torch.tensor([[1, token_h, token_w]], device=device)], dim=0)

    def _calculate_ar_generation_params(
        self, token_h: int, token_w: int, prev_token_h: int, prev_token_w: int, is_text_to_image: bool
    ) -> Tuple[int, int]:
        """
        Calculate max_new_tokens and large_image_start_offset for AR generation.
        """
        large_image_tokens = token_h * token_w
        small_image_tokens = prev_token_h * prev_token_w

        if is_text_to_image:
            max_new_tokens = small_image_tokens + large_image_tokens + 1
            large_image_start_offset = small_image_tokens
        else:
            max_new_tokens = large_image_tokens + 1
            large_image_start_offset = 0

        return max_new_tokens, large_image_start_offset

    def _extract_large_image_tokens(
        self, outputs: torch.Tensor, input_length: int, large_image_start_offset: int, large_image_tokens: int
    ) -> torch.Tensor:
        generated_tokens = outputs[0][input_length:]
        large_image_start = large_image_start_offset
        large_image_end = large_image_start + large_image_tokens
        return generated_tokens[large_image_start:large_image_end]

    def _upsample_d32_to_d16(self, token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        """
        Upsample token IDs from d32 format to d16 format.

        AR model generates tokens at d32 resolution (each token = 32x32 pixels). DiT expects tokens at d16 resolution
        (each token = 16x16 pixels). This function performs 2x nearest-neighbor upsampling.

        Args:
            token_ids: Token IDs of shape [N] where N = token_h * token_w
            token_h: Height in d32 token units
            token_w: Width in d32 token units

        Returns:
            Upsampled token IDs of shape [1, N*4] where N*4 = (token_h*2) * (token_w*2)
        """
        token_ids = token_ids.view(1, 1, token_h, token_w)
        token_ids = torch.nn.functional.interpolate(token_ids.float(), scale_factor=2, mode="nearest").to(
            dtype=torch.long
        )

        token_ids = token_ids.view(1, -1)
        return token_ids

    def _build_prompt_with_shape(
        self,
        prompt: str,
        height: int,
        width: int,
        is_text_to_image: bool,
        factor: int = 32,
    ) -> Tuple[str, int, int, int, int]:
        """
        Build prompt with shape info (<sop>H W<eop>) based on height and width.

        Args:
            prompt: The raw text prompt without shape info
            height: Target image height in pixels
            width: Target image width in pixels
            is_text_to_image: Whether this is text-to-image (True) or image-to-image (False)

        Returns:
            Tuple of (expanded_prompt, token_h, token_w, prev_token_h, prev_token_w)
        """
        token_h = height // factor
        token_w = width // factor
        ratio = token_h / token_w
        prev_token_h = int(sqrt(ratio) * (factor // 2))
        prev_token_w = int(sqrt(1 / ratio) * (factor // 2))

        if is_text_to_image:
            expanded_prompt = f"{prompt}<sop>{token_h} {token_w}<eop><sop>{prev_token_h} {prev_token_w}<eop>"
        else:
            expanded_prompt = f"{prompt}<sop>{token_h} {token_w}<eop>"

        return expanded_prompt, token_h, token_w, prev_token_h, prev_token_w

    def generate_prior_tokens(
        self,
        prompt: str,
        height: int,
        width: int,
        image: Optional[List[PIL.Image.Image]] = None,
        factor: int = 32,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Generate prior tokens using the AR (vision_language_encoder) model.

        Automatically builds the prompt with shape info based on height/width. Users only need to provide the raw text
        prompt without <sop>...<eop> tags.

        Args:
            prompt: The raw text prompt (without shape info)
            height: Target image height in pixels (must be divisible by factor)
            width: Target image width in pixels (must be divisible by factor)
            image: Optional list of condition images for image-to-image generation
            factor: Token size factor (32 for d32 tokens)

        Returns:
            Tuple of (prior_token_ids, pixel_height, pixel_width)
            - prior_token_ids: Upsampled to d16 format, shape [1, token_h*token_w*4]
            - pixel_height: Image height in pixels (aligned to factor)
            - pixel_width: Image width in pixels (aligned to factor)

        """
        device = self.vision_language_encoder.device
        height = (height // factor) * factor
        width = (width // factor) * factor
        is_text_to_image = image is None or len(image) == 0
        expanded_prompt, token_h, token_w, prev_h, prev_w = self._build_prompt_with_shape(
            prompt, height, width, is_text_to_image
        )
        content = []
        if image is not None:
            for img in image:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": expanded_prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )

        existing_grid = inputs.get("image_grid_thw")
        inputs["image_grid_thw"] = self._build_image_grid_thw(
            token_h,
            token_w,
            prev_h,
            prev_w,
            existing_grid=existing_grid if not is_text_to_image else None,
            device=device,
        )

        max_new_tokens, large_image_offset = self._calculate_ar_generation_params(
            token_h, token_w, prev_h, prev_w, is_text_to_image
        )
        large_image_tokens = token_h * token_w

        inputs = inputs.to(device)
        input_length = inputs["input_ids"].shape[-1]

        outputs = self.vision_language_encoder.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        prior_token_ids_d32 = self._extract_large_image_tokens(
            outputs, input_length, large_image_offset, large_image_tokens
        )
        prior_token_ids = self._upsample_d32_to_d16(prior_token_ids_d32, token_h, token_w)

        pixel_height = token_h * factor
        pixel_width = token_w * factor

        return prior_token_ids, pixel_height, pixel_width

    def get_glyph_texts(self, prompt):
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        ocr_texts = (
            re.findall(r"'([^']*)'", prompt)
            + re.findall(r"“([^“”]*)”", prompt)
            + re.findall(r'"([^"]*)"', prompt)
            + re.findall(r"「([^「」]*)」", prompt)
        )
        return ocr_texts

    def _get_glyph_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        glyph_texts = self.get_glyph_texts(prompt)
        input_ids = self.tokenizer(
            glyph_texts if len(glyph_texts) > 0 else [""],
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids
        input_ids = [
            [self.tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + input_ids_ for input_ids_ in input_ids
        ]
        max_length = max(len(input_ids_) for input_ids_ in input_ids)
        attention_mask = torch.tensor(
            [[1] * len(input_ids_) + [0] * (max_length - len(input_ids_)) for input_ids_ in input_ids], device=device
        )
        input_ids = torch.tensor(
            [input_ids_ + [self.tokenizer.pad_token_id] * (max_length - len(input_ids_)) for input_ids_ in input_ids],
            device=device,
        )
        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)

        return glyph_embeds.to(device=device, dtype=dtype)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 2048,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
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

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_glyph_embeds(negative_prompt, max_sequence_length, device, dtype)

            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

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
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * self.transformer.config.patch_size) != 0
            or width is not None
            and width % (self.transformer.config.patch_size) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
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
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[
            Union[
                torch.Tensor, PIL.Image.Image, np.ndarray, List[torch.Tensor], List[PIL.Image.Image], List[np.ndarray]
            ]
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 2048,
    ) -> Union[GlmImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
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
        assert batch_size == 1, "batch_size must be 1"

        device = self._execution_device

        prior_token_id, prior_token_image_ids, ar_height, ar_width = self.generate_prior_tokens(
            prompt=prompt[0] if isinstance(prompt, list) else prompt,
            image=image,
            height=height,
            width=width,
        )

        height = height or ar_height
        width = width or ar_width

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.dtype,
        )

        # 4. process images
        condition_images_prior_token_id = None
        if image is not None:
            preprocessed_condition_images = []
            condition_images_prior_token_id = []
            for img in image:
                image_height, image_width = img.size[::-1] if isinstance(img, PIL.Image.Image) else img.shape[:2]
                multiple_of = self.vae_scale_factor * self.transformer.config.patch_size
                image_height = (image_height // multiple_of) * multiple_of
                image_width = (image_width // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width)
                preprocessed_condition_images.append(img)
                height = height or image_height
                width = width or image_width
            image = preprocessed_condition_images
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. Prepare latents and (optional) image kv cache
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        kv_caches = GlmImageKVCache(num_layers=self.transformer.config.num_layers)

        if image is not None:
            kv_caches.set_mode("write")
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(self.vae.device, self.vae.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(self.vae.device, self.vae.dtype)
            )
            empty_glyph_hiddens = torch.zeros_like(prompt_embeds)[:1, :0, ...]
            for condition_image, condition_image_prior_token_id in zip(image, condition_images_prior_token_id):
                condition_image = condition_image.to(device=device, dtype=self.vae.dtype)
                condition_latent = retrieve_latents(
                    self.vae.encode(condition_image), generator=generator, sample_mode="argmax"
                )
                condition_latent = (condition_latent - latents_mean) / latents_std
                _ = self.transformer(
                    hidden_states=condition_latent,
                    encoder_hidden_states=empty_glyph_hiddens,
                    prior_token_id=condition_image_prior_token_id,
                    prior_token_drop=torch.full_like(condition_image_prior_token_id, False, dtype=torch.bool),
                    timestep=torch.zeros((1,), device=device),
                    target_size=torch.tensor([condition_image.shape[-2:]], device=device),
                    crop_coords=torch.zeros((1, 2), device=device),
                    attention_kwargs=attention_kwargs,
                    kv_caches=kv_caches,
                )

        # 6. Prepare additional timestep conditions
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

        # 7. Denoising loop
        transformer_dtype = self.transformer.dtype
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        prior_token_drop_cond = torch.full_like(prior_token_id, False, dtype=torch.bool)
        prior_token_drop_uncond = torch.full_like(prior_token_id, True, dtype=torch.bool)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)

                timestep = t.expand(latents.shape[0]) - 1

                if image is not None:
                    kv_caches.set_mode("read")

                noise_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    prior_token_id=prior_token_id,
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
                    if image is not None:
                        kv_caches.set_mode("skip")
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=negative_prompt_embeds,
                        prior_token_id=prior_token_id,
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
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return GlmImagePipelineOutput(images=image)
