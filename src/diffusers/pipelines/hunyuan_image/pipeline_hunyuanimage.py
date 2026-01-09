# Copyright 2025 Hunyuan-Image Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import ByT5Tokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, T5EncoderModel

from ...guiders import AdaptiveProjectedMixGuidance
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLHunyuanImage, HunyuanImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HunyuanImagePipelineOutput


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
        ...     "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, negative_prompt="", num_inference_steps=50).images[0]
        >>> image.save("hunyuanimage.png")
        ```
"""


def extract_glyph_text(prompt: str):
    """
    Extract text enclosed in quotes for glyph rendering.

    Finds text in single quotes, double quotes, and Chinese quotes, then formats it for byT5 processing.

    Args:
        prompt: Input text prompt

    Returns:
        Formatted glyph text string or None if no quoted text found
    """
    text_prompt_texts = []
    pattern_quote_single = r"\'(.*?)\'"
    pattern_quote_double = r"\"(.*?)\""
    pattern_quote_chinese_single = r"‘(.*?)’"
    pattern_quote_chinese_double = r"“(.*?)”"

    matches_quote_single = re.findall(pattern_quote_single, prompt)
    matches_quote_double = re.findall(pattern_quote_double, prompt)
    matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, prompt)
    matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, prompt)

    text_prompt_texts.extend(matches_quote_single)
    text_prompt_texts.extend(matches_quote_double)
    text_prompt_texts.extend(matches_quote_chinese_single)
    text_prompt_texts.extend(matches_quote_chinese_double)

    if text_prompt_texts:
        glyph_text_formatted = ". ".join([f'Text "{text}"' for text in text_prompt_texts]) + ". "
    else:
        glyph_text_formatted = None

    return glyph_text_formatted


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
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

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
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


class HunyuanImagePipeline(DiffusionPipeline):
    r"""
    The HunyuanImage pipeline for text-to-image generation.

    Args:
        transformer ([`HunyuanImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLHunyuanImage`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.
        tokenizer (`Qwen2Tokenizer`): Tokenizer of class [Qwen2Tokenizer].
        text_encoder_2 ([`T5EncoderModel`]):
            [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel)
            variant.
        tokenizer_2 (`ByT5Tokenizer`): Tokenizer of class [ByT5Tokenizer]
        guider ([`AdaptiveProjectedMixGuidance`]):
            [AdaptiveProjectedMixGuidance]to be used to guide the image generation.
        ocr_guider ([`AdaptiveProjectedMixGuidance`], *optional*):
            [AdaptiveProjectedMixGuidance] to be used to guide the image generation when text rendering is needed.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    _optional_components = ["ocr_guider", "guider"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLHunyuanImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: ByT5Tokenizer,
        transformer: HunyuanImageTransformer2DModel,
        guider: Optional[AdaptiveProjectedMixGuidance] = None,
        ocr_guider: Optional[AdaptiveProjectedMixGuidance] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            guider=guider,
            ocr_guider=ocr_guider,
        )

        self.vae_scale_factor = self.vae.config.spatial_compression_ratio if getattr(self, "vae", None) else 32
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 128
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 64

    def _get_qwen_prompt_embeds(
        self,
        tokenizer: Qwen2Tokenizer,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tokenizer_max_length: int = 1000,
        template: str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>",
        drop_idx: int = 34,
        hidden_state_skip_layer: int = 2,
    ):
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        txt = [template.format(e) for e in prompt]
        txt_tokens = tokenizer(
            txt, max_length=tokenizer_max_length + drop_idx, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)

        encoder_hidden_states = text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = encoder_hidden_states.hidden_states[-(hidden_state_skip_layer + 1)]

        prompt_embeds = prompt_embeds[:, drop_idx:]
        encoder_attention_mask = txt_tokens.attention_mask[:, drop_idx:]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        encoder_attention_mask = encoder_attention_mask.to(device=device)

        return prompt_embeds, encoder_attention_mask

    def _get_byt5_prompt_embeds(
        self,
        tokenizer: ByT5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tokenizer_max_length: int = 128,
    ):
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        if isinstance(prompt, list):
            raise ValueError("byt5 prompt should be a string")
        elif prompt is None:
            raise ValueError("byt5 prompt should not be None")

        txt_tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        prompt_embeds = text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask.float(),
        )[0]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        encoder_attention_mask = txt_tokens.attention_mask.to(device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            batch_size (`int`):
                batch size of prompts, defaults to 1
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input
                argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
            prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
        """
        device = device or self._execution_device

        if prompt is None:
            prompt = [""] * batch_size

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=prompt,
                device=device,
                tokenizer_max_length=self.tokenizer_max_length,
                template=self.prompt_template_encode,
                drop_idx=self.prompt_template_encode_start_idx,
            )

        if prompt_embeds_2 is None:
            prompt_embeds_2_list = []
            prompt_embeds_mask_2_list = []

            glyph_texts = [extract_glyph_text(p) for p in prompt]
            for glyph_text in glyph_texts:
                if glyph_text is None:
                    glyph_text_embeds = torch.zeros(
                        (1, self.tokenizer_2_max_length, self.text_encoder_2.config.d_model), device=device
                    )
                    glyph_text_embeds_mask = torch.zeros(
                        (1, self.tokenizer_2_max_length), device=device, dtype=torch.int64
                    )
                else:
                    glyph_text_embeds, glyph_text_embeds_mask = self._get_byt5_prompt_embeds(
                        tokenizer=self.tokenizer_2,
                        text_encoder=self.text_encoder_2,
                        prompt=glyph_text,
                        device=device,
                        tokenizer_max_length=self.tokenizer_2_max_length,
                    )

                prompt_embeds_2_list.append(glyph_text_embeds)
                prompt_embeds_mask_2_list.append(glyph_text_embeds_mask)

            prompt_embeds_2 = torch.cat(prompt_embeds_2_list, dim=0)
            prompt_embeds_mask_2 = torch.cat(prompt_embeds_mask_2_list, dim=0)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(batch_size * num_images_per_prompt, seq_len_2, -1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(batch_size * num_images_per_prompt, seq_len_2)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

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
        prompt_embeds_2=None,
        prompt_embeds_mask_2=None,
        negative_prompt_embeds_2=None,
        negative_prompt_embeds_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
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

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if prompt is None and prompt_embeds_2 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_2`. Cannot leave both `prompt` and `prompt_embeds_2` undefined."
            )

        if prompt_embeds_2 is not None and prompt_embeds_mask_2 is None:
            raise ValueError(
                "If `prompt_embeds_2` are provided, `prompt_embeds_mask_2` also have to be passed. Make sure to generate `prompt_embeds_mask_2` from the same text encoder that was used to generate `prompt_embeds_2`."
            )
        if negative_prompt_embeds_2 is not None and negative_prompt_embeds_mask_2 is None:
            raise ValueError(
                "If `negative_prompt_embeds_2` are provided, `negative_prompt_embeds_mask_2` also have to be passed. Make sure to generate `negative_prompt_embeds_mask_2` from the same text encoder that was used to generate `negative_prompt_embeds_2`."
            )

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
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

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
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        distilled_guidance_scale: Optional[float] = 3.25,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined and negative_prompt_embeds is
                not provided, will use an empty negative prompt. Ignored when not using guidance. ).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            distilled_guidance_scale (`float`, *optional*, defaults to None):
                A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
                where the guidance scale is applied during inference through noise prediction rescaling, guidance
                distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
                is enabled by setting `distilled_guidance_scale > 1`. Higher guidance scale encourages to generate
                images that are closely linked to the text `prompt`, usually at the expense of lower image quality. For
                guidance distilled models, this parameter is required. For non-distilled models, this parameter will be
                ignored.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
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
                Pre-generated text embeddings mask. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, text embeddings mask will be generated from `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated text embeddings for ocr. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, text embeddings for ocr will be generated from `prompt` input argument.
            prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated text embeddings mask for ocr. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, text embeddings mask for ocr will be generated from `prompt` input
                argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings mask. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative text embeddings mask will be generated from `negative_prompt`
                input argument.
            negative_prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings for ocr. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative text embeddings for ocr will be generated from `negative_prompt`
                input argument.
            negative_prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings mask for ocr. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, negative text embeddings mask for ocr will be generated from
                `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.hunyuan_image.HunyuanImagePipelineOutput`] or `tuple`:
            [`~pipelines.hunyuan_image.HunyuanImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
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
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
        )

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

        # 3. prepare prompt embeds

        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
        )

        prompt_embeds = prompt_embeds.to(self.transformer.dtype)
        prompt_embeds_2 = prompt_embeds_2.to(self.transformer.dtype)

        # select guider
        if not torch.all(prompt_embeds_2 == 0) and self.ocr_guider is not None:
            # prompt contains ocr and pipeline has a guider for ocr
            guider = self.ocr_guider
        elif self.guider is not None:
            guider = self.guider
        # distilled model does not use guidance method, use default guider with enabled=False
        else:
            guider = AdaptiveProjectedMixGuidance(enabled=False)

        if guider._enabled and guider.num_conditions > 1:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds_2=negative_prompt_embeds_2,
                prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
            )

            negative_prompt_embeds = negative_prompt_embeds.to(self.transformer.dtype)
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(self.transformer.dtype)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance (for guidance-distilled model)
        if self.transformer.config.guidance_embeds and distilled_guidance_scale is None:
            raise ValueError("`distilled_guidance_scale` is required for guidance-distilled model.")

        if self.transformer.config.guidance_embeds:
            guidance = (
                torch.tensor(
                    [distilled_guidance_scale] * latents.shape[0], dtype=self.transformer.dtype, device=device
                )
                * 1000.0
            )

        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if self.transformer.config.use_meanflow:
                    if i == len(timesteps) - 1:
                        timestep_r = torch.tensor([0.0], device=device)
                    else:
                        timestep_r = timesteps[i + 1]
                    timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype)
                else:
                    timestep_r = None

                # Step 1: Collect model inputs needed for the guidance method
                # conditional inputs should always be first element in the tuple
                guider_inputs = {
                    "encoder_hidden_states": (prompt_embeds, negative_prompt_embeds),
                    "encoder_attention_mask": (prompt_embeds_mask, negative_prompt_embeds_mask),
                    "encoder_hidden_states_2": (prompt_embeds_2, negative_prompt_embeds_2),
                    "encoder_attention_mask_2": (prompt_embeds_mask_2, negative_prompt_embeds_mask_2),
                }

                # Step 2: Update guider's internal state for this denoising step
                guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)

                # Step 3: Prepare batched model inputs based on the guidance method
                # The guider splits model inputs into separate batches for conditional/unconditional predictions.
                # For CFG with guider_inputs = {"encoder_hidden_states": (prompt_embeds, negative_prompt_embeds)}:
                # you will get a guider_state with two batches:
                #   guider_state = [
                #       {"encoder_hidden_states": prompt_embeds, "__guidance_identifier__": "pred_cond"},      # conditional batch
                #       {"encoder_hidden_states": negative_prompt_embeds, "__guidance_identifier__": "pred_uncond"},  # unconditional batch
                #   ]
                # Other guidance methods may return 1 batch (no guidance) or 3+ batches (e.g., PAG, APG).
                guider_state = guider.prepare_inputs(guider_inputs)
                # Step 4: Run the denoiser for each batch
                # Each batch in guider_state represents a different conditioning (conditional, unconditional, etc.).
                # We run the model once per batch and store the noise prediction in guider_state_batch.noise_pred.
                for guider_state_batch in guider_state:
                    guider.prepare_models(self.transformer)

                    # Extract conditioning kwargs for this batch (e.g., encoder_hidden_states)
                    cond_kwargs = {
                        input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()
                    }

                    # e.g. "pred_cond"/"pred_uncond"
                    context_name = getattr(guider_state_batch, guider._identifier_key)
                    with self.transformer.cache_context(context_name):
                        # Run denoiser and store noise prediction in this batch
                        guider_state_batch.noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep,
                            timestep_r=timestep_r,
                            guidance=guidance,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                            **cond_kwargs,
                        )[0]

                    # Cleanup model (e.g., remove hooks)
                    guider.cleanup_models(self.transformer)

                # Step 5: Combine predictions using the guidance method
                # The guider takes all noise predictions from guider_state and combines them according to the guidance algorithm.
                # Continuing the CFG example, the guider receives:
                #   guider_state = [
                #       {"encoder_hidden_states": prompt_embeds, "noise_pred": noise_pred_cond, "__guidance_identifier__": "pred_cond"},      # batch 0
                #       {"encoder_hidden_states": negative_prompt_embeds, "noise_pred": noise_pred_uncond, "__guidance_identifier__": "pred_uncond"},  # batch 1
                #   ]
                # And extracts predictions using the __guidance_identifier__:
                #   pred_cond = guider_state[0]["noise_pred"]      # extracts noise_pred_cond
                #   pred_uncond = guider_state[1]["noise_pred"]    # extracts noise_pred_uncond
                # Then applies CFG formula:
                #   noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                # Returns GuiderOutput(pred=noise_pred, pred_cond=pred_cond, pred_uncond=pred_uncond)
                noise_pred = guider(guider_state)[0]

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
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return HunyuanImagePipelineOutput(images=image)
