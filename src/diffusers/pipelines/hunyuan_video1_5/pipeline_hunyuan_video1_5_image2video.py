# Copyright 2025 The HunyuanVideo Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLTextModel,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from ...guiders import ClassifierFreeGuidance
from ...models import AutoencoderKLHunyuanVideo15, HunyuanVideo15Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import HunyuanVideo15ImageProcessor
from .pipeline_output import HunyuanVideo15PipelineOutput


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
        >>> from diffusers import HunyuanVideo15ImageToVideoPipeline
        >>> from diffusers.utils import export_to_video

        >>> model_id = "hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v"
        >>> pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        >>> pipe.vae.enable_tiling()
        >>> pipe.to("cuda")

        >>> image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")

        >>> output = pipe(
        ...     prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        ...     image=image,
        ...     num_inference_steps=50,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=24)
        ```
"""


# Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.format_text_input
def format_text_input(prompt: List[str], system_message: str) -> List[Dict[str, Any]]:
    """
    Apply text to template.

    Args:
        prompt (List[str]): Input text.
        system_message (str): System message.

    Returns:
        List[Dict[str, Any]]: List of chat conversation.
    """

    template = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]

    return template


# Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.extract_glyph_texts
def extract_glyph_texts(prompt: str) -> List[str]:
    """
    Extract glyph texts from prompt using regex pattern.

    Args:
        prompt: Input prompt string

    Returns:
        List of extracted glyph texts
    """
    pattern = r"\"(.*?)\"|“(.*?)”"
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
    else:
        formatted_result = None

    return formatted_result


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


class HunyuanVideo15ImageToVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-video generation using HunyuanVideo1.5.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`HunyuanVideo15Transformer3DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        vae ([`AutoencoderKLHunyuanVideo15`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.
        tokenizer (`Qwen2Tokenizer`): Tokenizer of class [Qwen2Tokenizer].
        text_encoder_2 ([`T5EncoderModel`]):
            [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel)
            variant.
        tokenizer_2 (`ByT5Tokenizer`): Tokenizer of class [ByT5Tokenizer]
        guider ([`ClassifierFreeGuidance`]):
            [ClassifierFreeGuidance]for classifier free guidance.
        image_encoder ([`SiglipVisionModel`]):
            [SiglipVisionModel](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipVisionModel)
            variant.
        feature_extractor ([`SiglipImageProcessor`]):
            [SiglipImageProcessor](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipImageProcessor)
            variant.
    """

    model_cpu_offload_seq = "image_encoder->text_encoder->transformer->vae"

    def __init__(
        self,
        text_encoder: Qwen2_5_VLTextModel,
        tokenizer: Qwen2Tokenizer,
        transformer: HunyuanVideo15Transformer3DModel,
        vae: AutoencoderKLHunyuanVideo15,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: ByT5Tokenizer,
        guider: ClassifierFreeGuidance,
        image_encoder: SiglipVisionModel,
        feature_extractor: SiglipImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            guider=guider,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        self.video_processor = HunyuanVideo15ImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial, do_resize=False, do_convert_rgb=True
        )
        self.target_size = self.transformer.config.target_size if getattr(self, "transformer", None) else 640
        self.vision_states_dim = (
            self.transformer.config.image_embed_dim if getattr(self, "transformer", None) else 1152
        )
        self.num_channels_latents = self.vae.config.latent_channels if hasattr(self, "vae") else 32
        # fmt: off
        self.system_message = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
        # fmt: on
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.vision_num_semantic_tokens = 729

    @staticmethod
    # Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.HunyuanVideo15Pipeline._get_mllm_prompt_embeds
    def _get_mllm_prompt_embeds(
        text_encoder: Qwen2_5_VLTextModel,
        tokenizer: Qwen2Tokenizer,
        prompt: Union[str, List[str]],
        device: torch.device,
        tokenizer_max_length: int = 1000,
        num_hidden_layers_to_skip: int = 2,
        # fmt: off
        system_message: str = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.",
        # fmt: on
        crop_start: int = 108,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt = format_text_input(prompt, system_message)

        text_inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=tokenizer_max_length + crop_start,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    # Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.HunyuanVideo15Pipeline._get_byt5_prompt_embeds
    def _get_byt5_prompt_embeds(
        tokenizer: ByT5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, List[str]],
        device: torch.device,
        tokenizer_max_length: int = 256,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        glyph_texts = [extract_glyph_texts(p) for p in prompt]

        prompt_embeds_list = []
        prompt_embeds_mask_list = []

        for glyph_text in glyph_texts:
            if glyph_text is None:
                glyph_text_embeds = torch.zeros(
                    (1, tokenizer_max_length, text_encoder.config.d_model), device=device, dtype=text_encoder.dtype
                )
                glyph_text_embeds_mask = torch.zeros((1, tokenizer_max_length), device=device, dtype=torch.int64)
            else:
                txt_tokens = tokenizer(
                    glyph_text,
                    padding="max_length",
                    max_length=tokenizer_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                glyph_text_embeds = text_encoder(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0]
                glyph_text_embeds = glyph_text_embeds.to(device=device)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        prompt_embeds_mask = torch.cat(prompt_embeds_mask_list, dim=0)

        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _get_image_latents(
        vae: AutoencoderKLHunyuanVideo15,
        image_processor: HunyuanVideo15ImageProcessor,
        image: PIL.Image.Image,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        vae_dtype = vae.dtype
        image_tensor = image_processor.preprocess(image, height=height, width=width).to(device, dtype=vae_dtype)
        image_tensor = image_tensor.unsqueeze(2)
        image_latents = retrieve_latents(vae.encode(image_tensor), sample_mode="argmax")
        image_latents = image_latents * vae.config.scaling_factor
        return image_latents

    @staticmethod
    def _get_image_embeds(
        image_encoder: SiglipVisionModel,
        feature_extractor: SiglipImageProcessor,
        image: PIL.Image.Image,
        device: torch.device,
    ) -> torch.Tensor:
        image_encoder_dtype = next(image_encoder.parameters()).dtype
        image = feature_extractor.preprocess(images=image, do_resize=True, return_tensors="pt", do_convert_rgb=True)
        image = image.to(device=device, dtype=image_encoder_dtype)
        image_enc_hidden_states = image_encoder(**image).last_hidden_state

        return image_enc_hidden_states

    def encode_image(
        self,
        image: PIL.Image.Image,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_embeds = self._get_image_embeds(
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor,
            image=image,
            device=device,
        )
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(device=device, dtype=dtype)
        return image_embeds

    # Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.HunyuanVideo15Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 1,
        num_videos_per_prompt: int = 1,
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
        dtype = dtype or self.text_encoder.dtype

        if prompt is None:
            prompt = [""] * batch_size

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=prompt,
                device=device,
                tokenizer_max_length=self.tokenizer_max_length,
                system_message=self.system_message,
                crop_start=self.prompt_template_encode_start_idx,
            )

        if prompt_embeds_2 is None:
            prompt_embeds_2, prompt_embeds_mask_2 = self._get_byt5_prompt_embeds(
                tokenizer=self.tokenizer_2,
                text_encoder=self.text_encoder_2,
                prompt=prompt,
                device=device,
                tokenizer_max_length=self.tokenizer_2_max_length,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_videos_per_prompt, seq_len)

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(batch_size * num_videos_per_prompt, seq_len_2, -1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(batch_size * num_videos_per_prompt, seq_len_2)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype, device=device)
        prompt_embeds_2 = prompt_embeds_2.to(dtype=dtype, device=device)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    def check_inputs(
        self,
        prompt,
        image: PIL.Image.Image,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        prompt_embeds_2=None,
        prompt_embeds_mask_2=None,
        negative_prompt_embeds_2=None,
        negative_prompt_embeds_mask_2=None,
    ):
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `PIL.Image.Image` but is {type(image)}")

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

    # Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.HunyuanVideo15Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_cond_latents_and_mask(
        self,
        latents: torch.Tensor,
        image: PIL.Image.Image,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Prepare conditional latents and mask for t2v generation.

        Args:
            latents: Main latents tensor (B, C, F, H, W)

        Returns:
            tuple: (cond_latents_concat, mask_concat) - both are zero tensors for t2v
        """

        batch, channels, frames, height, width = latents.shape

        image_latents = self._get_image_latents(
            vae=self.vae,
            image_processor=self.video_processor,
            image=image,
            height=height,
            width=width,
            device=device,
        )

        latent_condition = image_latents.repeat(batch_size, 1, frames, 1, 1)
        latent_condition[:, :, 1:, :, :] = 0
        latent_condition = latent_condition.to(device=device, dtype=dtype)

        latent_mask = torch.zeros(batch, 1, frames, height, width, dtype=dtype, device=device)
        latent_mask[:, :, 0, :, :] = 1.0

        return latent_condition, latent_mask

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
        image: PIL.Image.Image,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
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
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image`):
                The input image to condition video generation on.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead.
            num_frames (`int`, defaults to `121`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated mask for prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated mask for negative prompt embeddings.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated text embeddings from the second text encoder. Can be used to easily tweak text inputs.
            prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated mask for prompt embeddings from the second text encoder.
            negative_prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings from the second text encoder.
            negative_prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated mask for negative prompt embeddings from the second text encoder.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between "np", "pt", or "latent".
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideo15PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~HunyuanVideo15PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideo15PipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated videos.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
        )

        height, width = self.video_processor.calculate_default_height_width(
            height=image.size[1], width=image.size[0], target_size=self.target_size
        )
        image = self.video_processor.resize(image, height=height, width=width, resize_mode="crop")

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode image
        image_embeds = self.encode_image(
            image=image,
            batch_size=batch_size * num_videos_per_prompt,
            device=device,
            dtype=self.transformer.dtype,
        )

        # 4. Encode input prompt
        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=self.transformer.dtype,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
        )

        if self.guider._enabled and self.guider.num_conditions > 1:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                device=device,
                dtype=self.transformer.dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_embeds_2=negative_prompt_embeds_2,
                prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
            )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=self.transformer.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        cond_latents_concat, mask_concat = self.prepare_cond_latents_and_mask(
            latents=latents,
            image=image,
            batch_size=batch_size * num_videos_per_prompt,
            height=height,
            width=width,
            dtype=self.transformer.dtype,
            device=device,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, cond_latents_concat, mask_concat], dim=1)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

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
                self.guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)

                # Step 3: Prepare batched model inputs based on the guidance method
                # The guider splits model inputs into separate batches for conditional/unconditional predictions.
                # For CFG with guider_inputs = {"encoder_hidden_states": (prompt_embeds, negative_prompt_embeds)}:
                # you will get a guider_state with two batches:
                #   guider_state = [
                #       {"encoder_hidden_states": prompt_embeds, "__guidance_identifier__": "pred_cond"},      # conditional batch
                #       {"encoder_hidden_states": negative_prompt_embeds, "__guidance_identifier__": "pred_uncond"},  # unconditional batch
                #   ]
                # Other guidance methods may return 1 batch (no guidance) or 3+ batches (e.g., PAG, APG).
                guider_state = self.guider.prepare_inputs(guider_inputs)
                # Step 4: Run the denoiser for each batch
                # Each batch in guider_state represents a different conditioning (conditional, unconditional, etc.).
                # We run the model once per batch and store the noise prediction in guider_state_batch.noise_pred.
                for guider_state_batch in guider_state:
                    self.guider.prepare_models(self.transformer)

                    # Extract conditioning kwargs for this batch (e.g., encoder_hidden_states)
                    cond_kwargs = {
                        input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()
                    }

                    # e.g. "pred_cond"/"pred_uncond"
                    context_name = getattr(guider_state_batch, self.guider._identifier_key)
                    with self.transformer.cache_context(context_name):
                        # Run denoiser and store noise prediction in this batch
                        guider_state_batch.noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            image_embeds=image_embeds,
                            timestep=timestep,
                            timestep_r=timestep_r,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                            **cond_kwargs,
                        )[0]

                    # Cleanup model (e.g., remove hooks)
                    self.guider.cleanup_models(self.transformer)

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
                noise_pred = self.guider(guider_state)[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideo15PipelineOutput(frames=video)
