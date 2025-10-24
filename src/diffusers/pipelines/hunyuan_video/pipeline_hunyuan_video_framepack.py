# Copyright 2025 The Framepack Team, The HunyuanVideo Team and The HuggingFace Team. All rights reserved.
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
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import HunyuanVideoLoraLoaderMixin
from ...models import AutoencoderKLHunyuanVideo, HunyuanVideoFramepackTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HunyuanVideoFramepackPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO(yiyi): We can pack the checkpoints nicely with modular loader
EXAMPLE_DOC_STRING = """
    Examples:
        ##### Image-to-Video

        ```python
        >>> import torch
        >>> from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import SiglipImageProcessor, SiglipVisionModel

        >>> transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        ...     "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        ... )
        >>> feature_extractor = SiglipImageProcessor.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        ... )
        >>> image_encoder = SiglipVisionModel.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
        ... )
        >>> pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        ...     "hunyuanvideo-community/HunyuanVideo",
        ...     transformer=transformer,
        ...     feature_extractor=feature_extractor,
        ...     image_encoder=image_encoder,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.vae.enable_tiling()
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
        ... )
        >>> output = pipe(
        ...     image=image,
        ...     prompt="A penguin dancing in the snow",
        ...     height=832,
        ...     width=480,
        ...     num_frames=91,
        ...     num_inference_steps=30,
        ...     guidance_scale=9.0,
        ...     generator=torch.Generator().manual_seed(0),
        ...     sampling_type="inverted_anti_drifting",
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=30)
        ```

        ##### First and Last Image-to-Video

        ```python
        >>> import torch
        >>> from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import SiglipImageProcessor, SiglipVisionModel

        >>> transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        ...     "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        ... )
        >>> feature_extractor = SiglipImageProcessor.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        ... )
        >>> image_encoder = SiglipVisionModel.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
        ... )
        >>> pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        ...     "hunyuanvideo-community/HunyuanVideo",
        ...     transformer=transformer,
        ...     feature_extractor=feature_extractor,
        ...     image_encoder=image_encoder,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.to("cuda")

        >>> prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
        >>> first_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
        ... )
        >>> last_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
        ... )
        >>> output = pipe(
        ...     image=first_image,
        ...     last_image=last_image,
        ...     prompt=prompt,
        ...     height=512,
        ...     width=512,
        ...     num_frames=91,
        ...     num_inference_steps=30,
        ...     guidance_scale=9.0,
        ...     generator=torch.Generator().manual_seed(0),
        ...     sampling_type="inverted_anti_drifting",
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=30)
        ```
"""


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


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


class FramepackSamplingType(str, Enum):
    VANILLA = "vanilla"
    INVERTED_ANTI_DRIFTING = "inverted_anti_drifting"


class HunyuanVideoFramepackPipeline(DiffusionPipeline, HunyuanVideoLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`LlamaModel`]):
            [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
        tokenizer (`LlamaTokenizer`):
            Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
        transformer ([`HunyuanVideoTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLHunyuanVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder_2 ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer_2 (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        text_encoder: LlamaModel,
        tokenizer: LlamaTokenizerFast,
        transformer: HunyuanVideoFramepackTransformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
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
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video.HunyuanVideoPipeline._get_llama_prompt_embeds
    def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2

        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_videos_per_prompt)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video.HunyuanVideoPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False).pooler_output

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video.HunyuanVideoPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
    ):
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )

        if pooled_prompt_embeds is None:
            if prompt_2 is None:
                prompt_2 = prompt
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    def encode_image(
        self, image: torch.Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        device = device or self._execution_device
        image = (image + 1) / 2.0  # [-1, 1] -> [0, 1]
        image = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False).to(
            device=device, dtype=self.image_encoder.dtype
        )
        image_embeds = self.image_encoder(**image).last_hidden_state
        return image_embeds.to(dtype=dtype)

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_template=None,
        image=None,
        image_latents=None,
        last_image=None,
        last_image_latents=None,
        sampling_type=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

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
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_template is not None:
            if not isinstance(prompt_template, dict):
                raise ValueError(f"`prompt_template` has to be of type `dict` but is {type(prompt_template)}")
            if "template" not in prompt_template:
                raise ValueError(
                    f"`prompt_template` has to contain a key `template` but only found {prompt_template.keys()}"
                )

        sampling_types = [x.value for x in FramepackSamplingType.__members__.values()]
        if sampling_type not in sampling_types:
            raise ValueError(f"`sampling_type` has to be one of '{sampling_types}' but is '{sampling_type}'")

        if image is not None and image_latents is not None:
            raise ValueError("Only one of `image` or `image_latents` can be passed.")
        if last_image is not None and last_image_latents is not None:
            raise ValueError("Only one of `last_image` or `last_image_latents` can be passed.")
        if sampling_type != FramepackSamplingType.INVERTED_ANTI_DRIFTING and (
            last_image is not None or last_image_latents is not None
        ):
            raise ValueError(
                'Only `"inverted_anti_drifting"` inference type supports `last_image` or `last_image_latents`.'
            )

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
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

    def prepare_image_latents(
        self,
        image: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        if latents is None:
            image = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
            latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            latents = latents * self.vae.config.scaling_factor
        return latents.to(device=device, dtype=dtype)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    @property
    def guidance_scale(self):
        return self._guidance_scale

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
        image: PipelineImageInput,
        last_image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        latent_window_size: int = 9,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        image_latents: Optional[torch.Tensor] = None,
        last_image_latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        sampling_type: FramepackSamplingType = FramepackSamplingType.INVERTED_ANTI_DRIFTING,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to be used as the starting point for the video generation.
            last_image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`, *optional*):
                The optional last image to be used as the ending point for the video generation. This is useful for
                generating transitions between two images.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            height (`int`, defaults to `720`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `129`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            guidance_scale (`float`, defaults to `6.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. Note that the only available
                HunyuanVideo model is CFG-distilled, which means that traditional guidance between unconditional and
                conditional latent is not applied.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            image_latents (`torch.Tensor`, *optional*):
                Pre-encoded image latents. If not provided, the image will be encoded using the VAE.
            last_image_latents (`torch.Tensor`, *optional*):
                Pre-encoded last image latents. If not provided, the last image will be encoded using the VAE.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoFramepackPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoFramepackPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoFramepackPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images and the second element is a list
                of `bool`s indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw)
                content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
            image,
            image_latents,
            last_image,
            last_image_latents,
            sampling_type,
        )

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        transformer_dtype = self.transformer.dtype
        vae_dtype = self.vae.dtype

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        transformer_dtype = self.transformer.dtype
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prompt_attention_mask=negative_prompt_attention_mask,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(transformer_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dtype)

        # 4. Prepare image
        image = self.video_processor.preprocess(image, height, width)
        image_embeds = self.encode_image(image, device=device).to(transformer_dtype)
        if last_image is not None:
            # Credits: https://github.com/lllyasviel/FramePack/pull/167
            # Users can modify the weighting strategy applied here
            last_image = self.video_processor.preprocess(last_image, height, width)
            last_image_embeds = self.encode_image(last_image, device=device).to(transformer_dtype)
            last_image_embeds = (image_embeds + last_image_embeds) / 2

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
        num_latent_sections = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        history_video = None
        total_generated_latent_frames = 0

        image_latents = self.prepare_image_latents(
            image, dtype=torch.float32, device=device, generator=generator, latents=image_latents
        )
        if last_image is not None:
            last_image_latents = self.prepare_image_latents(
                last_image, dtype=torch.float32, device=device, generator=generator
            )

        # Specific to the released checkpoints:
        #   - https://huggingface.co/lllyasviel/FramePackI2V_HY
        #   - https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503
        # TODO: find a more generic way in future if there are more checkpoints
        if sampling_type == FramepackSamplingType.INVERTED_ANTI_DRIFTING:
            history_sizes = [1, 2, 16]
            history_latents = torch.zeros(
                batch_size,
                num_channels_latents,
                sum(history_sizes),
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                device=device,
                dtype=torch.float32,
            )

        elif sampling_type == FramepackSamplingType.VANILLA:
            history_sizes = [16, 2, 1]
            history_latents = torch.zeros(
                batch_size,
                num_channels_latents,
                sum(history_sizes),
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                device=device,
                dtype=torch.float32,
            )
            history_latents = torch.cat([history_latents, image_latents], dim=2)
            total_generated_latent_frames += 1

        else:
            assert False

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * batch_size, dtype=transformer_dtype, device=device) * 1000.0

        # 7. Denoising loop
        for k in range(num_latent_sections):
            if sampling_type == FramepackSamplingType.INVERTED_ANTI_DRIFTING:
                latent_paddings = list(reversed(range(num_latent_sections)))
                if num_latent_sections > 4:
                    latent_paddings = [3] + [2] * (num_latent_sections - 3) + [1, 0]

                is_first_section = k == 0
                is_last_section = k == num_latent_sections - 1
                latent_padding_size = latent_paddings[k] * latent_window_size

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, *history_sizes]))
                (
                    indices_prefix,
                    indices_padding,
                    indices_latents,
                    indices_latents_history_1x,
                    indices_latents_history_2x,
                    indices_latents_history_4x,
                ) = indices.split([1, latent_padding_size, latent_window_size, *history_sizes], dim=0)
                # Inverted anti-drifting sampling: Figure 2(c) in the paper
                indices_clean_latents = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                latents_prefix = image_latents
                latents_history_1x, latents_history_2x, latents_history_4x = history_latents[
                    :, :, : sum(history_sizes)
                ].split(history_sizes, dim=2)
                if last_image is not None and is_first_section:
                    latents_history_1x = last_image_latents
                latents_clean = torch.cat([latents_prefix, latents_history_1x], dim=2)

            elif sampling_type == FramepackSamplingType.VANILLA:
                indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                (
                    indices_prefix,
                    indices_latents_history_4x,
                    indices_latents_history_2x,
                    indices_latents_history_1x,
                    indices_latents,
                ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                indices_clean_latents = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                latents_prefix = image_latents
                latents_history_4x, latents_history_2x, latents_history_1x = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)
                latents_clean = torch.cat([latents_prefix, latents_history_1x], dim=2)

            else:
                assert False

            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
            image_seq_len = (
                latents.shape[2] * latents.shape[3] * latents.shape[4] / self.transformer.config.patch_size**2
            )
            exp_max = 7.0
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            mu = min(mu, math.log(exp_max))
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
            )
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    timestep = t.expand(latents.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        image_embeds=image_embeds,
                        indices_latents=indices_latents,
                        guidance=guidance,
                        latents_clean=latents_clean.to(transformer_dtype),
                        indices_latents_clean=indices_clean_latents,
                        latents_history_2x=latents_history_2x.to(transformer_dtype),
                        indices_latents_history_2x=indices_latents_history_2x,
                        latents_history_4x=latents_history_4x.to(transformer_dtype),
                        indices_latents_history_4x=indices_latents_history_4x,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                    if do_true_cfg:
                        neg_noise_pred = self.transformer(
                            hidden_states=latents.to(transformer_dtype),
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_attention_mask=negative_prompt_attention_mask,
                            pooled_projections=negative_pooled_prompt_embeds,
                            image_embeds=image_embeds,
                            indices_latents=indices_latents,
                            guidance=guidance,
                            latents_clean=latents_clean.to(transformer_dtype),
                            indices_latents_clean=indices_clean_latents,
                            latents_history_2x=latents_history_2x.to(transformer_dtype),
                            indices_latents_history_2x=indices_latents_history_2x,
                            latents_history_4x=latents_history_4x.to(transformer_dtype),
                            indices_latents_history_4x=indices_latents_history_4x,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]

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

                if sampling_type == FramepackSamplingType.INVERTED_ANTI_DRIFTING:
                    if is_last_section:
                        latents = torch.cat([image_latents, latents], dim=2)
                    total_generated_latent_frames += latents.shape[2]
                    history_latents = torch.cat([latents, history_latents], dim=2)
                    real_history_latents = history_latents[:, :, :total_generated_latent_frames]
                    section_latent_frames = (
                        (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    )
                    index_slice = (slice(None), slice(None), slice(0, section_latent_frames))

                elif sampling_type == FramepackSamplingType.VANILLA:
                    total_generated_latent_frames += latents.shape[2]
                    history_latents = torch.cat([history_latents, latents], dim=2)
                    real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
                    section_latent_frames = latent_window_size * 2
                    index_slice = (slice(None), slice(None), slice(-section_latent_frames, None))

                else:
                    assert False

                if history_video is None:
                    if not output_type == "latent":
                        current_latents = real_history_latents.to(vae_dtype) / self.vae.config.scaling_factor
                        history_video = self.vae.decode(current_latents, return_dict=False)[0]
                    else:
                        history_video = [real_history_latents]
                else:
                    if not output_type == "latent":
                        overlapped_frames = (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
                        current_latents = (
                            real_history_latents[index_slice].to(vae_dtype) / self.vae.config.scaling_factor
                        )
                        current_video = self.vae.decode(current_latents, return_dict=False)[0]

                        if sampling_type == FramepackSamplingType.INVERTED_ANTI_DRIFTING:
                            history_video = self._soft_append(current_video, history_video, overlapped_frames)
                        elif sampling_type == FramepackSamplingType.VANILLA:
                            history_video = self._soft_append(history_video, current_video, overlapped_frames)
                        else:
                            assert False
                    else:
                        history_video.append(real_history_latents)

        self._current_timestep = None

        if not output_type == "latent":
            generated_frames = history_video.size(2)
            generated_frames = (
                generated_frames - 1
            ) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            history_video = history_video[:, :, :generated_frames]
            video = self.video_processor.postprocess_video(history_video, output_type=output_type)
        else:
            video = history_video

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoFramepackPipelineOutput(frames=video)

    def _soft_append(self, history: torch.Tensor, current: torch.Tensor, overlap: int = 0):
        if overlap <= 0:
            return torch.cat([history, current], dim=2)

        assert history.shape[2] >= overlap, f"Current length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert current.shape[2] >= overlap, f"History length ({current.shape[2]}) must be >= overlap ({overlap})"

        weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
        blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

        return output.to(history)
