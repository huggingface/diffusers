# Copyright 2025 HiDream-ai Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    T5EncoderModel,
    T5Tokenizer,
)

from ...image_processor import VaeImageProcessor
from ...loaders import HiDreamImageLoraLoaderMixin
from ...models import AutoencoderKL, HiDreamImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from ...utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HiDreamImagePipelineOutput


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
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        >>> from diffusers import HiDreamImagePipeline


        >>> tokenizer_4 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        >>> text_encoder_4 = LlamaForCausalLM.from_pretrained(
        ...     "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ...     output_hidden_states=True,
        ...     output_attentions=True,
        ...     torch_dtype=torch.bfloat16,
        ... )

        >>> pipe = HiDreamImagePipeline.from_pretrained(
        ...     "HiDream-ai/HiDream-I1-Full",
        ...     tokenizer_4=tokenizer_4,
        ...     text_encoder_4=text_encoder_4,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> image = pipe(
        ...     'A cat holding a sign that says "Hi-Dreams.ai".',
        ...     height=1024,
        ...     width=1024,
        ...     guidance_scale=5.0,
        ...     num_inference_steps=50,
        ...     generator=torch.Generator("cuda").manual_seed(0),
        ... ).images[0]
        >>> image.save("output.png")
        ```
"""


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


class HiDreamImagePipeline(DiffusionPipeline, HiDreamImageLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds_t5", "prompt_embeds_llama3", "pooled_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5Tokenizer,
        text_encoder_4: LlamaForCausalLM,
        tokenizer_4: PreTrainedTokenizerFast,
        transformer: HiDreamImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            text_encoder_4=text_encoder_4,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            tokenizer_4=tokenizer_4,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # HiDreamImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128
        if getattr(self, "tokenizer_4", None) is not None:
            self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_3.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_3.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(
                untruncated_ids[:, min(max_sequence_length, self.tokenizer_3.model_max_length) - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_3.model_max_length)} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device), attention_mask=attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, 218),
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, 218 - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {218} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def _get_llama3_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_4.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_4.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_4(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_4.batch_decode(
                untruncated_ids[:, min(max_sequence_length, self.tokenizer_4.model_max_length) - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_4.model_max_length)} tokens: {removed_text}"
            )

        outputs = self.text_encoder_4(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=True,
        )

        prompt_embeds = outputs.hidden_states[1:]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        prompt_embeds_t5: Optional[List[torch.FloatTensor]] = None,
        prompt_embeds_llama3: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds_t5: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds_llama3: Optional[List[torch.FloatTensor]] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
        lora_scale: Optional[float] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = pooled_prompt_embeds.shape[0]

        device = device or self._execution_device

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, prompt, max_sequence_length, device, dtype
            )

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if len(negative_prompt) > 1 and len(negative_prompt) != batch_size:
                raise ValueError(f"negative_prompt must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, negative_prompt, max_sequence_length, device, dtype
            )

            if negative_pooled_prompt_embeds_1.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_1 = negative_pooled_prompt_embeds_1.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            if len(prompt_2) > 1 and len(prompt_2) != batch_size:
                raise ValueError(f"prompt_2 must be of length 1 or {batch_size}")

            pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, prompt_2, max_sequence_length, device, dtype
            )

            if pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                pooled_prompt_embeds_2 = pooled_prompt_embeds_2.repeat(batch_size, 1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2

            if len(negative_prompt_2) > 1 and len(negative_prompt_2) != batch_size:
                raise ValueError(f"negative_prompt_2 must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, negative_prompt_2, max_sequence_length, device, dtype
            )

            if negative_pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_2 = negative_pooled_prompt_embeds_2.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1
            )

        if prompt_embeds_t5 is None:
            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            if len(prompt_3) > 1 and len(prompt_3) != batch_size:
                raise ValueError(f"prompt_3 must be of length 1 or {batch_size}")

            prompt_embeds_t5 = self._get_t5_prompt_embeds(prompt_3, max_sequence_length, device, dtype)

            if prompt_embeds_t5.shape[0] == 1 and batch_size > 1:
                prompt_embeds_t5 = prompt_embeds_t5.repeat(batch_size, 1, 1)

        if do_classifier_free_guidance and negative_prompt_embeds_t5 is None:
            negative_prompt_3 = negative_prompt_3 or negative_prompt
            negative_prompt_3 = [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3

            if len(negative_prompt_3) > 1 and len(negative_prompt_3) != batch_size:
                raise ValueError(f"negative_prompt_3 must be of length 1 or {batch_size}")

            negative_prompt_embeds_t5 = self._get_t5_prompt_embeds(
                negative_prompt_3, max_sequence_length, device, dtype
            )

            if negative_prompt_embeds_t5.shape[0] == 1 and batch_size > 1:
                negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(batch_size, 1, 1)

        if prompt_embeds_llama3 is None:
            prompt_4 = prompt_4 or prompt
            prompt_4 = [prompt_4] if isinstance(prompt_4, str) else prompt_4

            if len(prompt_4) > 1 and len(prompt_4) != batch_size:
                raise ValueError(f"prompt_4 must be of length 1 or {batch_size}")

            prompt_embeds_llama3 = self._get_llama3_prompt_embeds(prompt_4, max_sequence_length, device, dtype)

            if prompt_embeds_llama3.shape[0] == 1 and batch_size > 1:
                prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, batch_size, 1, 1)

        if do_classifier_free_guidance and negative_prompt_embeds_llama3 is None:
            negative_prompt_4 = negative_prompt_4 or negative_prompt
            negative_prompt_4 = [negative_prompt_4] if isinstance(negative_prompt_4, str) else negative_prompt_4

            if len(negative_prompt_4) > 1 and len(negative_prompt_4) != batch_size:
                raise ValueError(f"negative_prompt_4 must be of length 1 or {batch_size}")

            negative_prompt_embeds_llama3 = self._get_llama3_prompt_embeds(
                negative_prompt_4, max_sequence_length, device, dtype
            )

            if negative_prompt_embeds_llama3.shape[0] == 1 and batch_size > 1:
                negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, batch_size, 1, 1)

        # duplicate pooled_prompt_embeds for each generation per prompt
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # duplicate t5_prompt_embeds for batch_size and num_images_per_prompt
        bs_embed, seq_len, _ = prompt_embeds_t5.shape
        if bs_embed == 1 and batch_size > 1:
            prompt_embeds_t5 = prompt_embeds_t5.repeat(batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate prompt_embeds_t5 of batch size {bs_embed}")
        prompt_embeds_t5 = prompt_embeds_t5.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_t5 = prompt_embeds_t5.view(batch_size * num_images_per_prompt, seq_len, -1)

        # duplicate llama3_prompt_embeds for batch_size and num_images_per_prompt
        _, bs_embed, seq_len, dim = prompt_embeds_llama3.shape
        if bs_embed == 1 and batch_size > 1:
            prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate prompt_embeds_llama3 of batch size {bs_embed}")
        prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, 1, num_images_per_prompt, 1)
        prompt_embeds_llama3 = prompt_embeds_llama3.view(-1, batch_size * num_images_per_prompt, seq_len, dim)

        if do_classifier_free_guidance:
            # duplicate negative_pooled_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len = negative_pooled_prompt_embeds.shape
            if bs_embed == 1 and batch_size > 1:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_pooled_prompt_embeds of batch size {bs_embed}")
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

            # duplicate negative_t5_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len, _ = negative_prompt_embeds_t5.shape
            if bs_embed == 1 and batch_size > 1:
                negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_prompt_embeds_t5 of batch size {bs_embed}")
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.view(batch_size * num_images_per_prompt, seq_len, -1)

            # duplicate negative_prompt_embeds_llama3 for batch_size and num_images_per_prompt
            _, bs_embed, seq_len, dim = negative_prompt_embeds_llama3.shape
            if bs_embed == 1 and batch_size > 1:
                negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_prompt_embeds_llama3 of batch size {bs_embed}")
            negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, 1, num_images_per_prompt, 1)
            negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.view(
                -1, batch_size * num_images_per_prompt, seq_len, dim
            )

        return (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

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

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        prompt_4,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        negative_prompt_4=None,
        prompt_embeds_t5=None,
        prompt_embeds_llama3=None,
        negative_prompt_embeds_t5=None,
        negative_prompt_embeds_llama3=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `pooled_prompt_embeds`: {pooled_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `pooled_prompt_embeds`: {pooled_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds_t5 is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_3} and `prompt_embeds_t5`: {prompt_embeds_t5}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_4 is not None and prompt_embeds_llama3 is not None:
            raise ValueError(
                f"Cannot forward both `prompt_4`: {prompt_4} and `prompt_embeds_llama3`: {prompt_embeds_llama3}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and pooled_prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `pooled_prompt_embeds`. Cannot leave both `prompt` and `pooled_prompt_embeds` undefined."
            )
        elif prompt is None and prompt_embeds_t5 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_t5`. Cannot leave both `prompt` and `prompt_embeds_t5` undefined."
            )
        elif prompt is None and prompt_embeds_llama3 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_llama3`. Cannot leave both `prompt` and `prompt_embeds_llama3` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")
        elif prompt_4 is not None and (not isinstance(prompt_4, str) and not isinstance(prompt_4, list)):
            raise ValueError(f"`prompt_4` has to be of type `str` or `list` but is {type(prompt_4)}")

        if negative_prompt is not None and negative_pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_pooled_prompt_embeds`:"
                f" {negative_pooled_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_pooled_prompt_embeds`:"
                f" {negative_pooled_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds_t5 is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds_t5`:"
                f" {negative_prompt_embeds_t5}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_4 is not None and negative_prompt_embeds_llama3 is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_4`: {negative_prompt_4} and `negative_prompt_embeds_llama3`:"
                f" {negative_prompt_embeds_llama3}. Please make sure to only forward one of the two."
            )

        if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is not None:
            if pooled_prompt_embeds.shape != negative_pooled_prompt_embeds.shape:
                raise ValueError(
                    "`pooled_prompt_embeds` and `negative_pooled_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `pooled_prompt_embeds` {pooled_prompt_embeds.shape} != `negative_pooled_prompt_embeds`"
                    f" {negative_pooled_prompt_embeds.shape}."
                )
        if prompt_embeds_t5 is not None and negative_prompt_embeds_t5 is not None:
            if prompt_embeds_t5.shape != negative_prompt_embeds_t5.shape:
                raise ValueError(
                    "`prompt_embeds_t5` and `negative_prompt_embeds_t5` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_t5` {prompt_embeds_t5.shape} != `negative_prompt_embeds_t5`"
                    f" {negative_prompt_embeds_t5.shape}."
                )
        if prompt_embeds_llama3 is not None and negative_prompt_embeds_llama3 is not None:
            if prompt_embeds_llama3.shape != negative_prompt_embeds_llama3.shape:
                raise ValueError(
                    "`prompt_embeds_llama3` and `negative_prompt_embeds_llama3` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_llama3` {prompt_embeds_llama3.shape} != `negative_prompt_embeds_llama3`"
                    f" {negative_prompt_embeds_llama3.shape}."
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
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds_t5: Optional[torch.FloatTensor] = None,
        prompt_embeds_llama3: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds_t5: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds_llama3: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead.
            prompt_4 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_4` and `text_encoder_4`. If not defined, `prompt` is
                will be used instead.
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
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_4 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_4` and
                `text_encoder_4`. If not defined, `negative_prompt` is used in all the text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int` defaults to 128): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.hidream_image.HiDreamImagePipelineOutput`] or `tuple`:
            [`~pipelines.hidream_image.HiDreamImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated. images.
        """

        prompt_embeds = kwargs.get("prompt_embeds", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)

        if prompt_embeds is not None:
            deprecation_message = "The `prompt_embeds` argument is deprecated. Please use `prompt_embeds_t5` and `prompt_embeds_llama3` instead."
            deprecate("prompt_embeds", "0.35.0", deprecation_message)
            prompt_embeds_t5 = prompt_embeds[0]
            prompt_embeds_llama3 = prompt_embeds[1]

        if negative_prompt_embeds is not None:
            deprecation_message = "The `negative_prompt_embeds` argument is deprecated. Please use `negative_prompt_embeds_t5` and `negative_prompt_embeds_llama3` instead."
            deprecate("negative_prompt_embeds", "0.35.0", deprecation_message)
            negative_prompt_embeds_t5 = negative_prompt_embeds[0]
            negative_prompt_embeds_llama3 = negative_prompt_embeds[1]

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(height * scale // division * division)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            prompt_embeds_t5=prompt_embeds_t5,
            prompt_embeds_llama3=prompt_embeds_llama3,
            negative_prompt_embeds_t5=negative_prompt_embeds_t5,
            negative_prompt_embeds_llama3=negative_prompt_embeds_llama3,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif pooled_prompt_embeds is not None:
            batch_size = pooled_prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompt
        lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None
        (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds_t5=prompt_embeds_t5,
            prompt_embeds_llama3=prompt_embeds_llama3,
            negative_prompt_embeds_t5=negative_prompt_embeds_t5,
            negative_prompt_embeds_llama3=negative_prompt_embeds_llama3,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, prompt_embeds_t5], dim=0)
            prompt_embeds_llama3 = torch.cat([negative_prompt_embeds_llama3, prompt_embeds_llama3], dim=1)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pooled_prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        mu = calculate_shift(self.transformer.max_seq)
        scheduler_kwargs = {"mu": mu}
        if isinstance(self.scheduler, UniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device)  # , shift=math.exp(mu))
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    encoder_hidden_states_t5=prompt_embeds_t5,
                    encoder_hidden_states_llama3=prompt_embeds_llama3,
                    pooled_embeds=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

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
                    prompt_embeds_t5 = callback_outputs.pop("prompt_embeds_t5", prompt_embeds_t5)
                    prompt_embeds_llama3 = callback_outputs.pop("prompt_embeds_llama3", prompt_embeds_llama3)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return HiDreamImagePipelineOutput(images=image)
