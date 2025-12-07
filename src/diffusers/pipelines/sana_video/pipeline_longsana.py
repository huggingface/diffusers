# Copyright 2025 SANA-Video Authors and The HuggingFace Team. All rights reserved.
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

import html
import inspect
import re
import urllib.parse as ul
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import SanaLoraLoaderMixin
from ...models import AutoencoderDC, AutoencoderKLWan
from ...models.transformers.transformer_sana_video_causal import SanaVideoCausalTransformer3DModel, SanaBlockKvCache
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    BACKENDS_MAPPING,
    USE_PEFT_BACKEND,
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import get_device, is_torch_version, randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import SanaVideoPipelineOutput
from .pipeline_sana_video import ASPECT_RATIO_480_BIN, ASPECT_RATIO_720_BIN


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import SanaVideoPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = SanaVideoPipeline.from_pretrained("Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
        >>> pipe.transformer.to(torch.bfloat16)
        >>> pipe.text_encoder.to(torch.bfloat16)
        >>> pipe.vae.to(torch.float32)
        >>> pipe.to("cuda")
        >>> motion_score = 30

        >>> prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
        >>> negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
        >>> motion_prompt = f" motion score: {motion_score}."
        >>> prompt = prompt + motion_prompt

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=480,
        ...     width=832,
        ...     frames=81,
        ...     guidance_scale=6,
        ...     num_inference_steps=50,
        ...     generator=torch.Generator(device="cuda").manual_seed(42),
        ... ).frames[0]

        >>> export_to_video(output, "sana-video-output.mp4", fps=16)
        ```
"""


class LongSanaKvCache:
    def __init__(self, num_chunks: int, num_blocks: int):
        """
        Initialize KV cache for all chunks.
        
        Args:
            num_chunks: Number of chunks
            num_blocks: Number of transformer blocks
            
        Returns:
            List of KV cache for each chunk
        """
        kv_caches = []
        for _ in range(num_chunks):
            kv_caches.append([SanaBlockKvCache(vk=None, k_sum=None, temporal_cache=None) for _ in range(num_blocks)])
        self.num_chunks = num_chunks
        self.num_blocks = num_blocks
        self.kv_caches = kv_caches
    
    def get_chunk_cache(self, chunk_idx: int) -> List[SanaBlockKvCache]:
        return self.kv_caches[chunk_idx]
    
    def get_block_cache(self, chunk_idx: int, block_idx: int) -> SanaBlockKvCache:
        return self.kv_caches[chunk_idx][block_idx]

    def update_chunk_cache(self, chunk_idx: int, chunk_kv_cache: List[SanaBlockKvCache]):
        self.kv_caches[chunk_idx] = chunk_kv_cache

    def get_accumulated_chunk_cache(self, chunk_idx: int, num_cached_blocks: int = -1) -> List[SanaBlockKvCache]:
        """
        Accumulate KV cache from previous chunks.
        
        Args:
            chunk_idx: Current chunk index
            num_cached_blocks: Number of previous chunks to use for accumulation. -1 means use all previous chunks.
            
        Returns:
            Accumulated KV cache for current chunk, a list of SanaBlockKvCache.
        """
        if chunk_idx == 0:
            return self.kv_caches[0]

        accumulated_kv_caches = [] # a list of SanaBlockKvCache
        for block_id in range(self.num_blocks):

            start_chunk_idx = chunk_idx - num_cached_blocks if num_cached_blocks > 0 else 0
            # Initialize accumulated block cache, kv, k_sum, temporal cache are all None.
            acc_block_cache = SanaBlockKvCache(vk=None, k_sum=None, temporal_cache=None)
            # Accumulate spatial KV cache from previous chunks

            for prev_chunk_idx in range(start_chunk_idx, chunk_idx):
                prev_kv_cache = self.kv_caches[prev_chunk_idx][block_id]

                if prev_kv_cache.vk is None or prev_kv_cache.k_sum is None:
                    continue

                if acc_block_cache.vk is not None and acc_block_cache.k_sum is not None:
                    acc_block_cache.vk += prev_kv_cache.vk
                    acc_block_cache.k_sum += prev_kv_cache.k_sum
                else:
                    # initialize the vk and k_sum using the first chunk's block cache.
                    acc_block_cache.vk = prev_kv_cache.vk.clone()
                    acc_block_cache.k_sum = prev_kv_cache.k_sum.clone()
            # copy the temporal cache from the previous chunk.
            acc_block_cache.temporal_cache = self.kv_caches[chunk_idx-1][block_id].temporal_cache

            accumulated_kv_caches.append(acc_block_cache)

        return accumulated_kv_caches
    

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


class LongSanaVideoPipeline(DiffusionPipeline, SanaLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using [Sana](https://huggingface.co/papers/2509.24695). This model inherits
    from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods implemented for all
    pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`GemmaTokenizer`] or [`GemmaTokenizerFast`]):
            The tokenizer used to tokenize the prompt.
        text_encoder ([`Gemma2PreTrainedModel`]):
            Text encoder model to encode the input prompts.
        vae ([`AutoencoderKLWan` or `AutoencoderDCAEV`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer ([`SanaVideoCausalTransformer3DModel`]):
            Conditional Transformer with KV cache support to denoise the input latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A flow matching scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        base_chunk_frames (`int`, defaults to 10):
            Number of frames per chunk for autoregressive generation.
        num_cached_blocks (`int`, defaults to -1):
            Number of previous chunks to use for KV cache accumulation. -1 means use all previous chunks.
    """

    # fmt: off
    bad_punct_regex = re.compile(r"[" + "#®•©™&@·º½¾¿¡§~" + r"\)" + r"\(" + r"\]" + r"\[" + r"\}" + r"\{" + r"\|" + "\\" + r"\/" + r"\*" + r"]{1,}")
    # fmt: on

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        text_encoder: Gemma2PreTrainedModel,
        vae: Union[AutoencoderDC, AutoencoderKLWan],
        transformer: SanaVideoCausalTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        base_chunk_frames: int = 10,
        num_cached_blocks: int = -1,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self.vae_scale_factor = self.vae_scale_factor_spatial

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # LongSana specific parameters
        self.base_chunk_frames = base_chunk_frames
        self.num_cached_blocks = num_cached_blocks

    # Copied from diffusers.pipelines.sana.pipeline_sana.SanaPipeline._get_gemma_prompt_embeds
    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        dtype: torch.dtype,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
            complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`):
                If `complex_human_instruction` is not empty, the function will use the complex Human instruction for
                the prompt.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)

        # prepare complex human instruction
        if not complex_human_instruction:
            max_length_all = max_sequence_length
        else:
            chi_prompt = "\n".join(complex_human_instruction)
            prompt = [chi_prompt + p for p in prompt]
            num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + max_sequence_length - 2

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds = prompt_embeds[0].to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                number of videos that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Sana, it's should be the embeddings of the "" string.
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
            complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`):
                If `complex_human_instruction` is not empty, the function will use the complex Human instruction for
                the prompt.
        """

        if device is None:
            device = self._execution_device

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = None

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SanaLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        # See Section 3.1. of the paper.
        max_length = max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=complex_human_instruction,
            )

            prompt_embeds = prompt_embeds[:, select_index]
            prompt_attention_mask = prompt_attention_mask[:, select_index]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=False,
            )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_videos_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        if self.text_encoder is not None:
            if isinstance(self, SanaLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

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

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip addresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching prediction to x0 prediction.
        For flow matching: x_0 = x_t - sigma_t * flow_pred
        
        Args:
            flow_pred: Flow prediction from the model with shape [B*T, C, H, W] or [B, C, H, W]
            xt: Noisy latent at timestep t with same shape as flow_pred
            timestep: Current timestep with shape [B] (scalar timestep value, not batch*frames)
            
        Returns:
            Predicted clean latent x_0
        """
        original_dtype = flow_pred.dtype
        flow_pred_f64 = flow_pred.double()
        xt_f64 = xt.double()

        # Get sigma_t from scheduler
        sigmas = self.scheduler.sigmas.double().to(flow_pred.device)
        timesteps_sched = self.scheduler.timesteps.double().to(flow_pred.device)

        # Find closest timestep index
        # timestep is scalar or [B], we need to match it against scheduler timesteps
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep_f64 = timestep.double()
        timestep_id = torch.argmin((timesteps_sched.unsqueeze(0) - timestep_f64.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)

        # x_0 = x_t - sigma_t * flow_pred
        x0_pred = xt_f64 - sigma_t * flow_pred_f64

        return x0_pred.to(original_dtype)

    def _create_autoregressive_segments(self, total_frames: int, base_chunk_frames: int) -> List[int]:
        """
        Create autoregressive segments for long video generation.
        
        Args:
            total_frames: Total number of frames to generate
            base_chunk_frames: Base number of frames per chunk
            
        Returns:
            List of frame indices marking chunk boundaries
        """
        remained_frames = total_frames % base_chunk_frames
        num_chunks = total_frames // base_chunk_frames
        chunk_indices = [0]
        for i in range(num_chunks):
            cur_idx = chunk_indices[-1] + base_chunk_frames
            if i == 0:
                cur_idx += remained_frames
            chunk_indices.append(cur_idx)
        if chunk_indices[-1] < total_frames:
            chunk_indices.append(total_frames)
        return chunk_indices


    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

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
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        height: int = 480,
        width: int = 832,
        frames: int = 81,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        denoising_step_list: Optional[List[int]] = [1000, 960, 889, 727],
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        complex_human_instruction: List[str] = [
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
    ) -> Union[SanaVideoPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for multi-step denoising per chunk. For LongSana, if provided, these timesteps
                override `num_inference_steps` and `denoising_step_list`, and will be used as the denoising schedule
                for each chunk. For example: `timesteps=[1000, 960, 889, 727, 0]`. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for multi-step denoising per chunk. Similar to `timesteps`, if provided, these
                sigmas override `num_inference_steps` and `denoising_step_list`. The sigmas will be converted to
                timesteps internally. For example: `sigmas=[1.0, 0.96, 0.889, 0.727, 0.0]`.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to
                the text `prompt`, usually at the expense of lower video quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            height (`int`, *optional*, defaults to 480):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to 832):
                The width in pixels of the generated video.
            frames (`int`, *optional*, defaults to 81):
                The number of frames in the generated video.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
                applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video. Choose between mp4 or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SanaVideoPipelineOutput`] instead of a plain tuple.
            attention_kwargs:
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_480_BIN` or `ASPECT_RATIO_720_BIN`. After the produced latents are decoded into videos,
                they are resized back to the requested resolution. Useful for generating non-square videos.
            denoising_step_list (`List[int]`, *optional*, defaults to `[1000, 960, 889, 727]`):
                Custom list of timesteps for multi-step denoising per chunk. Each chunk will be progressively
                denoised through these timesteps. Multi-step denoising helps improve quality for long videos.
                Note: This parameter is overridden if `timesteps` or `sigmas` are provided.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `300`):
                Maximum sequence length to use with the `prompt`.
            complex_human_instruction (`List[str]`, *optional*):
                Instructions for complex human attention:
                https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

        Examples:

        Returns:
            [`~pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated videos
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if self.transformer.config.sample_size == 30:
                aspect_ratio_bin = ASPECT_RATIO_480_BIN
            elif self.transformer.config.sample_size == 22:
                aspect_ratio_bin = ASPECT_RATIO_720_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.video_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        self.check_inputs(
            prompt,
            height,
            width,
            callback_on_step_end_tensor_inputs,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare timesteps
        # For LongSana, if timesteps are provided, they override num_inference_steps
        # These timesteps will be used as denoising_step_list for multi-step denoising per chunk
        if timesteps is not None or sigmas is not None:
            timesteps_from_user, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas
            )
            # Use these as the denoising step list
            if denoising_step_list is None or denoising_step_list == [1000, 960, 889, 727]:
                denoising_step_list = timesteps_from_user.cpu().tolist()
        else:
            # Standard timesteps for scheduler setup
            timesteps_temp, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, None, None
            )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            height,
            width,
            frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Chunked denoising loop for long video generation
        # Default denoising step list if not provided
        if denoising_step_list is None:
            # Use the standard timesteps from the scheduler
            denoising_step_list = timesteps.cpu().tolist()

        device = latents.device
        batch_size_latents, _, total_frames, height_latent, width_latent = latents.shape

        # Create autoregressive segments
        chunk_indices = self._create_autoregressive_segments(total_frames, self.base_chunk_frames)
        num_chunks = len(chunk_indices) - 1

        # Get number of transformer blocks
        num_blocks = self.transformer.config.num_layers

        # Initialize KV cache for all chunks
        kv_cache = LongSanaKvCache(num_chunks=num_chunks, num_blocks=num_blocks)

        # Output tensor to store denoised results
        output = torch.zeros_like(latents)

        transformer_dtype = self.transformer.dtype

        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]

            # Extract chunk latents
            local_latent = latents[:, :, start_f:end_f].clone()

            # Accumulate KV cache from previous chunks
            chunk_kv_cache = kv_cache.get_accumulated_chunk_cache(chunk_idx, num_cached_blocks=self.num_cached_blocks)
            for block_cache in chunk_kv_cache:
                block_cache.disable_save()

            # Multi-step denoising for this chunk
            with self.progress_bar(total=len(denoising_step_list)) as progress_bar:
                for step_idx, current_timestep in enumerate(denoising_step_list):
                    if self.interrupt:
                        continue

                    # Prepare model input
                    latent_model_input = (
                        torch.cat([local_latent] * 2) if self.do_classifier_free_guidance else local_latent
                    )

                    # Create timestep tensor
                    t = torch.tensor([current_timestep], device=device, dtype=torch.long)
                    timestep = t.expand(latent_model_input.shape[0])


                    # Predict flow
                    flow_pred, _ = self.transformer(
                        latent_model_input.to(dtype=transformer_dtype),
                        encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=timestep,
                        return_dict=False,
                        kv_caches=chunk_kv_cache,
                        attention_kwargs=self.attention_kwargs,
                    )


                    flow_pred = flow_pred.float()

                    # Perform guidance on flow prediction
                    if self.do_classifier_free_guidance:
                        flow_pred_uncond, flow_pred_text = flow_pred.chunk(2)
                        flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_text - flow_pred_uncond)

                    # Handle learned sigma
                    if self.transformer.config.out_channels // 2 == latent_channels:
                        flow_pred = flow_pred.chunk(2, dim=1)[0]

                    # Convert flow prediction to x0 prediction
                    # Need to rearrange dimensions: b c f h w -> b f c h w for conversion
                    flow_pred_bfchw = rearrange(flow_pred, "b c f h w -> b f c h w")
                    local_latent_bfchw = rearrange(local_latent, "b c f h w -> b f c h w")

                    # Convert to x0 (flatten batch and frames for conversion)
                    pred_x0_flat = self._convert_flow_pred_to_x0(
                        flow_pred=flow_pred_bfchw.flatten(0, 1),
                        xt=local_latent_bfchw.flatten(0, 1),
                        timestep=t
                    )
                    pred_x0_bfchw = pred_x0_flat.unflatten(0, (flow_pred_bfchw.shape[0], flow_pred_bfchw.shape[1]))
                    pred_x0 = rearrange(pred_x0_bfchw, "b f c h w -> b c f h w")

                    # Denoise: x_t -> x_0, then add noise for next timestep
                    if step_idx < len(denoising_step_list) - 1:
                        # Not the last step, add noise for next timestep
                        next_timestep = denoising_step_list[step_idx + 1]
                        next_t = torch.tensor([next_timestep], device=device, dtype=torch.float32)

                        # Rearrange for scale_noise: b c f h w -> b f c h w
                        pred_x0_for_noise = rearrange(pred_x0, "b c f h w -> b f c h w")
                        noise = randn_tensor(
                            pred_x0_for_noise.shape, generator=generator, device=device, dtype=pred_x0_for_noise.dtype
                        )

                        # Add noise using scale_noise: flatten batch and frames
                        # scale_noise formula: sigma * noise + (1 - sigma) * sample
                        local_latent_flat = self.scheduler.scale_noise(
                            pred_x0_for_noise.flatten(0, 1),
                            next_t.expand(pred_x0_for_noise.shape[0] * pred_x0_for_noise.shape[1]),
                            noise.flatten(0, 1)
                        )
                        local_latent_bfchw = local_latent_flat.unflatten(0, (pred_x0_for_noise.shape[0], pred_x0_for_noise.shape[1]))
                        local_latent = rearrange(local_latent_bfchw, "b f c h w -> b c f h w")
                    else:
                        # Last step, use x_0 as final result
                        local_latent = pred_x0

                    progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

                # Store the denoised chunk
                output[:, :, start_f:end_f] = local_latent

                # Update KV cache for this chunk by running forward pass at timestep 0
                latent_for_cache = output[:, :, start_f:end_f]
                timestep_zero = torch.zeros(latent_for_cache.shape[0], device=device, dtype=torch.long)

                for block_cache in chunk_kv_cache:
                    block_cache.enable_save()

                # Forward pass to update KV cache
                _, chunk_kv_cache = self.transformer(
                    latent_for_cache.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep_zero,
                    return_dict=False,
                    kv_caches=chunk_kv_cache,
                    attention_kwargs=self.attention_kwargs,
                )

                kv_cache.update_chunk_cache(chunk_idx, chunk_kv_cache)

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = output

        if output_type == "latent":
            video = latents
        else:
            latents = latents.to(self.vae.dtype)
            torch_accelerator_module = getattr(torch, get_device(), torch.cuda)
            oom_error = (
                torch.OutOfMemoryError
                if is_torch_version(">=", "2.5.0")
                else torch_accelerator_module.OutOfMemoryError
            )
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            try:
                video = self.vae.decode(latents, return_dict=False)[0]
            except oom_error as e:
                warnings.warn(
                    f"{e}. \n"
                    f"Try to use VAE tiling for large images. For example: \n"
                    f"pipe.vae.enable_tiling(tile_sample_min_width=512, tile_sample_min_height=512)"
                )

            if use_resolution_binning:
                video = self.video_processor.resize_and_crop_tensor(video, orig_width, orig_height)

            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return SanaVideoPipelineOutput(frames=video)
