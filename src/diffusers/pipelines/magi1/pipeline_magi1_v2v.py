# Copyright 2025 The SandAI Team and The HuggingFace Team. All rights reserved.
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

# MAGI-1 V2V Pipeline with Autoregressive Chunked Generation
#
# ✅ IMPLEMENTED:
# - Video-to-Video generation using prefix video conditioning
# - Autoregressive chunked generation (always enabled, matching original MAGI-1)
# - Window-based scheduling: chunk_width=6, window_size=4
# - Progressive denoising across overlapping temporal windows
# - Proper CFG with separate forward passes (diffusers style)
# - Input video frames encoding to VAE latent as clean prefix chunks
#
# ⚠️ CURRENT LIMITATION:
# - No KV caching: attention is recomputed for previous chunks
# - This is less efficient than the original but fully functional
#
# ⏳ FUTURE OPTIMIZATIONS (when diffusers adds generic KV caching):
# 1. **KV Cache Management**:
#    - Cache attention keys/values for previously denoised chunks
#    - Reuse cached computations instead of recomputing
#    - Will significantly speed up generation (2-3x faster expected)
#
# 2. **Special Token Support** (optional enhancement):
#    - Duration tokens: indicate how many chunks remain to generate
#    - Quality tokens: HQ_TOKEN for high-quality generation
#    - Style tokens: THREE_D_MODEL_TOKEN, TWO_D_ANIME_TOKEN
#    - Motion tokens: STATIC_FIRST_FRAMES_TOKEN, DYNAMIC_FIRST_FRAMES_TOKEN
#
# 3. **Streaming Generation**:
#    - Yield clean chunks as they complete (generator pattern)
#    - Enable real-time preview during generation
#
# Reference: https://github.com/SandAI/MAGI-1/blob/main/inference/pipeline/video_generate.py

import html
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import Magi1LoraLoaderMixin
from ...models import AutoencoderKLMagi1, Magi1Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import Magi1PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


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


def generate_chunk_sequences(chunk_num: int, window_size: int, chunk_offset: int = 0):
    """
    Generate chunk scheduling sequences for autoregressive video generation.

    Args:
        chunk_num: Total number of chunks to generate
        window_size: Number of chunks to process in each window
        chunk_offset: Number of clean prefix chunks (for I2V/V2V)

    Returns:
        ```
        clip_start: Start index of chunks to process
        clip_end: End index of chunks to process
        t_start: Start index in time dimension
        t_end: End index in time dimension
        ```

    Examples:
        ```
        chunk_num=8, window_size=4, chunk_offset=0
        Stage 0: Process chunks [0:1], denoise chunk 0
        Stage 1: Process chunks [0:2], denoise chunk 1
        Stage 2: Process chunks [0:3], denoise chunk 2
        Stage 3: Process chunks [0:4], denoise chunk 3
        Stage 4: Process chunks [1:5], denoise chunk 4
        ...
        ```
    """
    start_index = chunk_offset
    end_index = chunk_num + window_size - 1

    clip_start = [max(chunk_offset, i - window_size + 1) for i in range(start_index, end_index)]
    clip_end = [min(chunk_num, i + 1) for i in range(start_index, end_index)]

    t_start = [max(0, i - chunk_num + 1) for i in range(start_index, end_index)]
    t_end = [
        min(window_size, i - chunk_offset + 1) if i - chunk_offset < window_size else window_size
        for i in range(start_index, end_index)
    ]

    return clip_start, clip_end, t_start, t_end


def load_special_tokens(special_tokens_path: Optional[str] = None) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load special conditioning tokens from numpy file.

    Args:
        special_tokens_path: Path to special_tokens.npz file. If None, returns None (no special tokens).

    Returns:
        Dictionary mapping token names to embeddings, or None if path not provided or file doesn't exist.
    """
    if special_tokens_path is None:
        return None

    try:
        import os

        import numpy as np

        if not os.path.exists(special_tokens_path):
            logger.warning(f"Special tokens file not found at {special_tokens_path}, skipping special token loading.")
            return None

        special_token_data = np.load(special_tokens_path)
        caption_token = torch.tensor(special_token_data["caption_token"].astype(np.float16))
        logo_token = torch.tensor(special_token_data["logo_token"].astype(np.float16))
        other_tokens = special_token_data["other_tokens"]

        tokens = {
            "CAPTION_TOKEN": caption_token,
            "LOGO_TOKEN": logo_token,
            "TRANS_TOKEN": torch.tensor(other_tokens[:1].astype(np.float16)),
            "HQ_TOKEN": torch.tensor(other_tokens[1:2].astype(np.float16)),
            "STATIC_FIRST_FRAMES_TOKEN": torch.tensor(other_tokens[2:3].astype(np.float16)),
            "DYNAMIC_FIRST_FRAMES_TOKEN": torch.tensor(other_tokens[3:4].astype(np.float16)),
            "BORDERNESS_TOKEN": torch.tensor(other_tokens[4:5].astype(np.float16)),
            "THREE_D_MODEL_TOKEN": torch.tensor(other_tokens[15:16].astype(np.float16)),
            "TWO_D_ANIME_TOKEN": torch.tensor(other_tokens[16:17].astype(np.float16)),
        }

        # Duration tokens (8 total, representing 1-8 chunks remaining)
        for i in range(8):
            tokens[f"DURATION_TOKEN_{i + 1}"] = torch.tensor(other_tokens[i + 7 : i + 8].astype(np.float16))

        logger.info(f"Loaded {len(tokens)} special tokens from {special_tokens_path}")
        return tokens
    except Exception as e:
        logger.warning(f"Failed to load special tokens: {e}")
        return None


def prepend_special_tokens(
    prompt_embeds: torch.Tensor,
    special_tokens: Optional[Dict[str, torch.Tensor]],
    use_hq_token: bool = False,
    use_3d_style: bool = False,
    use_2d_anime_style: bool = False,
    use_static_first_frames: bool = False,
    use_dynamic_first_frames: bool = False,
    max_sequence_length: int = 800,
) -> torch.Tensor:
    """
    Prepend special conditioning tokens to text embeddings.

    Args:
        prompt_embeds: Text embeddings [batch, seq_len, hidden_dim]
        special_tokens: Dictionary of special token embeddings
        use_hq_token: Whether to add high-quality token
        use_3d_style: Whether to add 3D model style token
        use_2d_anime_style: Whether to add 2D anime style token
        use_static_first_frames: Whether to add static motion token
        use_dynamic_first_frames: Whether to add dynamic motion token
        max_sequence_length: Maximum sequence length after prepending

    Returns:
        Text embeddings with special tokens prepended
    """
    if special_tokens is None:
        return prompt_embeds

    device = prompt_embeds.device
    dtype = prompt_embeds.dtype
    batch_size, seq_len, hidden_dim = prompt_embeds.shape

    # Collect tokens to prepend (in order: motion, quality, style)
    tokens_to_add = []
    if use_static_first_frames and "STATIC_FIRST_FRAMES_TOKEN" in special_tokens:
        tokens_to_add.append(special_tokens["STATIC_FIRST_FRAMES_TOKEN"])
    if use_dynamic_first_frames and "DYNAMIC_FIRST_FRAMES_TOKEN" in special_tokens:
        tokens_to_add.append(special_tokens["DYNAMIC_FIRST_FRAMES_TOKEN"])
    if use_hq_token and "HQ_TOKEN" in special_tokens:
        tokens_to_add.append(special_tokens["HQ_TOKEN"])
    if use_3d_style and "THREE_D_MODEL_TOKEN" in special_tokens:
        tokens_to_add.append(special_tokens["THREE_D_MODEL_TOKEN"])
    if use_2d_anime_style and "TWO_D_ANIME_TOKEN" in special_tokens:
        tokens_to_add.append(special_tokens["TWO_D_ANIME_TOKEN"])

    # Prepend tokens
    for token in tokens_to_add:
        token = token.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        prompt_embeds = torch.cat([token, prompt_embeds], dim=1)

    # Truncate to max length
    if prompt_embeds.shape[1] > max_sequence_length:
        prompt_embeds = prompt_embeds[:, :max_sequence_length, :]

    return prompt_embeds


def prepare_v2v_embeddings(
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    num_chunks: int,
    clean_chunk_num: int,
    max_sequence_length: int = 800,
    prompt_mask: Optional[torch.Tensor] = None,
    negative_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepare per-chunk text embeddings for V2V generation.

    In V2V, clean prefix chunks (from the input video) use null embeddings, while chunks to be denoised use the actual
    text embeddings.

    Args:
        prompt_embeds: Text embeddings [batch_size, seq_len, hidden_dim]
        negative_prompt_embeds: Negative text embeddings (optional)
        num_chunks: Total number of chunks
        clean_chunk_num: Number of clean prefix chunks
        max_sequence_length: Maximum sequence length

    Returns:
        - prompt_embeds_per_chunk: [B, num_chunks, L, D]
        - negative_prompt_embeds_per_chunk: [B, num_chunks, L, D] or None
        - prompt_masks_per_chunk: [B, num_chunks, L] or None
        - negative_masks_per_chunk: [B, num_chunks, L] or None
    """
    batch_size = prompt_embeds.shape[0]
    seq_len = prompt_embeds.shape[1]
    hidden_dim = prompt_embeds.shape[2]
    device = prompt_embeds.device
    dtype = prompt_embeds.dtype

    # Number of chunks that need denoising
    denoise_chunk_num = num_chunks - clean_chunk_num

    # Create null embeddings (zeros) for clean chunks
    null_embeds = torch.zeros(batch_size, 1, seq_len, hidden_dim, device=device, dtype=dtype)

    # Expand prompt embeddings for denoise chunks
    # Shape: [batch_size, denoise_chunk_num, seq_len, hidden_dim]
    denoise_embeds = prompt_embeds.unsqueeze(1).repeat(1, denoise_chunk_num, 1, 1)

    # Concatenate: [null_embeds for clean chunks] + [text_embeds for denoise chunks]
    # Shape: [batch_size, num_chunks, seq_len, hidden_dim]
    if clean_chunk_num > 0:
        null_embeds_expanded = null_embeds.repeat(1, clean_chunk_num, 1, 1)
        prompt_embeds_per_chunk = torch.cat([null_embeds_expanded, denoise_embeds], dim=1)
    else:
        prompt_embeds_per_chunk = denoise_embeds

    # Build prompt masks per chunk
    prompt_masks_per_chunk = None
    negative_masks_per_chunk = None
    if prompt_mask is not None:
        denoise_masks = prompt_mask.unsqueeze(1).repeat(1, denoise_chunk_num, 1)
        if clean_chunk_num > 0:
            null_masks = torch.zeros(
                prompt_mask.shape[0], clean_chunk_num, prompt_mask.shape[1], device=device, dtype=prompt_mask.dtype
            )
            prompt_masks_per_chunk = torch.cat([null_masks, denoise_masks], dim=1)
        else:
            prompt_masks_per_chunk = denoise_masks

    # Same for negative embeddings
    if negative_prompt_embeds is not None:
        denoise_neg_embeds = negative_prompt_embeds.unsqueeze(1).repeat(1, denoise_chunk_num, 1, 1)
        if clean_chunk_num > 0:
            null_neg_embeds_expanded = null_embeds.repeat(1, clean_chunk_num, 1, 1)
            negative_prompt_embeds_per_chunk = torch.cat([null_neg_embeds_expanded, denoise_neg_embeds], dim=1)
        else:
            negative_prompt_embeds_per_chunk = denoise_neg_embeds
    else:
        negative_prompt_embeds_per_chunk = None
    if negative_mask is not None:
        denoise_neg_masks = negative_mask.unsqueeze(1).repeat(1, denoise_chunk_num, 1)
        if clean_chunk_num > 0:
            null_neg_masks = torch.zeros(
                negative_mask.shape[0],
                clean_chunk_num,
                negative_mask.shape[1],
                device=device,
                dtype=negative_mask.dtype,
            )
            negative_masks_per_chunk = torch.cat([null_neg_masks, denoise_neg_masks], dim=1)
        else:
            negative_masks_per_chunk = denoise_neg_masks

    return (
        prompt_embeds_per_chunk,
        negative_prompt_embeds_per_chunk,
        prompt_masks_per_chunk,
        negative_masks_per_chunk,
    )


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import Magi1VideoToVideoPipeline, AutoencoderKLMagi1, FlowMatchEulerDiscreteScheduler
        >>> from diffusers.utils import export_to_video, load_video

        >>> model_id = "SandAI/Magi1-V2V-14B-480P-Diffusers"
        >>> vae = AutoencoderKLMagi1.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

        >>> # IMPORTANT: MAGI-1 requires shift=3.0 for the scheduler (SD3-style time resolution transform)
        >>> scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", shift=3.0)

        >>> pipe = Magi1VideoToVideoPipeline.from_pretrained(
        ...     model_id, vae=vae, scheduler=scheduler, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> # Load prefix video (e.g., 24 frames)
        >>> video = load_video("path/to/input_video.mp4", num_frames=24)
        >>> prompt = (
        ...     "Continue this video with smooth camera motion and consistent style. "
        ...     "The scene evolves naturally with coherent motion."
        ... )
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, worst quality, low quality"

        >>> output = pipe(
        ...     video=video,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=480,
        ...     width=832,
        ...     num_frames=81,  # Total frames including prefix
        ...     guidance_scale=5.0,
        ...     num_inference_steps=50,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


class Magi1VideoToVideoPipeline(DiffusionPipeline, Magi1LoraLoaderMixin):
    r"""
    Pipeline for video-to-video generation using Magi1.

    MAGI-1 is a DiT-based video generation model that supports autoregressive chunked generation for long videos. This
    V2V pipeline takes an input video and generates a continuation or extension of that video.

    **Note**: This implementation uses autoregressive chunked generation (chunk_width=6, window_size=4) as in the
    original MAGI-1 paper. The input video frames are encoded to latent representations and used as clean prefix chunks
    to condition the generation. Text prompts provide additional semantic guidance for the video continuation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`Magi1Transformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A flow matching scheduler with Euler discretization, using SD3-style time resolution transform.
        vae ([`AutoencoderKLMagi1`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: Magi1Transformer3DModel,
        vae: AutoencoderKLMagi1,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = self.vae.config.spatial_compression_ratio if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Special tokens for conditioning (optional)
        self.special_tokens = None

    def load_special_tokens_from_file(self, special_tokens_path: str):
        """
        Load special conditioning tokens from a numpy file.

        Args:
            special_tokens_path: Path to special_tokens.npz file
        """
        self.special_tokens = load_special_tokens(special_tokens_path)
        if self.special_tokens is not None:
            logger.info("Special tokens loaded successfully. You can now use quality, style, and motion control.")

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 800,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        mask = mask.repeat(1, num_videos_per_prompt)
        mask = mask.view(batch_size * num_videos_per_prompt, -1).to(device)
        return prompt_embeds, mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 800,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_mask = None

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
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

            negative_prompt_embeds, negative_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            negative_mask = None

        return prompt_embeds, negative_prompt_embeds, prompt_mask, negative_mask

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        video,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if video is None:
            raise ValueError(
                "Provide `video` for video-to-video generation. Cannot leave `video` undefined for V2V pipeline."
            )
        if video is not None and not isinstance(video, list):
            raise ValueError(f"`video` has to be of type `list` (list of PIL Images or tensors) but is {type(video)}")

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
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        video: Optional[List[PIL.Image.Image]],
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare latents for V2V generation, including encoding the input video frames as prefix_video.

        Args:
            video: Input video frames for V2V generation (list of PIL Images)
            batch_size: Batch size
            num_channels_latents: Number of latent channels
            height: Video height
            width: Video width
            num_frames: Total number of frames to generate (including prefix)
            dtype: Data type
            device: Device
            generator: Random generator
            latents: Pre-generated latents (optional)

        Returns:
            Tuple of (latents, prefix_video) where:
            - latents: Random noise tensor for generation [batch, channels, num_latent_frames, H, W]
            - prefix_video: Encoded video frames as clean latent [batch, channels, prefix_frames, H, W]
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Prepare random latents for generation
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Encode input video frames to latent as prefix_video
        prefix_video = None
        if video is not None and len(video) > 0:
            # Preprocess video frames to target size
            # video_processor.preprocess_video expects list of PIL Images
            video_tensor = self.video_processor.preprocess_video(video, height=height, width=width).to(
                device, dtype=torch.float32
            )

            # video_tensor shape: [batch, channels, num_frames, height, width]
            # For single batch, expand if needed
            if video_tensor.ndim == 4:
                # [channels, num_frames, height, width] -> [1, channels, num_frames, height, width]
                video_tensor = video_tensor.unsqueeze(0)

            # Encode to latent space using VAE
            # VAE expects [batch, channels, frames, height, width]
            if isinstance(generator, list):
                prefix_video = [
                    retrieve_latents(self.vae.encode(vid.unsqueeze(0)), sample_mode="sample", generator=g)
                    for g, vid in zip(generator, video_tensor)
                ]
                prefix_video = torch.cat(prefix_video)
            else:
                prefix_video = retrieve_latents(
                    self.vae.encode(video_tensor), sample_mode="sample", generator=generator
                )
                if prefix_video.shape[0] < batch_size:
                    prefix_video = prefix_video.repeat(batch_size, 1, 1, 1, 1)

            # Normalize latent using VAE's latent statistics
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(prefix_video.device, dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                prefix_video.device, dtype
            )
            prefix_video = prefix_video.to(dtype)
            prefix_video = (prefix_video - latents_mean) * latents_std

        return latents, prefix_video

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: List[PIL.Image.Image],
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 800,
        use_hq_token: bool = False,
        use_3d_style: bool = False,
        use_2d_anime_style: bool = False,
        use_static_first_frames: bool = False,
        use_dynamic_first_frames: bool = False,
        enable_distillation: bool = False,
        distill_nearly_clean_chunk_threshold: float = 0.3,
    ):
        r"""
        The call function to the pipeline for video-to-video generation.

        **Note**: This implementation uses autoregressive chunked generation (chunk_width=6, window_size=4) as in the
        original MAGI-1 paper. The input video frames are encoded to VAE latents and used as clean prefix chunks to
        condition the video generation. The implementation currently works without KV caching (attention is recomputed
        for previous chunks), which is less efficient than the original but still functional. KV caching optimization
        will be added when diffusers implements generic caching support for transformers.

        Args:
            video (`List[PIL.Image.Image]`):
                The input video frames to condition the video generation on. Must be a list of PIL Images representing
                the prefix video (e.g., first 24 frames of a video).
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during video generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
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
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, negative_prompt_embeds will be generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `"latent"`, `"pt"`, or `"np"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`Magi1PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `800`):
                The maximum sequence length for the text encoder. Sequences longer than this will be truncated. MAGI-1
                uses a max length of 800 tokens.
            use_hq_token (`bool`, *optional*, defaults to `False`):
                Whether to prepend the high-quality control token to the text embeddings. This token conditions the
                model to generate higher quality outputs. Requires special tokens to be loaded via
                `load_special_tokens_from_file`.
            use_3d_style (`bool`, *optional*, defaults to `False`):
                Whether to prepend the 3D model style token to the text embeddings. This token conditions the model to
                generate outputs with 3D modeling aesthetics. Requires special tokens to be loaded.
            use_2d_anime_style (`bool`, *optional*, defaults to `False`):
                Whether to prepend the 2D anime style token to the text embeddings. This token conditions the model to
                generate outputs with 2D anime aesthetics. Requires special tokens to be loaded.
            use_static_first_frames (`bool`, *optional*, defaults to `False`):
                Whether to prepend the static first frames token to the text embeddings. This token conditions the
                model to start the video with minimal motion in the first few frames. Requires special tokens to be
                loaded.
            use_dynamic_first_frames (`bool`, *optional*, defaults to `False`):
                Whether to prepend the dynamic first frames token to the text embeddings. This token conditions the
                model to start the video with significant motion in the first few frames. Requires special tokens to be
                loaded.
            enable_distillation (`bool`, *optional*, defaults to `False`):
                Whether to enable distillation mode. In distillation mode, the model uses modified timestep embeddings
                to support distilled (faster) inference. This requires a distilled model checkpoint.
            distill_nearly_clean_chunk_threshold (`float`, *optional*, defaults to `0.3`):
                Threshold for identifying nearly-clean chunks in distillation mode. Chunks with timestep > threshold
                are considered nearly clean and processed differently. Only used when `enable_distillation=True`.

        Examples:

        Returns:
            [`~Magi1PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`Magi1PipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated videos.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            video,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
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

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, prompt_mask, negative_mask = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.text_encoder.dtype,
        )

        # 3.5. Prepend special tokens if requested
        if self.special_tokens is not None and any(
            [use_hq_token, use_3d_style, use_2d_anime_style, use_static_first_frames, use_dynamic_first_frames]
        ):
            prompt_embeds = prepend_special_tokens(
                prompt_embeds=prompt_embeds,
                special_tokens=self.special_tokens,
                use_hq_token=use_hq_token,
                use_3d_style=use_3d_style,
                use_2d_anime_style=use_2d_anime_style,
                use_static_first_frames=use_static_first_frames,
                use_dynamic_first_frames=use_dynamic_first_frames,
                max_sequence_length=max_sequence_length,
            )
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = prepend_special_tokens(
                    prompt_embeds=negative_prompt_embeds,
                    special_tokens=self.special_tokens,
                    use_hq_token=use_hq_token,
                    use_3d_style=use_3d_style,
                    use_2d_anime_style=use_2d_anime_style,
                    use_static_first_frames=use_static_first_frames,
                    use_dynamic_first_frames=use_dynamic_first_frames,
                    max_sequence_length=max_sequence_length,
                )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, prefix_video = self.prepare_latents(
            video,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop (autoregressive chunked generation with V2V prefix conditioning)
        # MAGI-1 V2V uses autoregressive generation with chunk_width=6 and window_size=4
        # The input video frames are encoded as clean prefix chunks and used to condition the generation
        # Note: num_warmup_steps is calculated for compatibility but not used in progress bar logic
        # because autoregressive generation has a different iteration structure (stages × steps)
        # For FlowMatchEulerDiscreteScheduler (order=1), this doesn't affect the results
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Autoregressive chunked generation parameters
        chunk_width = 6  # Original MAGI-1 default
        window_size = 4  # Original MAGI-1 default

        num_latent_frames = latents.shape[2]
        num_chunks = (num_latent_frames + chunk_width - 1) // chunk_width

        # Calculate chunk_offset from prefix_video (for V2V, these are the clean video frame chunks)
        chunk_offset = 0
        if prefix_video is not None:
            # prefix_video has shape [batch, channels, num_prefix_frames, height, width] for V2V
            # Calculate how many chunks are covered by the prefix
            prefix_latent_frames = prefix_video.shape[2]
            chunk_offset = prefix_latent_frames // chunk_width

            # Pad prefix_video into latents at the beginning
            # The prefix frames are already clean and don't need denoising
            if prefix_latent_frames > 0:
                prefix_video = prefix_video.to(latents.dtype)
                latents[:, :, :prefix_latent_frames] = prefix_video

        # Calculate chunk scheduling: which chunks to process at each stage
        # chunk_offset skips the clean prefix chunks
        clip_start, clip_end, t_start, t_end = generate_chunk_sequences(num_chunks, window_size, chunk_offset)
        num_stages = len(clip_start)

        # Prepare per-chunk text embeddings for V2V
        # Clean chunks (from input video) use null embeddings, denoise chunks use text embeddings
        (
            prompt_embeds_per_chunk,
            negative_prompt_embeds_per_chunk,
            prompt_masks_per_chunk,
            negative_masks_per_chunk,
        ) = prepare_v2v_embeddings(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_chunks=num_chunks,
            clean_chunk_num=chunk_offset,
            max_sequence_length=max_sequence_length,
            prompt_mask=prompt_mask,
            negative_mask=negative_mask,
        )

        # Number of denoising steps per stage
        denoise_step_per_stage = len(timesteps) // window_size

        # Track how many times each chunk has been denoised
        chunk_denoise_count = {i: 0 for i in range(num_chunks)}

        with self.progress_bar(total=num_stages * denoise_step_per_stage) as progress_bar:
            for stage_idx in range(num_stages):
                # Determine which chunks to process in this stage
                chunk_start_idx = clip_start[stage_idx]
                chunk_end_idx = clip_end[stage_idx]
                t_start_idx = t_start[stage_idx]
                t_end_idx = t_end[stage_idx]

                # Extract chunk range in latent space
                latent_start = chunk_start_idx * chunk_width
                latent_end = min(chunk_end_idx * chunk_width, num_latent_frames)

                # Number of chunks in current window
                num_chunks_in_window = chunk_end_idx - chunk_start_idx

                # Prepare per-chunk conditioning with duration/borderness tokens
                # Duration tokens indicate how many chunks remain in the video
                # Borderness tokens condition on chunk boundaries
                chunk_prompt_embeds_list = []
                chunk_negative_prompt_embeds_list = []
                chunk_prompt_masks_list = []
                chunk_negative_masks_list = []

                if self.special_tokens is not None:
                    # Prepare per-chunk embeddings with duration tokens
                    # Each chunk gets a different duration token based on chunks remaining
                    for i, chunk_idx in enumerate(range(chunk_start_idx, chunk_end_idx)):
                        chunks_remaining = num_chunks - chunk_idx - 1
                        # Duration token ranges from 1-8 chunks
                        duration_idx = min(chunks_remaining, 7) + 1

                        # Get base embeddings for this chunk (clean chunks have null embeds, denoise chunks have text embeds)
                        token_embeds = prompt_embeds_per_chunk[:, chunk_idx].clone()

                        # Add duration and borderness tokens for this chunk
                        if f"DURATION_TOKEN_{duration_idx}" in self.special_tokens:
                            duration_token = self.special_tokens[f"DURATION_TOKEN_{duration_idx}"]
                            duration_token = duration_token.to(device=token_embeds.device, dtype=token_embeds.dtype)
                            duration_token = duration_token.unsqueeze(0).expand(batch_size, -1, -1)
                            token_embeds = torch.cat([duration_token, token_embeds], dim=1)

                        if "BORDERNESS_TOKEN" in self.special_tokens:
                            borderness_token = self.special_tokens["BORDERNESS_TOKEN"]
                            borderness_token = borderness_token.to(
                                device=token_embeds.device, dtype=token_embeds.dtype
                            )
                            borderness_token = borderness_token.unsqueeze(0).expand(batch_size, -1, -1)
                            token_embeds = torch.cat([borderness_token, token_embeds], dim=1)

                        # Build mask for this chunk from base per-chunk mask
                        token_mask = (
                            prompt_masks_per_chunk[:, chunk_idx]
                            if prompt_masks_per_chunk is not None
                            else torch.ones(
                                batch_size, token_embeds.shape[1], device=token_embeds.device, dtype=torch.int64
                            )
                        )
                        add_count = 0
                        if f"DURATION_TOKEN_{duration_idx}" in self.special_tokens:
                            add_count += 1
                        if "BORDERNESS_TOKEN" in self.special_tokens:
                            add_count += 1
                        if add_count > 0:
                            prepend = torch.ones(
                                batch_size, add_count, dtype=token_mask.dtype, device=token_mask.device
                            )
                            token_mask = torch.cat([prepend, token_mask], dim=1)
                        # Truncate to max length
                        if token_embeds.shape[1] > max_sequence_length:
                            token_embeds = token_embeds[:, :max_sequence_length, :]
                            token_mask = token_mask[:, :max_sequence_length]

                        chunk_prompt_embeds_list.append(token_embeds)
                        chunk_prompt_masks_list.append(token_mask)

                        # Same for negative prompts
                        if self.do_classifier_free_guidance and negative_prompt_embeds_per_chunk is not None:
                            neg_token_embeds = negative_prompt_embeds_per_chunk[:, chunk_idx].clone()

                            if f"DURATION_TOKEN_{duration_idx}" in self.special_tokens:
                                duration_token = self.special_tokens[f"DURATION_TOKEN_{duration_idx}"]
                                duration_token = duration_token.to(
                                    device=neg_token_embeds.device, dtype=neg_token_embeds.dtype
                                )
                                duration_token = duration_token.unsqueeze(0).expand(batch_size, -1, -1)
                                neg_token_embeds = torch.cat([duration_token, neg_token_embeds], dim=1)

                            if "BORDERNESS_TOKEN" in self.special_tokens:
                                borderness_token = self.special_tokens["BORDERNESS_TOKEN"]
                                borderness_token = borderness_token.to(
                                    device=neg_token_embeds.device, dtype=neg_token_embeds.dtype
                                )
                                borderness_token = borderness_token.unsqueeze(0).expand(batch_size, -1, -1)
                                neg_token_embeds = torch.cat([borderness_token, neg_token_embeds], dim=1)

                            # Build negative mask for this chunk
                            neg_mask = (
                                negative_masks_per_chunk[:, chunk_idx]
                                if negative_masks_per_chunk is not None
                                else torch.ones(
                                    batch_size,
                                    neg_token_embeds.shape[1],
                                    device=neg_token_embeds.device,
                                    dtype=torch.int64,
                                )
                            )
                            add_count = 0
                            if f"DURATION_TOKEN_{duration_idx}" in self.special_tokens:
                                add_count += 1
                            if "BORDERNESS_TOKEN" in self.special_tokens:
                                add_count += 1
                            if add_count > 0:
                                prepend = torch.ones(
                                    batch_size, add_count, dtype=neg_mask.dtype, device=neg_mask.device
                                )
                                neg_mask = torch.cat([prepend, neg_mask], dim=1)
                            if neg_token_embeds.shape[1] > max_sequence_length:
                                neg_token_embeds = neg_token_embeds[:, :max_sequence_length, :]
                                neg_mask = neg_mask[:, :max_sequence_length]

                            chunk_negative_prompt_embeds_list.append(neg_token_embeds)
                            chunk_negative_masks_list.append(neg_mask)

                # Denoise this chunk range for denoise_step_per_stage steps
                for denoise_idx in range(denoise_step_per_stage):
                    if self.interrupt:
                        break

                    # Calculate timestep index for each chunk in the current window
                    # Chunks at different stages get different timesteps based on their denoise progress
                    timestep_indices = []
                    for offset in range(num_chunks_in_window):
                        # Map offset within window to time index
                        t_idx_within_window = t_start_idx + offset
                        if t_idx_within_window < t_end_idx:
                            # This chunk is actively being denoised in this window
                            t_idx = t_idx_within_window * denoise_step_per_stage + denoise_idx
                        else:
                            # This chunk is beyond the active window, use max timestep (it's already cleaner)
                            t_idx = min((window_size - 1) * denoise_step_per_stage + denoise_idx, len(timesteps) - 1)
                        timestep_indices.append(t_idx)

                    # Reverse order: chunks further from start are noisier
                    timestep_indices.reverse()

                    # Get actual timesteps (reversed order: high noise to low noise)
                    current_timesteps = timesteps[timestep_indices]

                    # Create per-chunk timestep tensor: [batch_size, num_chunks_in_window]
                    # Each chunk gets its own timestep based on how many times it's been denoised
                    timestep_per_chunk = current_timesteps.unsqueeze(0).expand(batch_size, -1)

                    # Store first timestep for progress tracking
                    self._current_timestep = current_timesteps[0]

                    # Extract chunk
                    latent_chunk = latents[:, :, latent_start:latent_end].to(transformer_dtype)

                    # Prepare distillation parameters if enabled
                    num_steps = None
                    distill_interval = None
                    distill_nearly_clean_chunk = None

                    if enable_distillation:
                        # distill_interval represents the time interval between denoising steps
                        distill_interval = len(timesteps) / num_inference_steps

                        # Determine if chunks are nearly clean (low noise) based on their timesteps
                        # Check the first active chunk's timestep (after reversing, this is the noisiest chunk being actively denoised)
                        # Normalize timestep to [0, 1] range where 0=clean, 1=noise
                        nearly_clean_chunk_t = current_timesteps[0].item() / self.scheduler.config.num_train_timesteps
                        distill_nearly_clean_chunk = nearly_clean_chunk_t < distill_nearly_clean_chunk_threshold

                        num_steps = num_inference_steps

                    # Prepare per-chunk embeddings
                    # The transformer expects embeddings in shape [batch_size * num_chunks_in_window, seq_len, hidden_dim]
                    # Each chunk gets its own embedding with appropriate duration/borderness tokens
                    if chunk_prompt_embeds_list:
                        # Stack per-chunk embeddings: [num_chunks_in_window, batch_size, seq_len, hidden_dim]
                        chunk_prompt_embeds = torch.stack(chunk_prompt_embeds_list, dim=0)
                        # Reshape to [batch_size * num_chunks_in_window, seq_len, hidden_dim]
                        chunk_prompt_embeds = chunk_prompt_embeds.transpose(0, 1).flatten(0, 1)

                        if chunk_negative_prompt_embeds_list:
                            chunk_negative_prompt_embeds = torch.stack(chunk_negative_prompt_embeds_list, dim=0)
                            chunk_negative_prompt_embeds = chunk_negative_prompt_embeds.transpose(0, 1).flatten(0, 1)
                        else:
                            chunk_negative_prompt_embeds = None
                    else:
                        # Fallback: use per-chunk embeddings without special tokens
                        # Extract embeddings for the current chunk range
                        chunk_prompt_embeds = prompt_embeds_per_chunk[:, chunk_start_idx:chunk_end_idx]
                        chunk_prompt_embeds = chunk_prompt_embeds.flatten(0, 1)
                        chunk_prompt_masks = (
                            prompt_masks_per_chunk[:, chunk_start_idx:chunk_end_idx].flatten(0, 1)
                            if prompt_masks_per_chunk is not None
                            else None
                        )

                        if negative_prompt_embeds_per_chunk is not None:
                            chunk_negative_prompt_embeds = negative_prompt_embeds_per_chunk[
                                :, chunk_start_idx:chunk_end_idx
                            ]
                            chunk_negative_prompt_embeds = chunk_negative_prompt_embeds.flatten(0, 1)
                            chunk_negative_masks = (
                                negative_masks_per_chunk[:, chunk_start_idx:chunk_end_idx].flatten(0, 1)
                                if negative_masks_per_chunk is not None
                                else None
                            )
                        else:
                            chunk_negative_prompt_embeds = None
                            chunk_negative_masks = None

                    # Create encoder attention mask(s) from tokenizer masks
                    if chunk_prompt_embeds_list:
                        prompt_masks_stacked = (
                            torch.stack(chunk_prompt_masks_list, dim=0).transpose(0, 1).flatten(0, 1)
                        )
                    else:
                        prompt_masks_stacked = chunk_prompt_masks
                    if prompt_masks_stacked is None:
                        prompt_masks_stacked = torch.ones(
                            batch_size * num_chunks_in_window,
                            chunk_prompt_embeds.shape[1],
                            dtype=torch.int64,
                            device=chunk_prompt_embeds.device,
                        )
                    encoder_attention_mask = prompt_masks_stacked.to(dtype=chunk_prompt_embeds.dtype).view(
                        batch_size * num_chunks_in_window, 1, 1, -1
                    )

                    if self.do_classifier_free_guidance:
                        if chunk_negative_prompt_embeds_list:
                            negative_masks_stacked = (
                                torch.stack(chunk_negative_masks_list, dim=0).transpose(0, 1).flatten(0, 1)
                            )
                        else:
                            negative_masks_stacked = chunk_negative_masks
                        if negative_masks_stacked is None and chunk_negative_prompt_embeds is not None:
                            negative_masks_stacked = torch.ones(
                                batch_size * num_chunks_in_window,
                                chunk_negative_prompt_embeds.shape[1],
                                dtype=torch.int64,
                                device=chunk_negative_prompt_embeds.device,
                            )
                        encoder_attention_mask_neg = (
                            negative_masks_stacked.to(dtype=chunk_prompt_embeds.dtype).view(
                                batch_size * num_chunks_in_window, 1, 1, -1
                            )
                            if negative_masks_stacked is not None
                            else encoder_attention_mask
                        )

                    # Pad prefix video into latent_chunk if applicable (I2V)
                    # This ensures clean prefix frames are maintained during denoising
                    if prefix_video is not None:
                        prefix_length = prefix_video.shape[2]
                        prefix_video_start = chunk_start_idx * chunk_width

                        if prefix_length > prefix_video_start:
                            # Calculate how many frames to pad
                            padding_length = min(prefix_length - prefix_video_start, latent_chunk.shape[2])
                            prefix_video_end = prefix_video_start + padding_length

                            # Pad clean prefix frames into latent_chunk
                            latent_chunk = latent_chunk.clone()
                            latent_chunk[:, :, :padding_length] = prefix_video[
                                :, :, prefix_video_start:prefix_video_end
                            ]

                            # Set timesteps for clean prefix chunks to maximum (indicates "already clean")
                            # This matches original MAGI-1's try_pad_prefix_video logic
                            num_clean_chunks_in_window = padding_length // chunk_width
                            if num_clean_chunks_in_window > 0:
                                # Get max timestep from scheduler
                                max_timestep = timesteps[0]
                                timestep_per_chunk[:, :num_clean_chunks_in_window] = max_timestep

                    # Generate KV range for autoregressive attention
                    # Each chunk can attend to itself and all previous chunks in the sequence
                    # Shape: [batch_size * num_chunks_in_window, 2] where each row is [start_token_idx, end_token_idx]
                    chunk_token_nums = (
                        (latent_chunk.shape[2] // num_chunks_in_window)  # frames per chunk
                        * (latent_chunk.shape[3] // self.transformer.config.patch_size[1])  # height tokens
                        * (latent_chunk.shape[4] // self.transformer.config.patch_size[2])  # width tokens
                    )
                    kv_range = []
                    for b in range(batch_size):
                        # batch_offset should be based on total chunks in the video, not chunk_end_idx
                        batch_offset = b * num_chunks
                        for c in range(num_chunks_in_window):
                            # This chunk can attend from the start of the video up to its own end
                            chunk_global_idx = chunk_start_idx + c
                            k_start = batch_offset * chunk_token_nums
                            k_end = (batch_offset + chunk_global_idx + 1) * chunk_token_nums
                            kv_range.append([k_start, k_end])
                    kv_range = torch.tensor(kv_range, dtype=torch.int32, device=device)

                    # Predict noise (conditional)
                    # Note: MAGI-1 uses velocity field (flow matching), but following diffusers convention
                    # we use noise_pred naming for consistency across all pipelines
                    noise_pred = self.transformer(
                        hidden_states=latent_chunk,
                        timestep=timestep_per_chunk,
                        encoder_hidden_states=chunk_prompt_embeds,
                        encoder_attention_mask=encoder_attention_mask,
                        attention_kwargs=attention_kwargs,
                        denoising_range_num=num_chunks_in_window,
                        range_num=chunk_end_idx,
                        slice_point=chunk_start_idx,
                        kv_range=kv_range,
                        num_steps=num_steps,
                        distill_interval=distill_interval,
                        distill_nearly_clean_chunk=distill_nearly_clean_chunk,
                        return_dict=False,
                    )[0]

                    # Classifier-free guidance: separate forward pass for unconditional
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_chunk,
                            timestep=timestep_per_chunk,
                            encoder_hidden_states=chunk_negative_prompt_embeds,
                            encoder_attention_mask=encoder_attention_mask_neg,
                            attention_kwargs=attention_kwargs,
                            denoising_range_num=num_chunks_in_window,
                            range_num=chunk_end_idx,
                            slice_point=chunk_start_idx,
                            kv_range=kv_range,
                            num_steps=num_steps,
                            distill_interval=distill_interval,
                            distill_nearly_clean_chunk=distill_nearly_clean_chunk,
                            return_dict=False,
                        )[0]
                        # Apply classifier-free guidance
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                    # CRITICAL: Apply per-chunk Euler integration with different delta_t for each chunk
                    # This matches the original MAGI-1's integrate() function
                    # Each chunk in the window is at a different noise level and needs its own time step

                    # Calculate per-chunk timesteps for integration
                    # Get current timesteps (t_before)
                    current_timesteps = timesteps[timestep_indices]

                    # Get next timesteps (t_after) - one step forward for each chunk
                    next_timestep_indices = [min(idx + 1, len(timesteps) - 1) for idx in timestep_indices]
                    next_timesteps = timesteps[next_timestep_indices]

                    # Convert timesteps to sigmas (matching FlowMatchEulerDiscreteScheduler)
                    current_sigmas = current_timesteps / self.scheduler.config.num_train_timesteps
                    next_sigmas = next_timesteps / self.scheduler.config.num_train_timesteps

                    # Calculate delta_t for each chunk: [num_chunks_in_window]
                    delta_t = next_sigmas - current_sigmas

                    # Reshape latent_chunk and noise_pred to separate chunks
                    # From: [batch, channels, frames, height, width]
                    # To: [batch, channels, num_chunks, chunk_width, height, width]
                    batch_size_actual, num_channels, total_frames, height_latent, width_latent = latent_chunk.shape

                    # Ensure total_frames is divisible by chunk_width for reshaping
                    # (it should be, but let's handle edge cases)
                    num_complete_chunks = total_frames // chunk_width
                    remainder_frames = total_frames % chunk_width

                    if remainder_frames == 0:
                        # Perfect division: reshape and apply per-chunk delta_t
                        latent_chunk = latent_chunk.reshape(
                            batch_size_actual,
                            num_channels,
                            num_complete_chunks,
                            chunk_width,
                            height_latent,
                            width_latent,
                        )
                        noise_pred = noise_pred.reshape(
                            batch_size_actual,
                            num_channels,
                            num_complete_chunks,
                            chunk_width,
                            height_latent,
                            width_latent,
                        )

                        # Apply Euler integration: x_chunk = x_chunk + velocity * delta_t
                        # delta_t shape: [num_chunks] -> broadcast to [1, 1, num_chunks, 1, 1, 1]
                        delta_t_broadcast = delta_t.reshape(1, 1, -1, 1, 1, 1).to(
                            latent_chunk.device, latent_chunk.dtype
                        )
                        latent_chunk = latent_chunk + noise_pred * delta_t_broadcast

                        # Reshape back to original dimensions
                        latent_chunk = latent_chunk.reshape(
                            batch_size_actual, num_channels, total_frames, height_latent, width_latent
                        )
                    else:
                        # Handle remainder frames separately (edge case for last incomplete chunk)
                        complete_frames = num_complete_chunks * chunk_width

                        # Process complete chunks
                        latent_chunk_complete = latent_chunk[:, :, :complete_frames]
                        noise_pred_complete = noise_pred[:, :, :complete_frames]

                        latent_chunk_complete = latent_chunk_complete.reshape(
                            batch_size_actual,
                            num_channels,
                            num_complete_chunks,
                            chunk_width,
                            height_latent,
                            width_latent,
                        )
                        noise_pred_complete = noise_pred_complete.reshape(
                            batch_size_actual,
                            num_channels,
                            num_complete_chunks,
                            chunk_width,
                            height_latent,
                            width_latent,
                        )

                        # Apply per-chunk delta_t to complete chunks
                        delta_t_broadcast = (
                            delta_t[:num_complete_chunks]
                            .reshape(1, 1, -1, 1, 1, 1)
                            .to(latent_chunk.device, latent_chunk.dtype)
                        )
                        latent_chunk_complete = latent_chunk_complete + noise_pred_complete * delta_t_broadcast
                        latent_chunk_complete = latent_chunk_complete.reshape(
                            batch_size_actual, num_channels, complete_frames, height_latent, width_latent
                        )

                        # Process remainder frames with last delta_t
                        if remainder_frames > 0:
                            latent_chunk_remainder = latent_chunk[:, :, complete_frames:]
                            noise_pred_remainder = noise_pred[:, :, complete_frames:]
                            delta_t_remainder = delta_t[-1].to(latent_chunk.device, latent_chunk.dtype)
                            latent_chunk_remainder = latent_chunk_remainder + noise_pred_remainder * delta_t_remainder

                            # Concatenate complete and remainder
                            latent_chunk = torch.cat([latent_chunk_complete, latent_chunk_remainder], dim=2)
                        else:
                            latent_chunk = latent_chunk_complete

                    # Write back to full latents
                    latents[:, :, latent_start:latent_end] = latent_chunk

                    # Update chunk denoise counts
                    for chunk_idx in range(chunk_start_idx, chunk_end_idx):
                        chunk_denoise_count[chunk_idx] += 1

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        # Use first timestep for callback (most representative)
                        callback_timestep = current_timesteps[0]
                        callback_outputs = callback_on_step_end(
                            self, stage_idx * denoise_step_per_stage + denoise_idx, callback_timestep, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return Magi1PipelineOutput(frames=video)
