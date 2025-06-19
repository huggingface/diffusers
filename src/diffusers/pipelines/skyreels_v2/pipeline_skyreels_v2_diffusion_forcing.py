# Copyright 2025 The SkyReels-V2 Team, The Wan Team and The HuggingFace Team. All rights reserved.
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
import math
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import ftfy
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import SkyReelsV2LoraLoaderMixin
from ...models import AutoencoderKLWan, SkyReelsV2Transformer3DModel
from ...schedulers import FlowMatchUniPCMultistepScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import SkyReelsV2PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """\
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import (
        ...     SkyReelsV2DiffusionForcingPipeline,
        ...     FlowMatchUniPCMultistepScheduler,
        ...     AutoencoderKLWan,
        ... )
        >>> from diffusers.utils import export_to_video

        >>> # Load the pipeline
        >>> # Available models:
        >>> # - Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
        >>> # - Skywork/SkyReels-V2-DF-14B-540P-Diffusers
        >>> # - Skywork/SkyReels-V2-DF-14B-720P-Diffusers
        >>> vae = AutoencoderKLWan.from_pretrained(
        ...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
        ...     subfolder="vae",
        ...     torch_dtype=torch.float32,
        ... )
        >>> pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        ...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
        ...     vae=vae,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> shift = 8.0  # 8.0 for T2V, 5.0 for I2V
        >>> pipe.scheduler = FlowMatchUniPCMultistepScheduler.from_config(pipe.scheduler.config, shift=shift)
        >>> pipe = pipe.to("cuda")
        >>> pipe.transformer.set_ar_attention(causal_block_size=5)

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

        >>> output = pipe(
        ...     prompt=prompt,
        ...     num_inference_steps=30,
        ...     height=544,
        ...     width=960,
        ...     guidance_scale=6.0,  # 6.0 for T2V, 5.0 for I2V
        ...     num_frames=97,
        ...     ar_step=5,  # Controls asynchronous inference (0 for synchronous mode)
        ...     overlap_history=None,  # Number of frames to overlap for smooth transitions in long videos
        ...     addnoise_condition=20,  # Improves consistency in long video generation
        ... ).frames[0]
        >>> export_to_video(output, "video.mp4", fps=24, quality=8)
        ```
"""


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


class SkyReelsV2DiffusionForcingPipeline(DiffusionPipeline, SkyReelsV2LoraLoaderMixin):
    """
    Pipeline for Text-to-Video (t2v) generation using SkyReels-V2 with diffusion forcing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a specific device, etc.).

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`UMT5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`SkyReelsV2Transformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchUniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: SkyReelsV2Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchUniPCMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
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

        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
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
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        overlap_history=None,
        num_frames=None,
        base_num_frames=None,
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

        if num_frames > base_num_frames and overlap_history is None:
            raise ValueError(
                "`overlap_history` is required when `num_frames` exceeds `base_num_frames` to ensure smooth transitions in long video generation. "
                "Please specify a value for `overlap_history`. Recommended values are 17 or 37."
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        base_latent_num_frames: Optional[int] = None,
        video_latents: Optional[torch.Tensor] = None,
        causal_block_size: Optional[int] = None,
        overlap_history_latent_frames: Optional[int] = None,
        long_video_iter: Optional[int] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        prefix_video_latents = None
        prefix_video_latents_length = 0

        if video_latents is not None:  # long video generation at the iterations other than the first one
            prefix_video_latents = video_latents[:, :, -overlap_history_latent_frames:]

            if prefix_video_latents.shape[2] % causal_block_size != 0:
                truncate_len_latents = prefix_video_latents.shape[2] % causal_block_size
                logger.warning(
                    f"The length of prefix video latents is truncated by {truncate_len_latents} frames for the causal block size alignment. "
                    f"This truncation ensures compatibility with the causal block size, which is required for proper processing. "
                    f"However, it may slightly affect the continuity of the generated video at the truncation boundary."
                )
                prefix_video_latents = prefix_video_latents[:, :, :-truncate_len_latents]
            prefix_video_latents_length = prefix_video_latents.shape[2]

            finished_frame_num = (
                long_video_iter * (base_latent_num_frames - overlap_history_latent_frames) + overlap_history_latent_frames
            )
            left_frame_num = num_latent_frames - finished_frame_num
            num_latent_frames = min(left_frame_num + overlap_history_latent_frames, base_latent_num_frames)
        elif base_latent_num_frames is not None:  # long video generation at the first iteration
            num_latent_frames = base_latent_num_frames
        else:  # short video generation
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents, num_latent_frames, prefix_video_latents, prefix_video_latents_length

    def generate_timestep_matrix(
        self,
        num_latent_frames: int,
        step_template: torch.Tensor,
        base_num_latent_frames: int,
        ar_step: int = 5,
        num_pre_ready: int = 0,
        causal_block_size: int = 1,
        shrink_interval_with_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        """
        Generate timestep matrix for (a)synchronous inference.

        Args:
            num_latent_frames: Number of latent frames to process
            step_template: Timestep schedule
            base_num_latent_frames: Defines the processing window that contains all the asynchronous blocks
            ar_step: Delay between starting each block (0 = synchronous)
            num_pre_ready: Number of frames that are ready before the first step
            causal_block_size: Frames per block
            shrink_interval_with_mask: If True, the valid interval will be shrunk based on the update mask

        An example:
        base_num_frames=97, num_frames=97, num_inference_steps=30, ar_step=5, causal_block_size=5

        vae_scale_factor_temporal -> 4
        num_latent_frames: (97-1)//vae_scale_factor_temporal+1 = 25 frames -> 5 blocks of 5 frames each

        base_num_latent_frames = (97-1)//vae_scale_factor_temporal+1 = 25 → blocks = 25//5 = 5 blocks
        This 5 blocks means the maximum context length of the model is 25 frames in the latent space.

        Asynchronous Processing Timeline within one chunk:
        ┌─────────────────────────────────────────────────────────────────┐
        │ Steps:    1    6   11   16   21   26   31   36   41   46   50   │
        │ Block 1: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                       │
        │ Block 2:      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                  │
        │ Block 3:           [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]             │
        │ Block 4:                [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]        │
        │ Block 5:                     [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]   │
        └─────────────────────────────────────────────────────────────────┘

        For Long Videos (num_frames > base_num_frames):
        base_num_frames acts as the "sliding window size" for chunked processing.

        Example: 300-frame video with base_num_frames=97, overlap_history=17
        ┌────── Chunk 1 (frames 1-97) ───────┐
        │ Processing window: 97 frames       │ → 5 blocks, async processing
        │ Generates: frames 1-97             │
        └────────────────────────────────────┘
                    ┌────── Chunk 2 (frames 81-177) ───────┐
                    │ Processing window: 97 frames         │ → 5 blocks, async processing
                    │ Overlap: 17 frames (81-97) from prev │
                    │ Generates: frames 98-177             │
                    └──────────────────────────────────────┘
                                ┌────── Chunk 3 (frames 161-260) ───────┐
                                │ Processing window: 97 frames          │ → 5 blocks, async processing
                                │ Overlap: 17 frames (161-177) from prev│
                                │ Generates: frames 178-260             │
                                └───────────────────────────────────────┘

        Each chunk independently runs the asynchronous processing with its own 5 blocks.
        base_num_frames controls:
        1. Memory usage (larger window = more VRAM)
        2. Model context length (must match training constraints)
        3. Number of blocks per chunk (base_num_latent_frames // causal_block_size)

        Each block takes 30 steps to complete denoising.
        Block N starts at step: 1 + (N-1) x ar_step
        Total steps: 30 + (5-1) x 5 = 50 steps


        Synchronous mode (ar_step=0) would process all blocks/frames simultaneously:
        ┌──────────────────────────────────────────────┐
        │ Steps:       1            ...            30  │
        │ All blocks: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] │
        └──────────────────────────────────────────────┘
        Total steps: 30 steps


        Returns:
            Tuple of (step_matrix, step_index, step_update_mask, valid_interval)
        """
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_blocks = num_latent_frames // causal_block_size
        base_num_blocks = base_num_latent_frames // causal_block_size
        if base_num_blocks < num_blocks:
            min_ar_step = len(step_template) / base_num_blocks
            if ar_step < min_ar_step:
                raise ValueError(f"`ar_step` should be at least {math.ceil(min_ar_step)} in your setting")

        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_blocks, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // causal_block_size] = num_iterations

        while not torch.all(pre_row >= (num_iterations - 1)):
            new_row = torch.zeros(num_blocks, dtype=torch.long)
            for i in range(num_blocks):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_latent_frames is set to the model max length (for training)
        terminal_flag = base_num_blocks
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_blocks, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1

        for curr_mask in update_mask:
            if terminal_flag < num_blocks and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_blocks, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if causal_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            valid_interval = [(s * causal_block_size, e * causal_block_size) for s, e in valid_interval]

        return step_matrix, step_index, step_update_mask, valid_interval

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
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
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
        max_sequence_length: int = 512,
        overlap_history: Optional[int] = None,
        shift: float = 8.0,
        addnoise_condition: float = 0,
        base_num_frames: int = 97,
        ar_step: int = 0,
        causal_block_size: Optional[int] = None,
        fps: int = 24,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `544`):
                The height of the generated video.
            width (`int`, defaults to `960`):
                The width of the generated video.
            num_frames (`int`, defaults to `97`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `6.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. (**6.0 for T2V**, **5.0 for I2V**)
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SkyReelsV2PipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int`, *optional*, defaults to `512`):
                The maximum sequence length of the prompt.
            shift (`float`, *optional*, defaults to `8.0`):
                Flow matching scheduler parameter (**5.0 for I2V**, **8.0 for T2V**)
            overlap_history (`int`, *optional*, defaults to `None`):
                Number of frames to overlap for smooth transitions in long videos. If `None`, the pipeline assumes
                short video generation mode, and no overlap is applied. 17 and 37 are recommended to set.
            addnoise_condition (`float`, *optional*, defaults to `0`):
                This is used to help smooth the long video generation by adding some noise to the clean condition. Too
                large noise can cause the inconsistency as well. 20 is a recommended value, and you may try larger
                ones, but it is recommended to not exceed 50.
            base_num_frames (`int`, *optional*, defaults to `97`):
                97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)
            ar_step (`int`, *optional*, defaults to `0`):
                Controls asynchronous inference (0 for synchronous mode) You can set `ar_step=5` to enable asynchronous
                inference. When asynchronous inference, `causal_block_size=5` is recommended while it is not supposed
                to be set for synchronous generation. Asynchronous inference will take more steps to diffuse the whole
                sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous
                inference may improve the instruction following and visual consistent performance.
            causal_block_size (`int`, *optional*, defaults to `None`):
                The number of frames in each block/chunk. Recommended when using asynchronous inference (when ar_step >
                0)
            fps (`int`, *optional*, defaults to `24`):
                Frame rate of the generated video

        Examples:

        Returns:
            [`~SkyReelsV2PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`SkyReelsV2PipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            overlap_history,
            num_frames,
            base_num_frames,
        )

        if addnoise_condition > 60:
            logger.warning(
                f"The value of 'addnoise_condition' is too large ({addnoise_condition}) and may cause inconsistencies in long video generation. A value of 20 is recommended."
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
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        timesteps = self.scheduler.timesteps

        if causal_block_size is None:
            causal_block_size = self.transformer.config.num_frame_per_block
        fps_embeds = [fps] * prompt_embeds.shape[0]
        fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]

        # Determine if we're doing long video generation
        is_long_video = overlap_history is not None and base_num_frames is not None and num_frames > base_num_frames
        # Initialize accumulated_latents to store all latents in one tensor
        accumulated_latents = None
        if is_long_video:
            # Long video generation setup
            overlap_history_latent_frames = (overlap_history - 1) // self.vae_scale_factor_temporal + 1
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            base_latent_num_frames = (
                (base_num_frames - 1) // self.vae_scale_factor_temporal + 1
                if base_num_frames is not None
                else num_latent_frames
            )
            n_iter = (
                1 + (num_latent_frames - base_latent_num_frames - 1) // (base_latent_num_frames - overlap_history_latent_frames) + 1
            )

        else:
            # Short video generation setup
            n_iter = 1
            base_latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # Loop through iterations (multiple iterations only for long videos)
        for iter_idx in range(n_iter):
            if is_long_video:
                print(f"long_video_iter:{iter_idx}")

            # 5. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels
            latents, current_num_latent_frames, prefix_video_latents, prefix_video_latents_length = (
                self.prepare_latents(
                    batch_size * num_videos_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    num_frames,
                    torch.float32,
                    device,
                    generator,
                    latents if iter_idx == 0 else None,
                    video_latents=accumulated_latents,  # Pass latents directly instead of decoded video
                    base_latent_num_frames=base_latent_num_frames if is_long_video else None,
                    causal_block_size=causal_block_size,
                    overlap_history_latent_frames=overlap_history_latent_frames if is_long_video else None,
                    long_video_iter=iter_idx if is_long_video else None,
                )
            )

            if prefix_video_latents_length > 0:
                latents[:, :, :prefix_video_latents_length, :, :] = prefix_video_latents.to(transformer_dtype)

            # 6. Prepare sample schedulers and timestep matrix
            sample_schedulers = []
            for _ in range(current_num_latent_frames):
                sample_scheduler = deepcopy(self.scheduler)
                sample_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
                sample_schedulers.append(sample_scheduler)

            # Different matrix generation for short vs long video
            step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                current_num_latent_frames,
                timesteps,
                current_num_latent_frames if is_long_video else base_latent_num_frames,
                ar_step,
                prefix_video_latents_length,
                causal_block_size,
            )

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(step_matrix)

            with self.progress_bar(total=len(step_matrix)) as progress_bar:
                for i, t in enumerate(step_matrix):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    valid_interval_start, valid_interval_end = valid_interval[i]
                    latent_model_input = (
                        latents[:, :, valid_interval_start:valid_interval_end, :, :].to(transformer_dtype).clone()
                    )
                    timestep = t.expand(latents.shape[0], -1)[:, valid_interval_start:valid_interval_end].clone()

                    if addnoise_condition > 0 and valid_interval_start < prefix_video_latents_length:
                        noise_factor = 0.001 * addnoise_condition
                        latent_model_input[:, :, valid_interval_start:prefix_video_latents_length, :, :] = (
                            latent_model_input[:, :, valid_interval_start:prefix_video_latents_length, :, :]
                            * (1.0 - noise_factor)
                            + torch.randn_like(
                                latent_model_input[:, :, valid_interval_start:prefix_video_latents_length, :, :]
                            )
                            * noise_factor
                        )
                        timestep[:, valid_interval_start:prefix_video_latents_length] = addnoise_condition

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        enable_diffusion_forcing=True,
                        fps=fps_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    if self.do_classifier_free_guidance:
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            enable_diffusion_forcing=True,
                            fps=fps_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    update_mask_i = step_update_mask[i]
                    for idx in range(valid_interval_start, valid_interval_end):
                        if update_mask_i[idx].item():
                            latents[:, :, idx, :, :] = sample_schedulers[idx].step(
                                noise_pred[:, :, idx - valid_interval_start, :, :],
                                t[idx],
                                latents[:, :, idx, :, :],
                                return_dict=False,
                                generator=generator,
                            )[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(step_matrix) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

            # Handle latent accumulation for long videos or use the current latents for short videos
            if is_long_video:
                if accumulated_latents is None:
                    accumulated_latents = latents
                else:
                    # Keep overlap frames for conditioning but don't include them in final output
                    accumulated_latents = torch.cat(
                        [accumulated_latents, latents[:, :, overlap_history_latent_frames:]], dim=2
                    )

        if is_long_video:
            latents = accumulated_latents

        self._current_timestep = None

        # Final decoding step - convert latents to pixels
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

        return SkyReelsV2PipelineOutput(frames=video)
