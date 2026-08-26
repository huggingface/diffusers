# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

"""Shared DFR denoise core used by [`LTX2DFRPipeline`] and [`LTX2DFRTemporalRefinePipeline`]."""

import copy
import math
from typing import Any, Callable

import numpy as np
import PIL.Image
import torch

from ...image_processor import PipelineImageInput
from ...schedulers import LTXEulerAncestralRFScheduler
from ...utils import is_torch_xla_available, logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from .latent_upsampler import LTX2LatentUpsamplerModel
from .pipeline_ltx2_condition import LTX2VideoCondition
from .pipeline_output import LTX2DFRPipelineOutput
from .prompt_enhancement import (
    _pad_inputs_for_attention_alignment,
    _prepare_enhance_image,
    clean_response,
)
from .utils import (
    GEMMA3_PROMPT_ENHANCEMENT_CONFIG,
    GEMMA4_PROMPT_ENHANCEMENT_CONFIG,
    apply_image_conditioning_crf,
    resolve_default_image_crf,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Keyframes carried between temporal rounds are pinned just short of fully clean so a tile can still settle its seam
# frame.
ANCHOR_KEYFRAME_STRENGTH = 0.95

# Ancestral noise fraction used by the temporal refine rounds. Their short schedule densifies detail rather than
# building structure, so a partly stochastic step is what fills in the freshly interpolated frames.
TEMPORAL_ANCESTRAL_ETA = 0.5

# RoPE time is `pixel_frame / fps`. The transformer is trained around 24/25/30 and 60 fps, not 48 or 120. A 120 fps
# time base halves every token's temporal span and the model can no longer lay out the VAE's pixel frames inside one
# latent -- it decodes as a motion spike at each latent border followed by a stall. 48 fps stretches the same span the
# other way. Condition at 60 in both cases and treat the decoded frames at the playback rate (120 fps: generate 2x
# frames at 60, mux as 120; 48 fps: generate the 24x2 canvas at 60, mux as 48). Playback fps is used for the returned
# frame count and the audio trim only.
MAX_CONDITIONING_FPS = 60.0
SNAP_CONDITIONING_FPS_ABOVE = 30.0


def _conditioning_fps(playback_fps: float) -> float:
    """Fps the transformer sees. Playback may be 48, 96, 120, ...; those rates only affect muxing."""
    return MAX_CONDITIONING_FPS if playback_fps > SNAP_CONDITIONING_FPS_ABOVE else playback_fps


# The epilogue's keyframes arrive as finished frames, rebuilt at the output resolution, so they are pinned fully clean.
EPILOGUE_KEYFRAME_STRENGTH = 1.0


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


def _audio_window_for_tile(
    audio_latents: torch.Tensor,
    pixel_start: int,
    tile_frames: int,
    playback_fps: float,
    source_seconds: float,
    conditioning_fps: float,
    audio_latents_per_second: float,
) -> torch.Tensor:
    """
    Cut the frozen stage-1 audio to one temporal tile's window and resample it to that tile's token count.

    The window is wall clock: `pixel_start / playback_fps` through `(pixel_start + tile_frames) / playback_fps`, as a
    fraction of `source_seconds`. Taking a fraction of the *canvas* instead would drift, because a refine round maps `N
    -> 2 (N - 1) + 1` while the frame rate doubles, so each round's canvas is a hair shorter than twice the last one
    and the tail tiles would pull audio from past their own playback. `conditioning_fps` only sizes the output token
    count, matching what the video side asks the transformer for.

    Returns the packed `(batch_size, tile_audio_frames, channels * mel_bins)` window and its frame count.
    """
    source_frames = audio_latents.shape[1]
    tile_audio_frames = round(tile_frames / conditioning_fps * audio_latents_per_second)
    start = pixel_start / playback_fps / source_seconds * source_frames
    span = tile_frames / playback_fps / source_seconds * source_frames
    positions = start + (span / tile_audio_frames) * torch.arange(
        tile_audio_frames, device=audio_latents.device, dtype=torch.float32
    )
    positions = positions.clamp(0, source_frames - 1)
    low = positions.floor().long()
    high = (low + 1).clamp(max=source_frames - 1)
    weight = (positions - low).to(audio_latents.dtype).view(1, -1, 1)
    return audio_latents[:, low] * (1 - weight) + audio_latents[:, high] * weight


def trim_canvas(latents: torch.Tensor, num_frames: int, temporal_compression_ratio: int) -> torch.Tensor:
    """Drop the padded tail of a DFR canvas so it matches `num_frames` pixel frames.

    The canvas is padded to a whole number of keyframe segments; this keeps the tokens that cover
    `num_frames` and is the last step before VAE decode. `output_type="latent"` returns the padded
    grid so a slot on the pad (e.g. pixel 96 on an 81→97 canvas) is not dropped.
    """
    keep = (num_frames - 1) // temporal_compression_ratio + 1
    return latents[:, :, :keep]


class LTX2DFRCoreMixin:
    """Pack/unpack, latent prep, and the shared DFR denoise loop.

    Not a pipeline. [`LTX2DFRPipeline`] and [`LTX2DFRTemporalRefinePipeline`] both mix this in; they do
    not subclass each other.
    """

    def _init_dfr_runtime(self) -> None:
        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer", None) is not None else 1
        )

        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        )
        self.audio_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        self.audio_latent_channels = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, resample="bilinear")
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 1024
        )
        tokenizer_padding_side = "left"
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
        self.tokenizer_padding_side = tokenizer_padding_side

    def _maybe_normalize_video_latents(
        self, latents: torch.Tensor | None, latents_normalized: bool
    ) -> torch.Tensor | None:
        if latents is None or latents_normalized:
            return latents
        # Public latents are often bf16 from the upsampler; denoise and the old one-shot path pack in float32.
        latents = latents.float()
        return self._normalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )

    def _inference_timesteps(self, sigmas: list[float], device: torch.device) -> torch.Tensor:
        """Set the scheduler schedule and return the timesteps the denoise loop should walk.

        [`FlowMatchEulerDiscreteScheduler`] takes distilled sigmas *without* a terminal 0 (it appends
        one) and exposes `len(sigmas)` timesteps. [`LTXEulerAncestralRFScheduler`] takes distilled
        sigmas *plus* terminal 0 and the loop must skip the last timestep, matching
        [`LTXI2VLongMultiPromptPipeline`].
        """
        if isinstance(self.scheduler, LTXEulerAncestralRFScheduler):
            sigmas = [float(sigma) for sigma in sigmas]
            if not sigmas or sigmas[-1] != 0.0:
                sigmas = [*sigmas, 0.0]
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            return self.scheduler.timesteps[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        return self.scheduler.timesteps

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        spatio_temporal_guidance_blocks=None,
        stg_scale=None,
        audio_stg_scale=None,
        system_prompt=None,
        enable_prompt_enhancement=None,
        num_frames=None,
        min_seconds=1.0,
        max_seconds=20.0,
        image=None,
        image_crf=None,
        latents=None,
        audio_latents=None,
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

        if latents is not None and latents.ndim != 5:
            raise ValueError(
                f"Only unpacked (5D) video latents of shape `[batch_size, latent_channels, latent_frames,"
                f" latent_height, latent_width] are supported, but got {latents.ndim} dims. If you have packed (3D)"
                f" latents, please unpack them (e.g. using the `_unpack_latents` method)."
            )
        if audio_latents is not None and audio_latents.ndim != 4:
            raise ValueError(
                f"Only unpacked (4D) audio latents of shape `[batch_size, num_channels, audio_length, mel_bins] are"
                f" supported, but got {audio_latents.ndim} dims. If you have packed (3D) latents, please unpack them"
                f" (e.g. using the `_unpack_audio_latents` method)."
            )

        if ((stg_scale > 0.0) or (audio_stg_scale > 0.0)) and not spatio_temporal_guidance_blocks:
            raise ValueError(
                "Spatio-Temporal Guidance (STG) is specified but no STG blocks are supplied. Please supply a list of"
                "block indices at which to apply STG in `spatio_temporal_guidance_blocks`"
            )

        if (
            enable_prompt_enhancement
            and prompt is not None
            and system_prompt is None
            and getattr(self, "prompt_enhancer", None) is None
        ):
            raise ValueError(
                "`system_prompt` must be supplied to enable prompt enhancement when no dedicated "
                "`prompt_enhancer` component is configured (LTX-2.0/2.3)."
            )

        if min_seconds >= max_seconds:
            raise ValueError(
                f"`min_seconds` ({min_seconds}) must be less than `max_seconds` ({max_seconds})."
                " A collapsed range leaves no room for a prediction, and cannot generally be satisfied by a frame"
                " count on the VAE's temporal grid."
            )

        # Auto-duration path: `num_frames` omitted on a pipeline that has a `duration_head`.
        if num_frames is None and getattr(self, "duration_head", None) is not None:
            num_prompts = len(prompt) if isinstance(prompt, list) else 1 if prompt is not None else len(prompt_embeds)
            if num_prompts > 1:
                raise ValueError(
                    f"`num_frames` was omitted so the duration head would auto-predict, but {num_prompts} prompts were"
                    " supplied. The duration head predicts one duration, and prompts with different natural lengths"
                    " cannot share a single frame count. Call the pipeline once per prompt, or pass `num_frames` as an"
                    " integer."
                )

        if latents is None and image is not None:
            crf = image_crf if image_crf is not None else resolve_default_image_crf(self.text_encoder)
            if crf != 0 and not isinstance(image, PIL.Image.Image):
                raise ValueError(
                    f"`image_crf` re-compression requires a `PIL.Image.Image` input, got {type(image)}. "
                    "Pass a PIL image, or set `image_crf=0` to skip re-compression."
                )

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.enhance_prompt
    def enhance_prompt(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int | None = None,
        seed: int = 10,
        generator: torch.Generator | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        image: PipelineImageInput | None = None,
    ):
        """
        Enhances the supplied `prompt` by generating a new prompt using the prompt enhancer (a Gemma
        conditional-generation model) from it and a system prompt. When `image` is supplied, the enhancer is also
        conditioned on that reference frame (I2V / keyframe-style enhancement). Uses the dedicated `prompt_enhancer`
        component if one is configured (e.g. LTX-2.5, whose text encoder isn't trained for enhancement), otherwise
        falls back to the main `text_encoder` (LTX-2.0/2.3, which double as their own enhancer).

        Message templates, decoding kwargs, response cleaning, and image long-side prep match `ltx-core` /
        `ltx-pipelines` (`enhance_t2v` / `enhance_i2v` / `generate_enhanced_prompt`).
        """
        device = device or self._execution_device
        using_dedicated_enhancer = getattr(self, "prompt_enhancer", None) is not None
        enhancer = self.prompt_enhancer if using_dedicated_enhancer else self.text_encoder
        config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG if using_dedicated_enhancer else GEMMA3_PROMPT_ENHANCEMENT_CONFIG

        generation_kwargs = (
            dict(generation_kwargs) if generation_kwargs is not None else dict(config.generation_kwargs)
        )
        if max_new_tokens is None:
            max_new_tokens = config.max_new_tokens

        # Templates match ltx-core `LTXGemmaTextEncoder.enhance_t2v` / `enhance_i2v` for both Gemma 3 and 4.
        if image is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"user prompt: {prompt}"},
            ]
            enhance_image = None
        else:
            enhance_image = _prepare_enhance_image(image)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                    ],
                },
            ]

        template = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.processor(text=template, images=enhance_image, return_tensors="pt").to(device)
        pad_token_id = (
            self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        )
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)
        enhancer.to(device)

        # `transformers.GenerationMixin.generate` does not support using a `torch.Generator` to control randomness,
        # so manually apply a seed for reproducible generation.
        if generator is not None:
            seed = generator.initial_seed() if not isinstance(generator, list) else generator[0].initial_seed()
        torch.manual_seed(seed)
        generated_sequences = enhancer.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )  # tensor of shape [batch_size, seq_len]

        generated_ids = [seq[len(model_inputs.input_ids[i]) :] for i, seq in enumerate(generated_sequences)]
        enhanced_prompt = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [clean_response(text) for text in enhanced_prompt]

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._get_gemma_prompt_embeds
    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`str` or `torch.device`):
                torch device to place the resulting embeddings on
            dtype: (`torch.dtype`):
                torch dtype to cast the prompt embeds to
            max_sequence_length (`int`, defaults to 1024): Maximum sequence length to use for the prompt.
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if getattr(self, "tokenizer", None) is not None:
            # Gemma expects left padding for chat-style prompts
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(dtype=dtype)  # Pack to 3D

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        DFR runs the distilled sigma schedule, which is trained to be used without classifier-free guidance, so there
        is no negative branch here.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for `prompt_embeds`.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._pack_latents
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_latents
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_image2video.LTX2ImageToVideoPipeline._normalize_latents
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._normalize_audio_latents
    def _normalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) / latents_std

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_audio_latents
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._pack_audio_latents
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
    ) -> torch.Tensor:
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        if patch_size is not None and patch_size_t is not None:
            # Packs the latents into a patch sequence of shape [B, L // p_t * M // p, C * p_t * p] (a ndim=3 tnesor).
            # dim=1 is the effective audio sequence length and dim=2 is the effective audio input feature size.
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length / patch_size_t
            post_patch_mel_bins = latent_mel_bins / patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        else:
            # Packs the latents into a patch sequence of shape [B, L, C * M]. This implicitly assumes a (mel)
            # patch_size of M (all mel bins constitutes a single patch) and a patch_size_t of 1.
            latents = latents.transpose(1, 2).flatten(2, 3)  # [B, C, L, M] --> [B, L, C * M]
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_audio_latents
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> torch.Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.trim_conditioning_sequence
    def trim_conditioning_sequence(self, start_frame: int, sequence_num_frames: int, target_num_frames: int) -> int:
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.
        Returns:
            int: updated sequence length
        """
        scale_factor = self.vae_temporal_compression_ratio
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.preprocess_conditions
    def preprocess_conditions(
        self,
        conditions: LTX2VideoCondition | list[LTX2VideoCondition] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        device: torch.device | None = None,
    ) -> tuple[list[torch.Tensor], list[float], list[int], list[int]]:
        """
        Preprocesses the condition images/videos to torch tensors.

        Args:
            conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*, defaults to `None`):
                A list of image/video condition instances.
            height (`int`, *optional*, defaults to `512`):
                The desired height in pixels.
            width (`int`, *optional*, defaults to `768`):
                The desired width in pixels.
            num_frames (`int`, *optional*, defaults to `121`):
                The desired number of frames in the generated video.
            device (`torch.device`, *optional*, defaults to `None`):
                The device on which to put the preprocessed image/video tensors.

        Returns:
            `Tuple[List[torch.Tensor], List[float], List[int], List[int]]`:
                Returns a 4-tuple of lists of length `len(conditions)` as follows:
                    1. The first list is a list of preprocessed video tensors of shape [batch_size=1, num_channels,
                       num_frames, height, width].
                    2. The second list is a list of conditioning strengths.
                    3. The third list is a list of latent-space indices for each condition.
                    4. The fourth list is a list of (trimmed) pixel-space frame counts per condition. This is needed
                       for keyframe coord semantics (single-pixel-frame keyframes have a clamped temporal extent).
        """
        conditioning_frames, conditioning_strengths, conditioning_indices, conditioning_pixel_frames = [], [], [], []

        if conditions is None:
            conditions = []
        if isinstance(conditions, LTX2VideoCondition):
            conditions = [conditions]

        frame_scale_factor = self.vae_temporal_compression_ratio
        latent_num_frames = (num_frames - 1) // frame_scale_factor + 1
        for i, condition in enumerate(conditions):
            # Create a channels-last video-like array of shape (F, H, W, C) in preparation for resizing.
            if isinstance(condition.frames, PIL.Image.Image):
                arr = np.array(condition.frames.convert("RGB"))[None]  # (1, H, W, 3)
            elif isinstance(condition.frames, list) and all(isinstance(f, PIL.Image.Image) for f in condition.frames):
                arr = np.stack([np.array(f.convert("RGB")) for f in condition.frames])  # (F, H, W, 3)
            elif isinstance(condition.frames, np.ndarray):
                arr = condition.frames if condition.frames.ndim == 4 else condition.frames[None]
            elif isinstance(condition.frames, torch.Tensor):
                t = condition.frames if condition.frames.ndim == 4 else condition.frames.unsqueeze(0)
                # Reference layout for video tensors is (F, C, H, W); convert to (F, H, W, C) for the
                # resize logic, which expects channels-last.
                arr = t.detach().cpu().permute(0, 2, 3, 1).numpy()
            else:
                raise TypeError(f"Unsupported `frames` type for condition {i}: {type(condition.frames)}")

            # Single-frame image keyframes are H.264 re-compressed at the model CRF (ltx-pipelines
            # `ImageConditioner.resolve_crf` + `media_io.preprocess`). Multi-frame video conditions are not.
            if arr.shape[0] == 1:
                crf = condition.crf if condition.crf is not None else resolve_default_image_crf(self.text_encoder)
                if crf != 0 and arr.dtype != np.uint8:
                    raise ValueError(
                        f"Image conditioning CRF expects a uint8 RGB frame, got dtype={arr.dtype}. "
                        "Pass a PIL image / uint8 array, or set `crf=0` on the condition to skip re-compression."
                    )
                arr = apply_image_conditioning_crf(arr[0], crf)[None]

            src_h, src_w = arr.shape[1], arr.shape[2]
            num_cond_frames = arr.shape[0]
            # Convert the NumPy array to a channels-first tensor of shape (1, C, F, H, W)
            pixels = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
            pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, C, F, H, W)

            # Resize so the longer side fills the target, then center-crop to exact (height, width).
            scale = max(height / src_h, width / src_w)
            new_h = math.ceil(src_h * scale)
            new_w = math.ceil(src_w * scale)
            # Flatten (B, C, F, H, W) → (B*F, C, H, W) for the per-frame interpolation
            pixels = pixels.permute(0, 2, 1, 3, 4).reshape(num_cond_frames, 3, src_h, src_w)
            # NOTE: we avoid using VideoProcessor.preprocess_video here because it uses PIL.Image.resize under the
            # hood, which will apply an anti-aliasing pre-filter when downsampling. The original LTX-2.X code simply
            # uses F.interpolate, which is reproduced here.
            pixels = torch.nn.functional.interpolate(pixels, size=(new_h, new_w), mode="bilinear", align_corners=False)
            top = (new_h - height) // 2
            left = (new_w - width) // 2
            pixels = pixels[:, :, top : top + height, left : left + width]
            pixels = pixels.reshape(1, num_cond_frames, 3, height, width).permute(0, 2, 1, 3, 4)

            # Map [0, 255] → [-1, 1] (VAE input convention).
            condition_pixels = pixels / 127.5 - 1.0

            # Interpret the index as a latent index, following the original LTX-2 code.
            latent_start_idx = condition.index
            # Support negative latent indices (e.g. -1 for the last latent index)
            if latent_start_idx < 0:
                # latent_start_idx will be positive because latent_num_frames is positive
                latent_start_idx = latent_start_idx % latent_num_frames
            if latent_start_idx >= latent_num_frames:
                logger.warning(
                    f"The starting latent index {latent_start_idx} of condition {i} is too big for the specified number"
                    f" of latent frames {latent_num_frames}. This condition will be skipped."
                )
                continue

            cond_num_frames = condition_pixels.size(2)
            start_idx = max((latent_start_idx - 1) * frame_scale_factor + 1, 0)
            truncated_cond_frames = self.trim_conditioning_sequence(start_idx, cond_num_frames, num_frames)
            condition_pixels = condition_pixels[:, :, :truncated_cond_frames]

            conditioning_frames.append(condition_pixels.to(dtype=self.vae.dtype, device=device))
            conditioning_strengths.append(condition.strength)
            conditioning_indices.append(latent_start_idx)
            conditioning_pixel_frames.append(truncated_cond_frames)

        return conditioning_frames, conditioning_strengths, conditioning_indices, conditioning_pixel_frames

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline._prepare_keyframe_coords
    def _prepare_keyframe_coords(
        self,
        keyframe_latent_num_frames: int,
        keyframe_latent_height: int,
        keyframe_latent_width: int,
        pixel_frame_idx: int,
        num_pixel_frames: int,
        fps: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute positional coordinates for a keyframe condition being appended as extra tokens.

        Mirrors `VideoConditionByKeyframeIndex.apply_to` in the reference implementation:
        - Latent coords scaled to pixel space *without* the causal fix (since non-zero-index keyframes don't need the
          first-frame causal adjustment).
        - Temporal axis offset by `pixel_frame_idx` (the pixel-space index at which the keyframe appears).
        - For single-pixel-frame keyframes, the per-patch temporal extent is clamped to `[idx, idx + 1)` so the
          keyframe occupies a single pixel timestep rather than the VAE-scaled range.
        - Temporal coords divided by `fps` to produce seconds.
        """
        patch_size = self.transformer_spatial_patch_size
        patch_size_t = self.transformer_temporal_patch_size
        scale_factors = (
            self.vae_temporal_compression_ratio,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )

        grid_f = torch.arange(
            start=0, end=keyframe_latent_num_frames, step=patch_size_t, dtype=torch.float32, device=device
        )
        grid_h = torch.arange(start=0, end=keyframe_latent_height, step=patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=keyframe_latent_width, step=patch_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        latent_coords = torch.stack([grid, patch_ends], dim=-1)  # [3, N_F, N_H, N_W, 2]
        latent_coords = latent_coords.flatten(1, 3)  # [3, num_patches, 2]
        latent_coords = latent_coords.unsqueeze(0)  # [1, 3, num_patches, 2]

        scale_tensor = torch.tensor(scale_factors, device=device, dtype=latent_coords.dtype)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        # No causal fix: keyframe coords place the keyframe at `pixel_frame_idx` without the first-frame adjustment.
        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] + pixel_frame_idx

        if num_pixel_frames == 1:
            # Single-pixel-frame keyframe: clamp temporal extent to [idx, idx + 1).
            pixel_coords[:, 0, :, 1:] = pixel_coords[:, 0, :, :1] + 1

        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

        return pixel_coords

    def prepare_latents(
        self,
        conditions: list[LTX2VideoCondition] | None = None,
        condition_latents: list[tuple[int, torch.Tensor, float, int]] | None = None,
        keyframe_latents: list[tuple[int, torch.Tensor, float]] | None = None,
        slot_frame_indices: list[int] | None = None,
        slot_initial_latents: torch.Tensor | None = None,
        reference_latents: torch.Tensor | None = None,
        reference_downscale_factor: int = 1,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        noise_scale: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        latents_normalized: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, slice | None]:
        """
        Prepare the noisy packed video latents for one DFR denoising pass.

        The packed sequence is laid out as `[base | keyframes | slots | reference]`:

        - Base tokens cover the target latent grid, seeded from `latents` when supplied.
        - Frame conditions with `index == 0` set the clean target at the first-frame positions; those with `index > 0`
          and every entry of `keyframe_latents` are appended as extra keyframe tokens with a per-token conditioning
          mask equal to their strength.
        - `slot_frame_indices` appends one latent frame's worth of *generated* keyframe tokens per position, with
          conditioning mask `0` (fully denoised) and a RoPE temporal extent of exactly one pixel frame. These are the
          keyframe slots that give DFR its extra frames; `slot_initial_latents` seeds their content.
        - `reference_latents` appends the stage-1 half-resolution latent as a fully clean IC-LoRA reference, with
          spatial coordinates scaled by `reference_downscale_factor` so it maps into the target coordinate space.

        Appended conditioning tokens carry their content in `clean_latents` and a zero placeholder in `latents`, while
        keyframe slots carry their seed in `latents` and zeros in `clean_latents` -- the returned `latents` are the
        noised mix of the two (see the noising step at the end of this method).

        Args:
            conditions (`list[LTX2VideoCondition]`, *optional*):
                Frame-level image / video conditions, positioned by latent index.
            condition_latents (`list[tuple[int, torch.Tensor, float, int]]`, *optional*):
                Already-encoded stand-in for `conditions`, as `(pixel_frame_index, latent, strength,
                num_pixel_frames)`. Pixel rather than latent index, because a temporal refine round scales a
                condition's position by `2 ** round` and the result does not generally land on a latent boundary --
                only an appended keyframe token can sit there, and it is placed by pixel. `pixel_frame_index == 0`
                still means "replace the first frame".
            keyframe_latents (`list[tuple[int, torch.Tensor, float]]`, *optional*):
                Already-encoded keyframe guidance as `(pixel_frame_index, latent, strength)`, where `latent` has shape
                `(batch_size, num_channels_latents, 1, latent_height, latent_width)`. Used by the temporal refine
                rounds to pin the seam keyframes carried in from the previous round.
            slot_frame_indices (`list[int]`, *optional*):
                Pixel-frame positions of the generated keyframe slots.
            slot_initial_latents (`torch.Tensor`, *optional*):
                `(batch_size, num_channels_latents, len(slot_frame_indices), latent_height, latent_width)` content
                written into the slot tokens before noising.
            reference_latents (`torch.Tensor`, *optional*):
                `(batch_size, num_channels_latents, F, H, W)` IC-LoRA reference latent.
            reference_downscale_factor (`int`, defaults to `1`):
                Ratio between the target and the reference resolution.
            latents (`torch.Tensor`, *optional*):
                `(batch_size, num_channels_latents, F, H, W)` initial content for the base tokens. Public pipeline
                latents are raw (denormalized); pass `latents_normalized=False` at that boundary. Tile loops that
                already sit in VAE-normalized space leave the default.
            latents_normalized (`bool`, defaults to `True`):
                Whether `latents`, `slot_initial_latents`, `reference_latents`, and `keyframe_latents` are already
                VAE-normalized. Internal tile loops pass `True`; each pipeline `__call__` passes `False`.
            noise_scale (`float`, defaults to `1.0`):
                Noise level the unconditioned tokens are initialized at, i.e. the schedule's first sigma.

        Returns:
            `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, slice | None]`:
                `(latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, slot_token_slice)`.
                `slot_token_slice` indexes the generated keyframe slot tokens in the packed sequence, or `None` when no
                slots were requested.
        """
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        patch_size = self.transformer_spatial_patch_size
        patch_size_t = self.transformer_temporal_patch_size

        # `randn_tensor` draws per batch element from a list of generators, but the VAE's latent distribution is
        # sampled once for a batch-1 condition tensor, so that call takes a single generator.
        encode_generator = generator[0] if isinstance(generator, list) else generator

        latents = self._maybe_normalize_video_latents(latents, latents_normalized)
        slot_initial_latents = self._maybe_normalize_video_latents(slot_initial_latents, latents_normalized)
        reference_latents = self._maybe_normalize_video_latents(reference_latents, latents_normalized)
        if keyframe_latents:
            keyframe_latents = [
                (position, self._maybe_normalize_video_latents(latent, latents_normalized), strength)
                for position, latent, strength in keyframe_latents
            ]

        if latents is None:
            # NOTE: zeros rather than a Gaussian sample, because the per-token noise level is only known once the
            # conditioning mask below is complete.
            latents = torch.zeros(
                (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width),
                device=device,
                dtype=dtype,
            )
        latents = self._pack_latents(latents.to(device=device, dtype=dtype), patch_size, patch_size_t)
        conditioning_mask = latents.new_zeros((*latents.shape[:2], 1))
        clean_latents = torch.zeros_like(latents)

        # Frame conditions: encode each one, then either overwrite the first-frame tokens or queue it as a keyframe.
        if condition_latents is None:
            condition_latents = self.encode_conditions(
                conditions, height, width, num_frames, device=device, dtype=dtype, generator=encode_generator
            )
        appended_keyframes: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        for pixel_frame_idx, encoded, strength, num_pixel_frames in condition_latents:
            encoded = encoded.to(device=device, dtype=dtype)
            # Conditions are preprocessed as batch-1 tensors and shared across the batch.
            condition_tokens = self._pack_latents(encoded, patch_size, patch_size_t).expand(batch_size, -1, -1)

            if pixel_frame_idx == 0:
                # Overwrite the clean target and mask only; the noisy sequence keeps the base-grid seed.
                num_condition_tokens = condition_tokens.shape[1]
                conditioning_mask[:, :num_condition_tokens] = strength
                clean_latents[:, :num_condition_tokens] = condition_tokens
                continue

            coords = self._prepare_keyframe_coords(
                keyframe_latent_num_frames=encoded.shape[2],
                keyframe_latent_height=encoded.shape[3],
                keyframe_latent_width=encoded.shape[4],
                pixel_frame_idx=pixel_frame_idx,
                num_pixel_frames=num_pixel_frames,
                fps=frame_rate,
                device=device,
            )
            appended_keyframes.append((condition_tokens, coords, strength))

        # Pre-encoded keyframe guidance carried in from a previous temporal round.
        for pixel_frame_index, keyframe_latent, strength in keyframe_latents or []:
            keyframe_latent = keyframe_latent.to(device=device, dtype=dtype)
            appended_keyframes.append(
                (
                    self._pack_latents(keyframe_latent, patch_size, patch_size_t),
                    self._prepare_keyframe_coords(
                        keyframe_latent_num_frames=keyframe_latent.shape[2],
                        keyframe_latent_height=keyframe_latent.shape[3],
                        keyframe_latent_width=keyframe_latent.shape[4],
                        pixel_frame_idx=pixel_frame_index,
                        num_pixel_frames=1,
                        fps=frame_rate,
                        device=device,
                    ),
                    strength,
                )
            )

        appended_coords = []
        # Guidance keyframes carry given content and are not marked. The learned embedding is for generated
        # single-pixel-frame latents (causal first frame + slots).
        keyframes_mask = torch.zeros_like(conditioning_mask)
        # Causal encoding gives the first latent frame a temporal stride of 1, so it is marked like a slot.
        tokens_per_latent_frame = latent_height * latent_width
        keyframes_mask[:, :tokens_per_latent_frame] = 1.0
        for tokens, coords, strength in appended_keyframes:
            latents = torch.cat([latents, torch.zeros_like(tokens)], dim=1)
            clean_latents = torch.cat([clean_latents, tokens], dim=1)
            conditioning_mask = torch.cat(
                [conditioning_mask, conditioning_mask.new_full((batch_size, tokens.shape[1], 1), float(strength))],
                dim=1,
            )
            keyframes_mask = torch.cat([keyframes_mask, keyframes_mask.new_zeros((batch_size, tokens.shape[1], 1))], 1)
            appended_coords.append(coords)

        # Generated keyframe slots: empty, fully-denoised single-pixel-frame token blocks the model fills in.
        slot_token_slice = None
        if slot_frame_indices:
            if slot_initial_latents is None:
                num_slot_tokens = tokens_per_latent_frame * len(slot_frame_indices)
                slot_tokens = latents.new_zeros((batch_size, num_slot_tokens, latents.shape[2]))
            else:
                slot_initial_latents = slot_initial_latents.to(device=device, dtype=dtype)
                slot_tokens = torch.cat(
                    [
                        self._pack_latents(slot_initial_latents[:, :, index : index + 1], patch_size, patch_size_t)
                        for index in range(slot_initial_latents.shape[2])
                    ],
                    dim=1,
                )
            slot_token_slice = slice(latents.shape[1], latents.shape[1] + slot_tokens.shape[1])

            latents = torch.cat([latents, slot_tokens], dim=1)
            clean_latents = torch.cat([clean_latents, torch.zeros_like(slot_tokens)], dim=1)
            conditioning_mask = torch.cat(
                [conditioning_mask, conditioning_mask.new_zeros((batch_size, slot_tokens.shape[1], 1))], dim=1
            )
            keyframes_mask = torch.cat(
                [keyframes_mask, keyframes_mask.new_ones((batch_size, slot_tokens.shape[1], 1))], dim=1
            )
            appended_coords.extend(
                self._prepare_keyframe_coords(
                    keyframe_latent_num_frames=1,
                    keyframe_latent_height=latent_height,
                    keyframe_latent_width=latent_width,
                    pixel_frame_idx=position,
                    num_pixel_frames=1,
                    fps=frame_rate,
                    device=device,
                )
                for position in slot_frame_indices
            )

        # IC-LoRA reference: the stage-1 half-resolution latent, held fully clean.
        if reference_latents is not None:
            reference_latents = reference_latents.to(device=device, dtype=dtype)
            reference_tokens = self._pack_latents(reference_latents, patch_size, patch_size_t)
            reference_coords = self.transformer.rope.prepare_video_coords(
                batch_size=1,
                num_frames=reference_latents.shape[2],
                height=reference_latents.shape[3],
                width=reference_latents.shape[4],
                device=device,
                fps=frame_rate,
            )
            reference_coords[:, 1:, :, :] = reference_coords[:, 1:, :, :] * reference_downscale_factor

            latents = torch.cat([latents, torch.zeros_like(reference_tokens)], dim=1)
            clean_latents = torch.cat([clean_latents, reference_tokens], dim=1)
            conditioning_mask = torch.cat(
                [conditioning_mask, conditioning_mask.new_ones((batch_size, reference_tokens.shape[1], 1))], dim=1
            )
            keyframes_mask = torch.cat(
                [keyframes_mask, keyframes_mask.new_zeros((batch_size, reference_tokens.shape[1], 1))], dim=1
            )
            appended_coords.append(reference_coords)

        video_coords = self.transformer.rope.prepare_video_coords(
            batch_size, latent_num_frames, latent_height, latent_width, device, fps=frame_rate
        )
        if appended_coords:
            appended = torch.cat(appended_coords, dim=2).expand(batch_size, -1, -1, -1)
            video_coords = torch.cat([video_coords, appended], dim=2)

        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        latents = noise * noise_scale + latents * (1 - noise_scale)
        latents = clean_latents * conditioning_mask + latents * (1 - conditioning_mask)

        return latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, slot_token_slice

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.prepare_audio_latents
    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,  # 1 is just a dummy value
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            # latents expected to be unpacked (4D) with shape [B, C, L, M]
            latents = self._pack_audio_latents(latents)
            latents = self._normalize_audio_latents(latents, self.audio_vae.latents_mean, self.audio_vae.latents_std)
            latents = self._create_noised_state(latents, noise_scale, generator)
            return latents.to(device=device, dtype=dtype)

        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Sample in packed shape (B, L, C * M), following the original LTX-2.X code
        packed_shape = (batch_size, audio_latent_length, num_channels_latents * latent_mel_bins)
        latents = randn_tensor(packed_shape, generator=generator, device=device, dtype=dtype)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._create_noised_state
    def _create_noised_state(
        latents: torch.Tensor, noise_scale: float | torch.Tensor, generator: torch.Generator | None = None
    ):
        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        noised_latents = noise_scale * noise + (1 - noise_scale) * latents
        return noised_latents

    def _unpack_video_latents(self, tokens: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
        """Unpack a `(batch_size, tokens, channels)` block onto this transformer's patch grid."""
        return self._unpack_latents(
            tokens,
            num_frames,
            height,
            width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

    def upsample_latents(self, latents: torch.Tensor, upsampler: LTX2LatentUpsamplerModel) -> torch.Tensor:
        """Run `upsampler` on normalized latents, round-tripping through raw VAE latent space as it expects."""
        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )
        latents = upsampler(latents.to(upsampler.dtype))
        return self._normalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )

    def encode_conditions(
        self,
        conditions: list[LTX2VideoCondition] | None,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> list[tuple[int, torch.Tensor, float, int]]:
        """
        Preprocess and VAE-encode frame conditions, positioned by pixel frame.

        Returns `(pixel_frame_index, latent, strength, num_pixel_frames)` per condition, ready for
        [`~LTX2DFRPipeline.prepare_latents`]'s `condition_latents`. Encoding is kept separate from placement because
        the temporal refine rounds scale a condition's position by `2 ** round` and re-base it per tile, and should not
        re-encode the same still once per tile to do so.

        The returned index is on `num_frames`' own pixel grid; carrying it onto a refined canvas is the caller's job.
        """
        condition_frames, condition_strengths, condition_indices, condition_pixel_frames = self.preprocess_conditions(
            conditions, height, width, num_frames, device=device
        )
        encoded = []
        for pixels, strength, latent_index, num_pixel_frames in zip(
            condition_frames, condition_strengths, condition_indices, condition_pixel_frames
        ):
            latent = self._normalize_latents(
                retrieve_latents(self.vae.encode(pixels), generator=generator, sample_mode="argmax"),
                self.vae.latents_mean,
                self.vae.latents_std,
            ).to(device=device, dtype=dtype)
            pixel_index = 0 if latent_index == 0 else (latent_index - 1) * self.vae_temporal_compression_ratio + 1
            encoded.append((pixel_index, latent, strength, num_pixel_frames))
        return encoded

    def denoise(
        self,
        latents: torch.Tensor,
        conditioning_mask: torch.Tensor,
        clean_latents: torch.Tensor,
        video_coords: torch.Tensor,
        keyframes_mask: torch.Tensor,
        prompt_embeds: torch.Tensor,
        audio_prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigmas: list[float],
        frame_rate: float,
        audio_latents: torch.Tensor,
        freeze_audio: bool = False,
        video_tile_plan: list | None = None,
        generator: torch.Generator | None = None,
        use_cross_timestep: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        progress_bar=None,
        step_offset: int = 0,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one DFR denoising pass over `sigmas` and return `(latents, audio_latents)`, both still packed.

        The distilled schedule is used without classifier-free guidance, so this is a single transformer call per step.
        Every pass runs both streams, because the video branch needs the cross-modal attention even where the audio it
        produces is thrown away. `freeze_audio=True` keeps the audio stream at sigma 0 (no Euler step) so video can
        still cross-attend to it — the temporal refine tiles and the epilogue use this to follow stage-1 speech without
        each tile re-denoising a different audio realization.

        After the x0 conditioning blend, the velocity is `(latents - denoised) / sigma` so an RF step
        `x0 = x - σ v` recovers the blended `denoised`. Stage 1 / 2 / the epilogue keep
        [`FlowMatchEulerDiscreteScheduler`] and take that Euler step as-is. Temporal refine swaps in
        [`LTXEulerAncestralRFScheduler`] (`eta=0.5`); that step renoises every token, so the conditioning
        blend is applied again afterwards or strength-0.95 seam anchors erode.

        Args:
            sigmas (`list[float]`):
                Noise schedule for this pass, without the terminal `0.0`.
            freeze_audio (`bool`, *optional*, defaults to `False`):
                Hold `audio_latents` clean (timestep/sigma 0, no audio Euler step) while still running audio-to-video
                cross-attention. Ignored when `audio_latents` is `None`.
            video_tile_plan (`list`, *optional*):
                Per-tile token plan from [`video_tile_plan`][`~diffusers.pipelines.ltx2.dfr_layout.video_tile_plan`].
                When given, each step runs the transformer once per tile and blends the predictions, so the sampler
                still steps a single full canvas and the tiles agree on their overlaps at every step.
            generator (`torch.Generator`, *optional*):
                Forwarded to [`LTXEulerAncestralRFScheduler.step`]. Temporal tiles pass a per-tile seed so ancestral
                draws do not share a stream or consume the state the next tile's initial noising reads. Distilled Euler
                does not read it.
            step_offset (`int`):
                Index of this pass's first step within the pipeline's whole schedule, used for `callback_on_step_end`
                and the shared progress bar.
        """
        device = latents.device
        # The audio stream's own length; passing it separately only invites the two disagreeing.
        audio_num_frames = audio_latents.shape[1]

        timesteps = self._inference_timesteps(sigmas, device)
        # The video and audio streams step the same schedule, but `step` tracks its own index, so audio needs its own
        # scheduler instance.
        audio_scheduler = copy.deepcopy(self.scheduler)

        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, device
        )

        for index, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep_scalar = t.expand(latents.shape[0])
            # Conditioned tokens see a proportionally lower noise level, which is what holds them near their clean
            # content while the rest of the sequence denoises.
            video_timestep = timestep_scalar.unsqueeze(-1) * (1 - conditioning_mask.squeeze(-1))
            if freeze_audio:
                audio_timestep = torch.zeros(audio_latents.shape[0], device=device, dtype=t.dtype)
            else:
                audio_timestep = audio_scheduler.timesteps[index].expand(audio_latents.shape[0])

            # Everything a tile shares with the full canvas: audio, text, and the schedule's scalar sigma.
            transformer_kwargs = {
                "encoder_hidden_states": prompt_embeds,
                "audio_encoder_hidden_states": audio_prompt_embeds,
                "audio_timestep": audio_timestep,
                "sigma": timestep_scalar,
                "audio_sigma": audio_timestep,
                "encoder_attention_mask": prompt_attention_mask,
                "audio_encoder_attention_mask": prompt_attention_mask,
                "fps": frame_rate,
                "audio_num_frames": audio_num_frames,
                "audio_coords": audio_coords,
                "use_cross_timestep": use_cross_timestep,
                "attention_kwargs": attention_kwargs,
                "return_dict": False,
            }

            if video_tile_plan is None:
                noise_pred_video, noise_pred_audio = self.transformer(
                    hidden_states=latents.to(prompt_embeds.dtype),
                    audio_hidden_states=audio_latents.to(prompt_embeds.dtype),
                    timestep=video_timestep,
                    video_keyframes_mask=keyframes_mask,
                    video_coords=video_coords,
                    **transformer_kwargs,
                )
            else:
                # Blending the velocity is the same as blending x0: `x0 = x - v * sigma` is affine in `v`, every tile
                # reads the same `latents` and the same per-token sigma, and the weights sum to one.
                noise_pred_video = torch.zeros_like(latents, dtype=torch.float32)
                noise_pred_audio = None
                for tile in video_tile_plan:
                    keep = tile.keep
                    tile_pred, tile_audio_pred = self.transformer(
                        hidden_states=latents[:, keep].to(prompt_embeds.dtype),
                        audio_hidden_states=audio_latents.to(prompt_embeds.dtype),
                        timestep=video_timestep[:, keep],
                        video_keyframes_mask=keyframes_mask[:, keep],
                        video_coords=tile.coords,
                        **transformer_kwargs,
                    )
                    noise_pred_video.index_add_(
                        1, keep, tile_pred.float() * tile.weights.to(torch.float32).view(1, -1, 1)
                    )
                    if tile_audio_pred is not None:
                        # Every tile saw the whole audio under a different video context; average them.
                        contribution = tile_audio_pred.float() / len(video_tile_plan)
                        noise_pred_audio = (
                            contribution if noise_pred_audio is None else noise_pred_audio + contribution
                        )
                noise_pred_video = noise_pred_video.to(latents.dtype)

            # Conditioning is applied in x0 space. Convert velocity -> x0 with each token's own noise level: a token
            # held at strength `s` sits at `(1 - s) * sigma`, so the scalar schedule sigma would mis-scale it.
            # Deliberately not the scheduler's `per_token_timesteps` path: that steps each token from its own sigma to
            # the nearest schedule sigma below it, whereas a conditioned token has to stay pinned while the rest of the
            # sequence advances on the shared schedule.
            sigma = self.scheduler.sigmas[index]
            per_token_sigma = (video_timestep / self.scheduler.config.num_train_timesteps).unsqueeze(-1)
            denoised = latents.float() - noise_pred_video.float() * per_token_sigma
            denoised = denoised * (1 - conditioning_mask) + clean_latents.float() * conditioning_mask

            ancestral = (
                isinstance(self.scheduler, LTXEulerAncestralRFScheduler) and float(self.scheduler.config.eta) > 0
            )
            if ancestral:
                latents = self.scheduler.step(
                    (latents.float() - denoised) / sigma, t, latents, generator=generator, return_dict=False
                )[0]
                # Ancestral Euler noises every token, so re-apply the blend or strength-0.95 seam anchors erode.
                latents = (latents.float() * (1 - conditioning_mask) + clean_latents.float() * conditioning_mask).to(
                    latents.dtype
                )
            else:
                latents = self.scheduler.step((latents.float() - denoised) / sigma, t, latents, return_dict=False)[0]

            if not freeze_audio:
                audio_sigma = audio_scheduler.sigmas[index]
                audio_denoised = audio_latents.float() - noise_pred_audio.float() * audio_sigma
                audio_latents = audio_scheduler.step(
                    (audio_latents.float() - audio_denoised) / audio_sigma, t, audio_latents, return_dict=False
                )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs or []:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, step_offset + index, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if progress_bar is not None:
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

        return latents, audio_latents

    def _unpack_video_and_slots(
        self,
        packed: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        slot_token_slice: slice | None,
        num_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video = self._unpack_video_latents(
            packed[:, : latent_num_frames * latent_height * latent_width],
            latent_num_frames,
            latent_height,
            latent_width,
        )
        keyframes = None
        if slot_token_slice is not None and num_slots:
            keyframes = self._unpack_video_latents(packed[:, slot_token_slice], num_slots, latent_height, latent_width)
        return video, keyframes

    def _public_audio_from_packed(self, audio_latents: torch.Tensor, audio_num_frames: int) -> torch.Tensor:
        latent_mel_bins = self.audio_mel_bins // self.audio_vae_mel_compression_ratio
        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        return self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)

    def _pack_public_audio(self, audio_latents: torch.Tensor) -> torch.Tensor:
        packed = self._pack_audio_latents(audio_latents)
        return self._normalize_audio_latents(packed, self.audio_vae.latents_mean, self.audio_vae.latents_std)

    def _finalize_output(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        keyframe_latents: torch.Tensor | None,
        keyframe_positions: list[int] | None,
        output_type: str,
        return_dict: bool,
        output_cls,
        requested_frames: int | None,
        playback_fps: float,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        generator: torch.Generator | list[torch.Generator] | None,
        prompt_embeds: torch.Tensor,
    ):
        """Denormalize, optionally trim and decode, and wrap the DFR return value.

        `video_latents` / `keyframe_latents` are VAE-normalized 5D. `audio_latents` is unpacked denormalized 4D
        (the public audio contract). Latent output is untrimmed so a slot on the padded canvas is not dropped;
        decode trims to `requested_frames` when that is set.
        """
        output_cls = output_cls or LTX2DFRPipelineOutput
        batch_size = video_latents.shape[0]
        device = video_latents.device

        if output_type != "latent" and requested_frames is not None:
            video_latents = trim_canvas(video_latents, requested_frames, self.vae_temporal_compression_ratio)
            num_frames = requested_frames
        else:
            num_frames = (video_latents.shape[2] - 1) * self.vae_temporal_compression_ratio + 1

        video = self._denormalize_latents(
            video_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )
        keyframes = None
        if keyframe_latents is not None:
            keyframes = self._denormalize_latents(
                keyframe_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )

        if output_type == "latent":
            audio = audio_latents
        else:
            video_latents = video_latents.to(prompt_embeds.dtype)
            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(
                    video_latents.shape, generator=generator, device=device, dtype=video_latents.dtype
                )
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=video_latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=video_latents.dtype)[
                    :, None, None, None, None
                ]
                video_latents = (1 - decode_noise_scale) * video_latents + decode_noise_scale * noise

            video_latents = self._denormalize_latents(
                video_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            video = self.vae.decode(video_latents.to(self.vae.dtype), timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            audio = self.vocoder(self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0])
            audio_samples = min(
                audio.shape[-1], round(num_frames / playback_fps * self.vocoder.config.output_sampling_rate)
            )
            audio = audio[..., :audio_samples]

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)

        return output_cls(frames=video, audio=audio, keyframes=keyframes, keyframe_positions=keyframe_positions)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt
