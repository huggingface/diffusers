# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import (
    Cosmos3OmniTransformer,
)
from ...schedulers import UniPCMultistepScheduler
from ...utils import BaseOutput, is_cosmos_guardrail_available
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline


if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "`cosmos_guardrail` is not installed. Please install it to use the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )


# ============================================================================
# Sequence layout: data structures + builders for the joint token sequence
# ============================================================================


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for text tokens.

    For text tokens, all three axes (temporal, height, width) share the same monotonically increasing position IDs,
    starting from ``temporal_offset``.
    """
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)

    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()  # [3,num_tokens]
    next_temporal_offset = temporal_offset + num_tokens
    return mrope_ids, next_temporal_offset


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for VAE vision tokens (image/video latents)."""
    fps_modulation_enabled = fps is not None and grid_t > 1
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    max_position = mrope_ids.max().item()
    next_temporal_offset = math.ceil(max_position) + 1
    return mrope_ids, next_temporal_offset


# ============================================================================
# Pipeline output + IO helpers
# ============================================================================


_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."


@dataclass
class Cosmos3OmniPipelineOutput(BaseOutput):
    """Output dataclass for :class:`Cosmos3OmniDiffusersPipeline`.

    Attributes:
        video: The generated video. The exact type depends on ``output_type``
            passed to the pipeline: a list of PIL frames for ``"pil"`` (default), an ``np.ndarray`` of shape ``[T, H,
            W, C]`` for ``"np"``, a ``torch.Tensor`` of shape ``[T, C, H, W]`` for ``"pt"``, or a raw latent tensor
            when ``output_type="latent"``.
        sound: Decoded audio waveform of shape ``[C, N]``. ``None`` when
            ``enable_sound=False``.
    """

    video: Any
    sound: torch.Tensor | None = None


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


class Cosmos3OmniDiffusersPipeline(DiffusionPipeline):
    _optional_components = ["sound_tokenizer", "safety_checker"]
    _exclude_from_cpu_offload = ["safety_checker"]
    model_cpu_offload_seq = "transformer->vae->sound_tokenizer"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: Cosmos3OmniTransformer,
        text_tokenizer: AutoTokenizer,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        sound_tokenizer: Cosmos3AVAEAudioTokenizer | None = None,
        safety_checker: CosmosSafetyChecker | None = None,
        enable_safety_checker: bool = True,
    ):
        super().__init__()
        if enable_safety_checker:
            if safety_checker is None:
                safety_checker = CosmosSafetyChecker()
        else:
            safety_checker = None
        self.register_modules(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=vae,
            scheduler=scheduler,
            sound_tokenizer=sound_tokenizer,
            safety_checker=safety_checker,
        )
        # VAE latent normalization stats — precomputed in bfloat16 so `1/std` is
        # done in bfloat16 (matches Wan2pt2VAEInterface bit-for-bit).
        self._vae_dtype = torch.bfloat16
        self._vae_latents_mean = torch.tensor(vae.config.latents_mean, dtype=self._vae_dtype)
        self._vae_latents_inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=self._vae_dtype)

        # Image preprocessor for caller-supplied conditioning frames (PIL / tensor / numpy).
        self.vae_scale_factor_spatial = int(self.vae.config.scale_factor_spatial) if getattr(self, "vae", None) else 16
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial, resample="bilinear")

        self.llm_special_tokens = {
            "start_of_generation": text_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "eos_token_id": text_tokenizer.eos_token_id,
        }

        # Prompt-augmentation templates: appended inside `tokenize_prompt` so the LLM sees
        # the same metadata the model was trained with. Negative prompts use inverse templates.
        self.duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        self.image_resolution_template = "This image is of {height}x{width} resolution."
        self.video_resolution_template = "This video is of {height}x{width} resolution."
        self.inverse_duration_template = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
        self.inverse_image_resolution_template = "This image is not of {height}x{width} resolution."
        self.inverse_video_resolution_template = "This video is not of {height}x{width} resolution."

        # Recommended quality-control negative prompts are documented in the Cosmos3 docs
        # page (text2video / image2video). When the caller passes None we fall back to "".

    # TODO YiYi & Daniel: fix for this use case in the base class
    def _get_execution_device(self) -> torch.device:
        # `self._execution_device` walks `self.components` and ultimately falls back to
        # `self.device`, which iterates modules in sorted order and ignores
        # `_exclude_from_cpu_offload`. With `safety_checker` registered, that path picks
        # up `CosmosSafetyChecker.device` — which either raises `AttributeError`
        # (silently surfaced as "no attribute `_execution_device`") or returns `cpu`
        # because the auto-instantiated checker is on CPU. In both cases the pipeline
        # ends up running on the wrong device. Walk the actual compute modules first.
        for component in (self.transformer, self.vae, self.sound_tokenizer):
            if not isinstance(component, torch.nn.Module):
                continue

            for module in component.modules():
                hook = getattr(module, "_hf_hook", None)
                execution_device = getattr(hook, "execution_device", None)
                if execution_device is not None:
                    return torch.device(execution_device)

            try:
                return next(component.parameters()).device
            except StopIteration:
                continue

        try:
            return self._execution_device
        except AttributeError:
            return torch.device("cpu")

    def _encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] → normalized latents [B,z_dim,T//4,H//16,W//16]. Bit-for-bit
        matches Wan2pt2VAEInterface; no autocast (WanVAE was trained with is_amp=False)."""
        in_dtype = x.dtype
        dtype = self._vae_dtype
        mean = self._vae_latents_mean.to(device=x.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=x.device, dtype=dtype)
        raw_mu = retrieve_latents(self.vae.encode(x.to(dtype)), sample_mode="argmax")
        return ((raw_mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)).to(in_dtype)

    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a sound latent ``[C, T]`` to a waveform ``[audio_ch, N]``.

        Adds/removes the batch dimension expected by the sound tokenizer decoder.
        """
        assert self.sound_tokenizer is not None
        decoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        waveform = self.sound_tokenizer.decode(latent.unsqueeze(0).to(decoder_dtype))  # [1, audio_ch, N]
        return waveform.squeeze(0)  # [audio_ch, N]

    # ------------------------------------------------------------------
    # Joint-sequence packing — text is packed once per prompt (it is invariant
    # across denoising steps); vision and sound are repacked each step. The
    # per-step assembly happens inline in the denoising loop in __call__.
    # ------------------------------------------------------------------

    def _prepare_text_segment(
        self,
        input_ids: list[int],
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Build the text segment of the joint sequence.

        Text packing is invariant across denoising steps and across cond/uncond passes for a given prompt, so this is
        called once per prompt right after tokenization and the result is reused inside the denoising loop. The
        returned dict carries transformer-facing fields (``input_ids``, ``text_indexes``, ``und_len``) along with the
        assembly helpers needed by the per-step vision/sound packing — ``text_mrope_ids`` for the joint mRoPE concat,
        and ``vision_start_temporal_offset`` which both vision and sound mRoPE consume as their temporal offset (the
        two modalities are temporal siblings, not sequential).
        """
        config = self.transformer.config
        und_len = len(input_ids)
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=config.enable_fps_modulation,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
            "und_len": und_len,
            "text_mrope_ids": text_mrope_ids.to(device),
            "vision_start_temporal_offset": next_mrope_offset + config.unified_3d_mrope_temporal_modality_margin,
        }

    def _prepare_vision_segment(
        self,
        input_vision_tokens: torch.Tensor,
        has_image_condition: bool,
        mrope_offset: int | float,
        vision_fps: float | None,
        curr: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Build the static portion of the vision segment of the joint sequence.

        Step-varying fields (``vision_tokens`` and ``vision_timesteps``) are NOT included here — the caller splices
        them in inside the denoising loop. The method is called once per (cond/uncond) prompt before the loop, since
        everything else only depends on the prompt length and the vision shape.
        """
        config = self.transformer.config
        latent_patch_size = config.latent_patch_size
        _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
        patch_h = math.ceil(latent_h / latent_patch_size)
        patch_w = math.ceil(latent_w / latent_patch_size)
        num_vision_tokens = latent_t * patch_h * patch_w

        noisy_start = 1 if has_image_condition else 0
        noisy_frame_indexes = torch.arange(noisy_start, latent_t, device=device, dtype=torch.long)

        frame_token_stride = patch_h * patch_w
        mse_loss_indexes: list[int] = []
        for frame_idx in range(noisy_start, latent_t):
            frame_start = curr + frame_idx * frame_token_stride
            mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

        effective_fps = vision_fps if config.enable_fps_modulation else None
        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=self.vae.config.scale_factor_temporal,
        )

        return {
            # Transformer-facing fields (vision_tokens and vision_timesteps spliced per step).
            "vision_token_shapes": [(latent_t, patch_h, patch_w)],
            "vision_sequence_indexes": torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device),
            "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
            "vision_noisy_frame_indexes": [noisy_frame_indexes],
            # Assembly helpers (consumed inline before the transformer call).
            "vision_mrope_ids": vision_mrope_ids.to(device),
            "num_vision_tokens": num_vision_tokens,
            "num_noisy_vision_tokens": (latent_t - noisy_start) * frame_token_stride,
        }

    def _prepare_sound_segment(
        self,
        input_sound_tokens: torch.Tensor,
        mrope_offset: int | float,
        sound_fps: float | None,
        curr: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Build the static portion of the sound segment of the joint sequence.

        Step-varying fields (``sound_tokens`` and ``sound_timesteps``) are spliced in by the caller inside the
        denoising loop; everything here depends only on the prompt length and the sound shape. All sound frames are
        noisy.
        """
        config = self.transformer.config
        _, sound_len = input_sound_tokens.shape

        effective_fps = sound_fps if config.enable_fps_modulation else None
        sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=1,
        )

        sequence_indexes = torch.arange(curr, curr + sound_len, dtype=torch.long, device=device)
        return {
            # Transformer-facing fields (sound_tokens and sound_timesteps spliced per step).
            "sound_token_shapes": [(sound_len, 1, 1)],
            "sound_sequence_indexes": sequence_indexes,
            "sound_mse_loss_indexes": sequence_indexes.clone(),
            "sound_noisy_frame_indexes": [torch.arange(sound_len, device=device, dtype=torch.long)],
            # Assembly helpers (consumed inline before the transformer call).
            "sound_mrope_ids": sound_mrope_ids.to(device),
            "sound_len": sound_len,
        }

    def prepare_latents(
        self,
        image: torch.Tensor | None = None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        latents: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_sound: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        float,
        float | None,
    ]:
        """Build conditioning + initial noise for a single sample.

        Returns:
            ``(vision_latents, sound_latents, fps_vision, fps_sound)``. ``vision_latents`` is the noisy vision tensor;
            ``sound_latents`` is the noisy sound tensor (``None`` unless ``enable_sound`` was set). The FPS scalars
            feed the per-step :meth:`_pack_vision_tokens` / :meth:`_pack_sound_tokens` calls in the denoising loop.
        """
        is_image = num_frames == 1
        has_image_condition = image is not None and not is_image

        # video_processor.preprocess handles PIL/np/tensor → [1, 3, H, W] in [-1, 1], resized to (height, width).
        conditioning_frame_2d: torch.Tensor | None = None
        if image is not None:
            conditioning_frame_2d = self.video_processor.preprocess(image, height=height, width=width).to(
                device=device, dtype=dtype
            )

        # Build the vision conditioning tensor (always [1, 3, T, H, W], in [-1, 1], on device).
        if is_image:
            vision_tensor = (
                conditioning_frame_2d.unsqueeze(2)  # [1, 3, 1, H, W]
                if conditioning_frame_2d is not None
                else torch.zeros(1, 3, 1, height, width, dtype=dtype, device=device)
            )
        else:
            vision_tensor = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            if conditioning_frame_2d is not None:
                # Single conditioning frame at t=0, repeat-pad the rest with that same frame.
                vision_tensor[:, :, 0] = conditioning_frame_2d
                if num_frames > 1:
                    vision_tensor[:, :, 1:] = conditioning_frame_2d.unsqueeze(2).expand(-1, -1, num_frames - 1, -1, -1)

        x0_tokens_vision = self._encode_video(vision_tensor).contiguous().float()
        vision_shape = tuple(x0_tokens_vision.shape)

        x0_tokens_sound: torch.Tensor | None = None
        fps_sound: float | None = None
        if enable_sound:
            sound_dim = self.transformer.config.sound_dim
            fps_sound = float(self.transformer.config.sound_latent_fps)
            n_audio_samples = int(num_frames / fps * self.sound_tokenizer.config.sampling_rate)
            hop_size = self.sound_tokenizer._hop_size
            T_sound = (n_audio_samples + hop_size - 1) // hop_size
            x0_tokens_sound = torch.zeros(sound_dim, T_sound, device=device, dtype=dtype)

        # Vision conditioning mask [latent_t, 1, 1]: frame 0 anchored when image-conditioning, rest noisy.
        vision_condition_mask = torch.zeros((x0_tokens_vision.shape[2], 1, 1), device=device, dtype=dtype)
        if has_image_condition:
            vision_condition_mask[0, 0, 0] = 1.0

        if latents is None:
            pure_noise = randn_tensor(vision_shape, generator=generator, device=device, dtype=dtype)
            latents = (
                vision_condition_mask * x0_tokens_vision.to(device=device, dtype=dtype)
                + (1.0 - vision_condition_mask) * pure_noise
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        sound_condition_mask: torch.Tensor | None = None
        if enable_sound and x0_tokens_sound is not None:
            # All sound frames are noisy, so the conditioning mask is always zero.
            sound_condition_mask = torch.zeros((x0_tokens_sound.shape[1], 1), device=device, dtype=dtype)
            if sound_latents is None:
                pure_noise_sound = randn_tensor(
                    tuple(x0_tokens_sound.shape), generator=generator, device=device, dtype=dtype
                )
                sound_latents = (
                    sound_condition_mask.T * x0_tokens_sound + (1.0 - sound_condition_mask.T) * pure_noise_sound
                )
            else:
                sound_latents = sound_latents.to(device=device, dtype=dtype)

        return latents, sound_latents, fps, fps_sound, vision_condition_mask, sound_condition_mask

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height: int,
        width: int,
        num_frames: int,
        guidance_scale: float,
        enable_sound: bool,
    ) -> None:
        if not isinstance(prompt, (str, list)) or (
            isinstance(prompt, list) and not all(isinstance(p, str) for p in prompt)
        ):
            raise ValueError(f"`prompt` must be a str or list of str, got {type(prompt).__name__}.")
        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(
                f"`negative_prompt` must be a str, list of str, or None, got {type(negative_prompt).__name__}."
            )
        if num_frames < 1:
            raise ValueError(f"`num_frames` must be >= 1, got {num_frames}.")
        sf = int(self.vae.config.scale_factor_spatial)
        if height % sf != 0 or width % sf != 0:
            raise ValueError(f"`height` and `width` must be multiples of {sf}, got ({height}, {width}).")

        if enable_sound:
            if self.sound_tokenizer is None:
                raise ValueError("`enable_sound=True` requires a sound-capable checkpoint with a `sound_tokenizer`.")
            if not getattr(self.transformer.config, "sound_gen", False):
                raise ValueError("`enable_sound=True` but the transformer was not trained with `sound_gen=True`.")

    def tokenize_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        use_system_prompt: bool = True,
        add_resolution_template: bool = True,
        add_duration_template: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Apply prompt-augmentation templates and tokenize cond/uncond prompts via the Qwen2 chat template.

        This pipeline does not run a separate text encoder: the joint Cosmos3 transformer consumes raw Qwen2 token IDs
        alongside vision (and optionally sound) tokens.

        When ``negative_prompt`` is ``None``, an empty string is used; the Cosmos3 docs page documents recommended
        quality-control negative prompts to pass explicitly for text2video / image2video. The duration and resolution
        templates are appended to the prompt, and inverse templates are appended to the negative prompt, when enabled.

        Returns:
            ``(cond_input_ids, uncond_input_ids)`` — token-id lists for this sample.
        """
        is_image = num_frames == 1

        if negative_prompt is None:
            negative_prompt = ""

        resolution_template = self.image_resolution_template if is_image else self.video_resolution_template
        inverse_resolution_template = (
            self.inverse_image_resolution_template if is_image else self.inverse_video_resolution_template
        )

        def _append(base: str, addition: str) -> str:
            base = base.rstrip(".")
            return f"{base}. {addition}" if base else addition

        def _apply_templates(text: str, is_negative: bool = False) -> str:
            if not is_image and add_duration_template:
                duration_template = self.inverse_duration_template if is_negative else self.duration_template
                text = _append(text, duration_template.format(duration=num_frames / fps, fps=fps))
            if add_resolution_template:
                template = inverse_resolution_template if is_negative else resolution_template
                text = _append(text, template.format(height=height, width=width))
            return text

        def _tokenize(text: str) -> list[int]:
            conversations = []
            if use_system_prompt:
                system_prompt = _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO
                conversations.append({"role": "system", "content": system_prompt})
            conversations.append({"role": "user", "content": text})
            return self.text_tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
            )
        
        def _add_special_tokens(input_ids: list[int]) -> list[int]:
            return list(input_ids) + [
            self.llm_special_tokens["eos_token_id"],
            self.llm_special_tokens["start_of_generation"],
        ]

        cond_encodings = _tokenize(_apply_templates(prompt))
        cond_input_ids = _add_special_tokens(cond_encodings.input_ids)
        uncond_encodings = _tokenize(_apply_templates(negative_prompt, is_negative=True))
        uncond_input_ids = _add_special_tokens(uncond_encodings.input_ids)
        return cond_input_ids, uncond_input_ids

    @staticmethod
    def _mask_velocity_predictions(
        preds_vision: list[torch.Tensor],
        preds_sound: list[torch.Tensor] | None,
        vision_condition_mask: list[torch.Tensor],
        sound_condition_mask: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zero out conditioning positions in the transformer's velocity predictions.

        ``preds_vision`` / ``preds_sound`` are returned per-sample by the transformer; the pipeline runs batch=1, so we
        take the first entry and apply ``1 - condition_mask`` to keep only the noisy positions where the model produces
        meaningful velocity.
        """
        pred_v = preds_vision[0]
        m_v = vision_condition_mask[0]
        noisy_mask_v = (1.0 - m_v).to(dtype=pred_v.dtype, device=pred_v.device)
        velocity_vision = pred_v * noisy_mask_v if noisy_mask_v.sum() > 0 else torch.zeros_like(pred_v)

        velocity_sound: torch.Tensor | None = None
        if preds_sound is not None and sound_condition_mask is not None:
            pred_s = preds_sound[0]
            cond_mask_s = sound_condition_mask[0]
            noisy_mask_s = (1.0 - cond_mask_s).T.to(dtype=pred_s.dtype, device=pred_s.device)
            velocity_sound = pred_s * noisy_mask_s if noisy_mask_s.sum() > 0 else torch.zeros_like(pred_s)

        return velocity_vision, velocity_sound

    def _apply_video_safety_check(self, video: Any, output_type: str, device: torch.device) -> Any:
        """Run the Cosmos video guardrail on a postprocessed video and return it in the same format.

        The guardrail (``CosmosSafetyChecker.check_video_safety``) expects ``np.uint8`` frames in ``[T, H, W, C]``
        layout. This helper handles the round-trip from the requested ``output_type`` (``"pil"`` / ``"np"`` / ``"pt"``)
        into that format and back. The checker may pixelate detected faces; if the content is blocked it returns
        ``None`` and we raise ``ValueError``. ``output_type="latent"`` should be filtered out by the caller.
        """
        if output_type == "pil":
            frames_uint8 = np.stack([np.array(frame) for frame in video], axis=0)
        elif output_type == "np":
            frames_uint8 = (video * 255).astype(np.uint8)
        elif output_type == "pt":
            frames_uint8 = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported output_type for safety check: {output_type}")

        self.safety_checker.to(device)
        try:
            checked = self.safety_checker.check_video_safety(frames_uint8)
        finally:
            self.safety_checker.to("cpu")
        if checked is None:
            raise ValueError(
                "Cosmos Guardrail detected unsafe content in the generated video. "
                "Please ensure that the generation abides by the NVIDIA Open Model License Agreement."
            )

        if output_type == "pil":
            return [Image.fromarray(frame) for frame in checked]
        if output_type == "np":
            return checked.astype(np.float32) / 255.0
        # output_type == "pt"
        return torch.from_numpy(checked.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        image: torch.Tensor | None = None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        enable_sound: bool = False,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        use_system_prompt: bool = True,
        callback_on_step_end: Callable[[int, int, dict[str, Any]], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        add_resolution_template: bool = True,
        add_duration_template: bool = True,
        enable_safety_check: bool = True,
    ) -> Cosmos3OmniPipelineOutput:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(prompt, negative_prompt, height, width, num_frames, guidance_scale, enable_sound)
        if not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        self._current_timestep = None
        self._interrupt = False

        # Pipeline supports a single sample at a time; collapse list-style inputs to a single string.
        if isinstance(prompt, list):
            prompt = prompt[0]
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0]

        device = self._get_execution_device()
        dtype = self.transformer.dtype

        if enable_safety_check and isinstance(self.safety_checker, CosmosSafetyChecker):
            self.safety_checker.to(device)
            try:
                if not self.safety_checker.check_text_safety(prompt):
                    raise ValueError(
                        f"Cosmos Guardrail detected unsafe text in the prompt: {prompt}. "
                        f"Please ensure that the prompt abides by the NVIDIA Open Model License Agreement."
                    )
            finally:
                self.safety_checker.to("cpu")

        # 2. Tokenize prompt (applies metadata templates and selects mode-specific default negative prompt)
        cond_input_ids, uncond_input_ids = self.tokenize_prompt(
            prompt,
            negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            use_system_prompt=use_system_prompt,
            add_resolution_template=add_resolution_template,
            add_duration_template=add_duration_template,
        )

        # 3. Pre-pack the text segment for each prompt — text packing is invariant
        # across denoising steps, so we do it once here and reuse inside the loop.
        cond_text_segment = self._prepare_text_segment(cond_input_ids, device=device)
        uncond_text_segment = self._prepare_text_segment(uncond_input_ids, device=device)

        # 4. Prepare latents (initial noise per modality + pack metadata)
        has_image_condition = image is not None and num_frames > 1
        latents, sound_latents, fps_vision, fps_sound, vision_condition_mask, sound_condition_mask = (
            self.prepare_latents(
                image=image,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                latents=latents,
                sound_latents=sound_latents,
                generator=generator,
                device=device,
                dtype=dtype,
                enable_sound=enable_sound,
            )
        )

        # 5. Pre-pack the static per-prompt vision / sound sequence segments. The only
        # fields that vary across denoising steps are the modality token tensors and the
        # per-modality timestep tensors; everything else only depends on prompt length
        # and modality shape, so we hoist it out of the loop and splice the step-varying
        # fields back in below.
        cond_vision_segment = self._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=has_image_condition,
            mrope_offset=cond_text_segment["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=cond_text_segment["und_len"],
            device=device,
        )
        cond_sound_segment: dict[str, Any] = {}
        if sound_latents is not None:
            cond_sound_segment = self._prepare_sound_segment(
                input_sound_tokens=sound_latents,
                mrope_offset=cond_text_segment["vision_start_temporal_offset"],
                sound_fps=fps_sound,
                curr=cond_text_segment["und_len"] + cond_vision_segment["num_vision_tokens"],
                device=device,
            )
        cond_mrope_segments = [cond_text_segment["text_mrope_ids"], cond_vision_segment["vision_mrope_ids"]]
        if cond_sound_segment:
            cond_mrope_segments.append(cond_sound_segment["sound_mrope_ids"])
        cond_packed_static = {
            **cond_text_segment,
            **cond_vision_segment,
            **cond_sound_segment,
            "position_ids": torch.cat(cond_mrope_segments, dim=1),
            "sequence_length": cond_text_segment["und_len"]
            + cond_vision_segment["num_vision_tokens"]
            + cond_sound_segment.get("sound_len", 0),
        }

        uncond_vision_segment = self._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=has_image_condition,
            mrope_offset=uncond_text_segment["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=uncond_text_segment["und_len"],
            device=device,
        )
        uncond_sound_segment: dict[str, Any] = {}
        if sound_latents is not None:
            uncond_sound_segment = self._prepare_sound_segment(
                input_sound_tokens=sound_latents,
                mrope_offset=uncond_text_segment["vision_start_temporal_offset"],
                sound_fps=fps_sound,
                curr=uncond_text_segment["und_len"] + uncond_vision_segment["num_vision_tokens"],
                device=device,
            )
        uncond_mrope_segments = [uncond_text_segment["text_mrope_ids"], uncond_vision_segment["vision_mrope_ids"]]
        if uncond_sound_segment:
            uncond_mrope_segments.append(uncond_sound_segment["sound_mrope_ids"])
        uncond_packed_static = {
            **uncond_text_segment,
            **uncond_vision_segment,
            **uncond_sound_segment,
            "position_ids": torch.cat(uncond_mrope_segments, dim=1),
            "sequence_length": uncond_text_segment["und_len"]
            + uncond_vision_segment["num_vision_tokens"]
            + uncond_sound_segment.get("sound_len", 0),
        }
        num_noisy_vision_tokens = cond_vision_segment["num_noisy_vision_tokens"]
        sound_len = cond_sound_segment.get("sound_len")

        # 6. Set timesteps. UniPCMultistepScheduler keeps per-step state (_step_index,
        # model_outputs history) on the instance, so audio gets its own copy.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sound_scheduler = copy.deepcopy(self.scheduler) if sound_latents is not None else None

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.item()

                # The transformer projections (proj_in / audio_proj_in) are bf16; cast the per-step
                # noisy tokens before packing so the modality tokens enter the model in the right dtype.
                vision_tokens = latents.to(device=device, dtype=dtype)
                sound_tokens = sound_latents.to(device=device, dtype=dtype) if sound_latents is not None else None
                # The static packs both report the same num_noisy_vision_tokens / sound_len, so a
                # single per-step timestep tensor per modality is shared by the cond / uncond passes.
                vision_timesteps = torch.full((num_noisy_vision_tokens,), timestep, device=device)
                sound_timesteps = (
                    torch.full((sound_len,), timestep, device=device) if sound_tokens is not None else None
                )

                # --- Conditional pass ---
                preds_vision, preds_sound = self.transformer(
                    input_ids=cond_packed_static["input_ids"],
                    text_indexes=cond_packed_static["text_indexes"],
                    position_ids=cond_packed_static["position_ids"],
                    und_len=cond_packed_static["und_len"],
                    sequence_length=cond_packed_static["sequence_length"],
                    vision_tokens=[vision_tokens],
                    vision_token_shapes=cond_packed_static["vision_token_shapes"],
                    vision_sequence_indexes=cond_packed_static["vision_sequence_indexes"],
                    vision_mse_loss_indexes=cond_packed_static["vision_mse_loss_indexes"],
                    vision_timesteps=vision_timesteps,
                    vision_noisy_frame_indexes=cond_packed_static["vision_noisy_frame_indexes"],
                    sound_tokens=[sound_tokens] if sound_tokens is not None else None,
                    sound_token_shapes=cond_packed_static.get("sound_token_shapes"),
                    sound_sequence_indexes=cond_packed_static.get("sound_sequence_indexes"),
                    sound_mse_loss_indexes=cond_packed_static.get("sound_mse_loss_indexes"),
                    sound_timesteps=sound_timesteps,
                    sound_noisy_frame_indexes=cond_packed_static.get("sound_noisy_frame_indexes"),
                )
                cond_v_vision, cond_v_sound = self._mask_velocity_predictions(
                    preds_vision,
                    preds_sound,
                    vision_condition_mask=[vision_condition_mask],
                    sound_condition_mask=[sound_condition_mask] if sound_condition_mask is not None else None,
                )

                # --- Unconditional pass ---
                preds_vision, preds_sound = self.transformer(
                    input_ids=uncond_packed_static["input_ids"],
                    text_indexes=uncond_packed_static["text_indexes"],
                    position_ids=uncond_packed_static["position_ids"],
                    und_len=uncond_packed_static["und_len"],
                    sequence_length=uncond_packed_static["sequence_length"],
                    vision_tokens=[vision_tokens],
                    vision_token_shapes=uncond_packed_static["vision_token_shapes"],
                    vision_sequence_indexes=uncond_packed_static["vision_sequence_indexes"],
                    vision_mse_loss_indexes=uncond_packed_static["vision_mse_loss_indexes"],
                    vision_timesteps=vision_timesteps,
                    vision_noisy_frame_indexes=uncond_packed_static["vision_noisy_frame_indexes"],
                    sound_tokens=[sound_tokens] if sound_tokens is not None else None,
                    sound_token_shapes=uncond_packed_static.get("sound_token_shapes"),
                    sound_sequence_indexes=uncond_packed_static.get("sound_sequence_indexes"),
                    sound_mse_loss_indexes=uncond_packed_static.get("sound_mse_loss_indexes"),
                    sound_timesteps=sound_timesteps,
                    sound_noisy_frame_indexes=uncond_packed_static.get("sound_noisy_frame_indexes"),
                )
                uncond_v_vision, uncond_v_sound = self._mask_velocity_predictions(
                    preds_vision,
                    preds_sound,
                    vision_condition_mask=[vision_condition_mask],
                    sound_condition_mask=[sound_condition_mask] if sound_condition_mask is not None else None,
                )

                # --- CFG combine + per-modality scheduler step ---
                # UniPC's multistep_uni_p_bh_update einsum ("k,bkc...->bc...") requires sample
                # to carry a batch dim; per-modality latents have no batch axis, so wrap for the step.
                velocity_vision = uncond_v_vision + guidance_scale * (cond_v_vision - uncond_v_vision)
                latents = self.scheduler.step(
                    velocity_vision.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

                if sound_scheduler is not None and cond_v_sound is not None and uncond_v_sound is not None:
                    velocity_sound = uncond_v_sound + guidance_scale * (cond_v_sound - uncond_v_sound)
                    sound_latents = sound_scheduler.step(
                        velocity_sound.unsqueeze(0), t, sound_latents.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # 7. Postprocess + decode
        sound = self.decode_sound(sound_latents) if sound_latents is not None else None
        if output_type == "latent":
            video = latents
        else:
            in_dtype = latents.dtype
            dtype = self._vae_dtype
            mean = self._vae_latents_mean.to(device=latents.device, dtype=dtype)
            inv_std = self._vae_latents_inv_std.to(device=latents.device, dtype=dtype)
            z_raw = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
            decoded = self.vae.decode(z_raw).sample.to(in_dtype)
            video = self.video_processor.postprocess_video(decoded, output_type=output_type)[0]

        if enable_safety_check and isinstance(self.safety_checker, CosmosSafetyChecker) and output_type != "latent":
            video = self._apply_video_safety_check(video, output_type=output_type, device=device)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, sound)
        return Cosmos3OmniPipelineOutput(video=video, sound=sound)
