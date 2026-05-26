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

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import (
    Cosmos3OmniTransformer,
)
from ...schedulers import UniPCMultistepScheduler
from ...utils import BaseOutput
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline


# ============================================================================
# Sequence layout: data structures + builders for the joint token sequence
# ============================================================================


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for text tokens.

    For text tokens, all three axes (temporal, height, width) share the same
    monotonically increasing position IDs, starting from ``temporal_offset``.
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


@dataclass
class _ModalityData:
    """Per-modality slice of the joint sequence — all fields are finalized tensors
    (or lists of tensors, for per-item entries like ``tokens`` / ``condition_mask``)."""

    tokens: List[torch.Tensor]
    token_shapes: List[Tuple[int, int, int]]
    sequence_indexes: torch.Tensor
    mse_loss_indexes: torch.Tensor
    timesteps: torch.Tensor
    noisy_frame_indexes: List[torch.Tensor]
    condition_mask: List[torch.Tensor]


@dataclass
class Cosmos3PackedSequence:
    """Joint text + vision + (optional) sound sequence consumed by the Cosmos3 transformer.

    ``sequence_length`` is the full packed length; ``und_len`` is the causal-text prefix
    length used to split the understanding stream from the generation stream inside the
    transformer call. ``position_ids`` is the ``[3, sequence_length]`` mRoPE tensor.
    """

    sequence_length: int
    und_len: int
    text_ids: torch.Tensor
    text_indexes: torch.Tensor
    position_ids: torch.Tensor
    vision: _ModalityData
    sound: Optional[_ModalityData] = None


# ============================================================================
# Pipeline output + IO helpers
# ============================================================================


_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."


@dataclass
class Cosmos3OmniPipelineOutput(BaseOutput):
    """Output dataclass for :class:`Cosmos3OmniDiffusersPipeline`.

    Attributes:
        video: List with one entry per generated sample. The contents depend on
            ``output_type`` passed to the pipeline: a list of PIL frames for
            ``"pil"`` (default), an ``np.ndarray``/``torch.Tensor`` of shape
            ``[T, H, W, C]``/``[T, C, H, W]`` for ``"np"``/``"pt"``, or raw
            latents when ``output_type="latent"``.
        sound: List of decoded audio waveforms of shape ``[C, N]``, one per
            sample.  ``None`` when ``enable_sound=False``.
    """

    video: list
    sound: Optional[list] = None


@dataclass
class Cosmos3Condition:
    """Per-sample conditioning data computed by :meth:`prepare_latents` and threaded
    through the denoising loop. Holds the encoded vision (and optional sound)
    targets that get re-mixed with noise at each step, alongside the metadata that
    :func:`pack_input_sequence` needs (FPS values, frame indices kept as condition)."""

    num_vision_items: int
    x0_tokens_vision: List[torch.Tensor]
    fps_vision: torch.Tensor
    condition_frame_indexes_vision: List[int]
    x0_tokens_sound: Optional[List[torch.Tensor]] = None
    fps_sound: Optional[torch.Tensor] = None


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
    _optional_components = ["sound_tokenizer"]
    model_cpu_offload_seq = "transformer->vae->sound_tokenizer"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: Cosmos3OmniTransformer,
        text_tokenizer: AutoTokenizer,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        sound_tokenizer: Optional[Cosmos3AVAEAudioTokenizer] = None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=vae,
            scheduler=scheduler,
            sound_tokenizer=sound_tokenizer,
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

        # Prompt-augmentation templates: appended to the user-supplied prompt and negative prompt
        # inside `tokenize_prompt` so the LLM sees the same metadata the model was trained with.
        self.duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        self.image_resolution_template = "This image is of {height}x{width} resolution."
        self.video_resolution_template = "This video is of {height}x{width} resolution."

        # Recommended quality-control negative prompts are documented in the Cosmos3 docs
        # page (text2video / image2video). When the caller passes None we fall back to "".

    def _encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] → normalized latents [B,z_dim,T//4,H//16,W//16]. Bit-for-bit
        matches Wan2pt2VAEInterface; no autocast (WanVAE was trained with is_amp=False)."""
        in_dtype = x.dtype
        dtype = self._vae_dtype
        mean = self._vae_latents_mean.to(device=x.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=x.device, dtype=dtype)
        raw_mu = retrieve_latents(self.vae.encode(x.to(dtype)), sample_mode="argmax")
        return ((raw_mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)).to(in_dtype)

    def _decode_video(self, z: torch.Tensor) -> torch.Tensor:
        """[B,z_dim,T_lat,H_lat,W_lat] → raw pixels [B,3,T,H,W]."""
        in_dtype = z.dtype
        dtype = self._vae_dtype
        mean = self._vae_latents_mean.to(device=z.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=z.device, dtype=dtype)
        z_raw = z.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        return self.vae.decode(z_raw).sample.to(in_dtype)

    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a sound latent ``[C, T]`` to a waveform ``[audio_ch, N]``.

        Adds/removes the batch dimension expected by the sound tokenizer decoder.
        """
        assert self.sound_tokenizer is not None
        decoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        waveform = self.sound_tokenizer.decode(latent.unsqueeze(0).to(decoder_dtype))  # [1, audio_ch, N]
        return waveform.squeeze(0)  # [audio_ch, N]

    # ------------------------------------------------------------------
    # Joint-sequence packing — each segment is built by a small method
    # that returns its own data; pack_input_sequence stitches them together.
    # ------------------------------------------------------------------

    def _pack_text_tokens(
        self,
        text_tokens: List[int],
        mrope_offset: int | float,
    ) -> Tuple[List[int], torch.Tensor, int | float]:
        """Build the text segment of the joint sequence.

        Returns ``(text_ids, text_mrope_ids, next_mrope_offset)``.
        """
        text_ids = list(text_tokens) + [
            self.llm_special_tokens["eos_token_id"],
            self.llm_special_tokens["start_of_generation"],
        ]
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=len(text_ids),
            temporal_offset=mrope_offset,
            use_float_positions=self.transformer.config.enable_fps_modulation,
        )
        return text_ids, text_mrope_ids, next_mrope_offset

    def _pack_vision_tokens(
        self,
        input_vision_tokens: torch.Tensor,
        condition_frame_indexes: List[int],
        input_timestep: float | torch.Tensor,
        mrope_offset: int | float,
        vision_fps: float | None,
        curr: int,
        device: torch.device | str,
    ) -> Tuple[_ModalityData, torch.Tensor, int | float, int]:
        """Build the vision segment of the joint sequence.

        Returns ``(vision_data, vision_mrope_ids, next_mrope_offset, num_vision_tokens)``.
        """
        config = self.transformer.config
        latent_patch_size = config.latent_patch_size
        _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
        patch_h = math.ceil(latent_h / latent_patch_size)
        patch_w = math.ceil(latent_w / latent_patch_size)
        num_vision_tokens = latent_t * patch_h * patch_w

        condition_set = {idx for idx in condition_frame_indexes if 0 <= idx < latent_t}

        condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=input_vision_tokens.dtype)
        for frame_idx in condition_set:
            condition_mask[frame_idx, 0, 0] = 1.0

        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(latent_t) if idx not in condition_set],
            device=device,
            dtype=torch.long,
        )

        frame_token_stride = patch_h * patch_w
        mse_loss_indexes: list[int] = []
        timesteps: list[float] = []
        for frame_idx in range(latent_t):
            if frame_idx in condition_set:
                continue
            frame_start = curr + frame_idx * frame_token_stride
            mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))
            frame_ts = input_timestep[frame_idx].item() if isinstance(input_timestep, torch.Tensor) else input_timestep
            timesteps.extend([frame_ts] * frame_token_stride)

        effective_fps = vision_fps if config.enable_fps_modulation else None
        vision_mrope_ids, next_mrope_offset = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=self.vae.config.scale_factor_temporal,
        )

        vision_data = _ModalityData(
            tokens=[input_vision_tokens],
            token_shapes=[(latent_t, patch_h, patch_w)],
            sequence_indexes=torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device),
            mse_loss_indexes=torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
            timesteps=torch.tensor(timesteps, device=device),
            noisy_frame_indexes=[noisy_frame_indexes],
            condition_mask=[condition_mask],
        )
        return vision_data, vision_mrope_ids, next_mrope_offset, num_vision_tokens

    def _pack_sound_tokens(
        self,
        input_sound_tokens: torch.Tensor,
        input_timestep: float,
        mrope_offset: int | float,
        sound_fps: float | None,
        curr: int,
        device: torch.device | str,
    ) -> Tuple[_ModalityData, torch.Tensor, int]:
        """Build the sound segment of the joint sequence. All sound frames are noisy.

        Returns ``(sound_data, sound_mrope_ids, sound_len)``.
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
        sound_data = _ModalityData(
            tokens=[input_sound_tokens],
            token_shapes=[(sound_len, 1, 1)],
            sequence_indexes=sequence_indexes,
            mse_loss_indexes=sequence_indexes.clone(),
            timesteps=torch.full((sound_len,), float(input_timestep), device=device),
            noisy_frame_indexes=[torch.arange(sound_len, device=device, dtype=torch.long)],
            condition_mask=[torch.zeros((sound_len, 1), device=device, dtype=input_sound_tokens.dtype)],
        )
        return sound_data, sound_mrope_ids, sound_len

    def pack_input_sequence(
        self,
        text_tokens: List[int],
        input_timestep: torch.Tensor,
        vision_tokens: List[torch.Tensor],
        condition: "Cosmos3Condition",
        sound_tokens: Optional[List[torch.Tensor]] = None,
        device: torch.device | str = "cuda",
    ) -> Cosmos3PackedSequence:
        """Assemble the joint text + vision + (optional) sound sequence consumed by the transformer.

        ``vision_tokens`` / ``sound_tokens`` carry the per-step token tensors (noisy at each
        denoising step, encoded conditioning when extracting condition masks); the rest of
        the per-sample metadata lives on ``condition``.
        """
        if input_timestep.is_cuda:
            raise ValueError("input_timestep must be on CPU, not CUDA")
        if isinstance(text_tokens, torch.Tensor):
            raise ValueError("text_tokens must be a list, not a tensor")

        config = self.transformer.config
        has_sound = sound_tokens is not None

        _ts = input_timestep.flatten()
        input_timestep_val = _ts[0].item() if _ts.numel() == 1 else _ts

        curr = 0
        mrope_offset: int | float = 0
        position_ids_segments: list[torch.Tensor] = []

        # Text segment.
        text_ids, text_mrope_ids, mrope_offset = self._pack_text_tokens(text_tokens.input_ids, mrope_offset)
        und_len = len(text_ids)
        text_indexes = list(range(curr, curr + und_len))
        curr += und_len
        position_ids_segments.append(text_mrope_ids.to(device))

        mrope_offset += config.unified_3d_mrope_temporal_modality_margin
        vision_start_temporal_offset = mrope_offset

        # Vision segment.
        vision_fps: float | None = None
        if config.enable_fps_modulation and len(condition.fps_vision) > 0:
            vision_fps = float(condition.fps_vision[0].item())
        vision_data, vision_mrope_ids, mrope_offset, num_vision_tokens = self._pack_vision_tokens(
            input_vision_tokens=vision_tokens[0],
            condition_frame_indexes=condition.condition_frame_indexes_vision,
            input_timestep=input_timestep_val,
            mrope_offset=mrope_offset,
            vision_fps=vision_fps,
            curr=curr,
            device=device,
        )
        curr += num_vision_tokens
        position_ids_segments.append(vision_mrope_ids.to(device))

        # Sound segment (optional).
        sound_data: Optional[_ModalityData] = None
        if has_sound:
            sound_fps: float | None = None
            if config.enable_fps_modulation and condition.fps_sound is not None and len(condition.fps_sound) > 0:
                sound_fps = float(condition.fps_sound[0].item())
            sound_data, sound_mrope_ids, sound_len = self._pack_sound_tokens(
                input_sound_tokens=sound_tokens[0],
                input_timestep=input_timestep_val,
                mrope_offset=vision_start_temporal_offset,
                sound_fps=sound_fps,
                curr=curr,
                device=device,
            )
            curr += sound_len
            position_ids_segments.append(sound_mrope_ids.to(device))

        return Cosmos3PackedSequence(
            sequence_length=curr,
            und_len=und_len,
            text_ids=torch.tensor(text_ids, dtype=torch.long, device=device),
            text_indexes=torch.tensor(text_indexes, dtype=torch.long, device=device),
            position_ids=torch.cat(position_ids_segments, dim=1),
            vision=vision_data,
            sound=sound_data,
        )

    def prepare_latents(
        self,
        cond_tokens: list[int],
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_sound: bool = False,
    ) -> tuple[torch.Tensor, Cosmos3Condition]:
        """Build conditioning + initial noise for a single sample.

        Returns:
            ``(latents, condition)`` — ``latents`` is the flat initial-noise tensor
            consumed by the denoiser; ``condition`` is the :class:`Cosmos3Condition`
            re-used at every step.
        """
        is_image = num_frames == 1

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
            condition_frame_indexes_vision: List[int] = []
        else:
            vision_tensor = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            condition_frame_indexes_vision = []
            if conditioning_frame_2d is not None:
                # Single conditioning frame at t=0, repeat-pad the rest with that same frame.
                vision_tensor[:, :, 0] = conditioning_frame_2d
                if num_frames > 1:
                    vision_tensor[:, :, 1:] = conditioning_frame_2d.unsqueeze(2).expand(-1, -1, num_frames - 1, -1, -1)
                condition_frame_indexes_vision = [0]

        num_vision_items = 1
        x0_tokens_vision = [self._encode_video(vision_tensor).contiguous().float()]
        fps_vision = torch.tensor([float(fps)], device=device, dtype=dtype)

        x0_tokens_sound: list[torch.Tensor] | None = None
        fps_sound: torch.Tensor | None = None
        if enable_sound:
            sound_dim = self.transformer.config.sound_dim
            sound_latent_fps = float(self.transformer.config.sound_latent_fps)
            n_audio_samples = int(num_frames / fps * self.sound_tokenizer.sample_rate)
            hop_size = self.sound_tokenizer._hop_size
            T_sound = (n_audio_samples + hop_size - 1) // hop_size
            x0_tokens_sound = [torch.zeros(sound_dim, T_sound, device=device, dtype=dtype)]
            fps_sound = torch.tensor([sound_latent_fps], device=device, dtype=dtype)

        condition = Cosmos3Condition(
            num_vision_items=num_vision_items,
            x0_tokens_vision=x0_tokens_vision,
            fps_vision=fps_vision,
            condition_frame_indexes_vision=condition_frame_indexes_vision,
            x0_tokens_sound=x0_tokens_sound,
            fps_sound=fps_sound,
        )

        # Run pack_input_sequence with a dummy timestep to extract the condition_mask used for noise blending.
        mask_timestep = torch.zeros((1,), dtype=torch.float32)
        packed_seq = self.pack_input_sequence(
            text_tokens=cond_tokens,
            input_timestep=mask_timestep,
            vision_tokens=x0_tokens_vision,
            condition=condition,
            sound_tokens=x0_tokens_sound,
            device=device,
        )

        if latents is not None:
            return latents.to(device=device, dtype=dtype), condition

        noise_vision_list: list[torch.Tensor] = []
        for x0_token, cond_mask in zip(x0_tokens_vision, packed_seq.vision.condition_mask, strict=True):
            pure_noise = randn_tensor(tuple(x0_token.shape), generator=generator, device=device, dtype=dtype)
            noise_vision_list.append(
                cond_mask * x0_token.to(device=device, dtype=dtype) + (1.0 - cond_mask) * pure_noise
            )

        initial_noise = torch.cat([t.reshape(-1) for t in noise_vision_list])

        # Append sound noise (all noisy: cond_mask = 0 everywhere)
        if enable_sound and packed_seq.sound is not None:
            for x0_sound, cond_mask_sound in zip(x0_tokens_sound, packed_seq.sound.condition_mask):
                pure_noise_sound = randn_tensor(tuple(x0_sound.shape), generator=generator, device=device, dtype=dtype)
                noise_sound = cond_mask_sound.T * x0_sound + (1.0 - cond_mask_sound.T) * pure_noise_sound
                initial_noise = torch.cat([initial_noise, noise_sound.reshape(-1)])

        return initial_noise, condition

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
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
        if guidance_scale == 1.0:
            raise ValueError("`guidance_scale` must be != 1.0 (classifier-free guidance is required).")
        if enable_sound:
            if self.sound_tokenizer is None:
                raise ValueError("`enable_sound=True` requires a sound-capable checkpoint with a `sound_tokenizer`.")
            if not getattr(self.transformer.config, "sound_gen", False):
                raise ValueError("`enable_sound=True` but the transformer was not trained with `sound_gen=True`.")

    def tokenize_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        use_system_prompt: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Apply prompt-augmentation templates and tokenize cond/uncond prompts via the Qwen2 chat template.

        This pipeline does not run a separate text encoder: the joint Cosmos3 transformer
        consumes raw Qwen2 token IDs alongside vision (and optionally sound) tokens.

        When ``negative_prompt`` is ``None``, an empty string is used; the
        Cosmos3 docs page documents recommended quality-control negative
        prompts to pass explicitly for text2video / image2video. The duration
        and resolution templates are then appended to both prompts.

        Returns:
            ``(cond_text_tokens, uncond_text_tokens)`` — token-ID lists for this sample.
        """
        is_image = num_frames == 1

        if negative_prompt is None:
            negative_prompt = ""

        resolution_template = self.image_resolution_template if is_image else self.video_resolution_template

        def _apply_templates(text: str) -> str:
            if not is_image:
                text = text.rstrip(".") + ". " + self.duration_template.format(duration=num_frames / fps, fps=fps)
            text = text.rstrip(".") + ". " + resolution_template.format(height=height, width=width)
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

        cond_tokens = _tokenize(_apply_templates(prompt))
        uncond_tokens = _tokenize(_apply_templates(negative_prompt))
        return cond_tokens, uncond_tokens

    def decode_latents(self, vision_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode latents to pixel tensors of shape [C, T, H, W] in [-1, 1]."""
        frames = []
        for vision_latent in vision_list:
            vision = self._decode_video(vision_latent.cuda())  # [1, C, T, H, W]
            frames.append(vision.squeeze(0))
        return frames

    def _postprocess_latents(
        self,
        latents: torch.Tensor,
        condition: "Cosmos3Condition",
        enable_sound: bool,
        output_type: str,
    ) -> tuple[list, Optional[list[torch.Tensor]]]:
        """Extract vision/sound slices from the flat denoised latent, then decode.

        Returns ``(video, sound)``. ``video`` is a list with one entry per generated
        sample: raw latents when ``output_type == "latent"``, otherwise the output of
        :meth:`VideoProcessor.postprocess_video` (e.g. a list of PIL frames for
        ``output_type="pil"``). ``sound`` is ``None`` unless ``enable_sound`` was set.
        """
        result_vision: list[torch.Tensor] = []
        offset = 0
        for j in range(condition.num_vision_items):
            vision_shape = condition.x0_tokens_vision[j].shape
            vision_dim = math.prod(vision_shape)
            if j == condition.num_vision_items - 1:
                result_vision.append(latents[offset : offset + vision_dim].reshape(vision_shape))
            offset += vision_dim

        result_sound: Optional[list] = None
        if enable_sound and condition.x0_tokens_sound is not None:
            sound_shape = condition.x0_tokens_sound[0].shape  # [sound_dim, T_sound]
            sound_dim_flat = math.prod(sound_shape)
            sound_latent = latents[offset : offset + sound_dim_flat].reshape(sound_shape)
            result_sound = [self.decode_sound(sound_latent)]

        if output_type == "latent":
            return result_vision, result_sound

        decoded = self.decode_latents(result_vision)  # list of [C, T, H, W] in [-1, 1]
        video = [
            self.video_processor.postprocess_video(v.unsqueeze(0), output_type=output_type)[0]
            for v in decoded
        ]
        return video, result_sound

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        enable_sound: bool = False,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        use_system_prompt: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict[str, Any]], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ) -> Cosmos3OmniPipelineOutput:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(prompt, negative_prompt, image, height, width, num_frames, guidance_scale, enable_sound)
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

        device = self._execution_device
        dtype = self.transformer.dtype

        # 2. Tokenize prompt (applies metadata templates and selects mode-specific default negative prompt)
        cond_tokens, uncond_tokens = self.tokenize_prompt(
            prompt,
            negative_prompt,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            use_system_prompt=use_system_prompt,
        )

        # 4. Prepare latents (initial noise + conditioning metadata)
        latents, condition = self.prepare_latents(
            cond_tokens=cond_tokens,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            latents=latents,
            generator=generator,
            device=device,
            dtype=dtype,
            enable_sound=enable_sound,
        )

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        has_sound = condition.x0_tokens_sound is not None

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.reshape(1, 1)

                # Split flat latents → per-modality noisy tensors for this step
                noise_x_vision: list[torch.Tensor] = []
                offset = 0
                for j in range(condition.num_vision_items):
                    vision_shape = condition.x0_tokens_vision[j].shape
                    vision_dim = math.prod(vision_shape)
                    noise_x_vision.append(latents[offset : offset + vision_dim].reshape(vision_shape))
                    offset += vision_dim
                noise_x_sound: Optional[list] = None
                if has_sound:
                    sound_shape = condition.x0_tokens_sound[0].shape
                    sound_dim_flat = math.prod(sound_shape)
                    noise_x_sound = [latents[offset : offset + sound_dim_flat].reshape(sound_shape)]

                # The transformer projections (proj_in / audio_proj_in) are bf16; cast the per-step
                # noisy tokens before packing so the modality tokens enter the model in the right dtype.
                vision_tokens = [x.to(device=device, dtype=dtype) for x in noise_x_vision]
                sound_tokens = (
                    [x.to(device=device, dtype=dtype) for x in noise_x_sound] if noise_x_sound is not None else None
                )

                # --- Conditional pass ---
                packed_seq = self.pack_input_sequence(
                    text_tokens=cond_tokens,
                    input_timestep=timestep.cpu(),
                    vision_tokens=vision_tokens,
                    condition=condition,
                    sound_tokens=sound_tokens,
                    device=device,
                )
                preds_vision, preds_sound = self.transformer(packed_seq)

                velocity_vision = [
                    pred * (1.0 - m).to(dtype=pred.dtype, device=pred.device)
                    if (1.0 - m).sum() > 0
                    else torch.zeros_like(pred)
                    for pred, m in zip(preds_vision, packed_seq.vision.condition_mask)
                ]
                parts = [v.reshape(-1) for v in velocity_vision]
                if preds_sound is not None and packed_seq.sound is not None:
                    for pred_s, cond_mask_s in zip(preds_sound, packed_seq.sound.condition_mask):
                        noisy_mask_s = (1.0 - cond_mask_s).T.to(dtype=pred_s.dtype, device=pred_s.device)
                        v_sound = pred_s * noisy_mask_s if noisy_mask_s.sum() > 0 else torch.zeros_like(pred_s)
                        parts.append(v_sound.reshape(-1))
                cond_v = torch.cat(parts)

                # --- Unconditional pass ---
                packed_seq = self.pack_input_sequence(
                    text_tokens=uncond_tokens,
                    input_timestep=timestep.cpu(),
                    vision_tokens=vision_tokens,
                    condition=condition,
                    sound_tokens=sound_tokens,
                    device=device,
                )
                preds_vision, preds_sound = self.transformer(packed_seq)

                velocity_vision = [
                    pred * (1.0 - m).to(dtype=pred.dtype, device=pred.device)
                    if (1.0 - m).sum() > 0
                    else torch.zeros_like(pred)
                    for pred, m in zip(preds_vision, packed_seq.vision.condition_mask)
                ]
                parts = [v.reshape(-1) for v in velocity_vision]
                if preds_sound is not None and packed_seq.sound is not None:
                    for pred_s, cond_mask_s in zip(preds_sound, packed_seq.sound.condition_mask):
                        noisy_mask_s = (1.0 - cond_mask_s).T.to(dtype=pred_s.dtype, device=pred_s.device)
                        v_sound = pred_s * noisy_mask_s if noisy_mask_s.sum() > 0 else torch.zeros_like(pred_s)
                        parts.append(v_sound.reshape(-1))
                uncond_v = torch.cat(parts)

                # --- CFG combine + scheduler step ---
                # UniPC's multistep_uni_p_bh_update einsum ("k,bkc...->bc...") requires sample
                # to carry a batch dim; our latents are 1-D flat, so wrap for the step.
                velocity_pred = uncond_v + guidance_scale * (cond_v - uncond_v)
                latents = self.scheduler.step(velocity_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[
                    0
                ].squeeze(0)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # 7. Postprocess + decode
        video, sound = self._postprocess_latents(
            latents,
            condition=condition,
            enable_sound=enable_sound,
            output_type=output_type,
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, sound)
        return Cosmos3OmniPipelineOutput(video=video, sound=sound)
