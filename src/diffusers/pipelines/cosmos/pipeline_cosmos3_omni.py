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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import (
    Cosmos3OmniTransformer,
)
from ...schedulers import UniPCMultistepScheduler
from ...utils import BaseOutput, export_to_video
from ...utils.torch_utils import randn_tensor
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
class ModalityData:
    """Unified container for a single generation modality's data.

    Acts as a builder during packing (lists), then holds finalized tensors after ``finalize()``.
    """

    sequence_indexes: list[int] | torch.Tensor = field(default_factory=list)
    timesteps: list[float] | torch.Tensor = field(default_factory=list)
    mse_loss_indexes: list[int] | torch.Tensor = field(default_factory=list)
    token_shapes: list = field(default_factory=list)

    tokens: list[torch.Tensor] = field(default_factory=list)
    condition_mask: list[torch.Tensor] = field(default_factory=list)
    noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)


@dataclass
class PackedSequence:
    """Unified sequence container - works as builder during packing and final output."""

    # Sequence structure
    sample_lens: list[int] = field(default_factory=list)
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    sequence_length: int = 0

    # Build-time tracking
    curr: int = 0

    # Text modality (list during build, tensor after finalize)
    text_ids: list[int] | torch.Tensor = field(default_factory=list)
    text_indexes: list[int] | torch.Tensor = field(default_factory=list)
    position_ids: list[int] | torch.Tensor = field(default_factory=list)

    # Build-time mRoPE tracking
    _use_mrope: bool = False
    _mrope_temporal_offset: int | float = 0
    _mrope_reset_spatial: bool = True

    # Generation modalities
    vision: ModalityData | None = None
    sound: ModalityData | None = None

    def finalize(self) -> "PackedSequence":
        """Convert all lists to tensors and compute derived values."""
        sequence_length = sum(self.sample_lens)
        sample_lens = self.sample_lens.copy()
        split_lens = self.split_lens.copy()
        attn_modes = self.attn_modes.copy()

        vision: ModalityData | None = None
        if self.vision is not None and len(self.vision.sequence_indexes) > 0:
            vision = ModalityData(
                sequence_indexes=torch.tensor(self.vision.sequence_indexes, dtype=torch.long),
                timesteps=torch.tensor(self.vision.timesteps),
                mse_loss_indexes=torch.tensor(self.vision.mse_loss_indexes, dtype=torch.long),
                token_shapes=list(self.vision.token_shapes),
                tokens=self.vision.tokens,
                condition_mask=list(self.vision.condition_mask),
                noisy_frame_indexes=list(self.vision.noisy_frame_indexes),
            )

        sound: ModalityData | None = None
        if self.sound is not None and len(self.sound.sequence_indexes) > 0:
            sound = ModalityData(
                sequence_indexes=torch.tensor(self.sound.sequence_indexes, dtype=torch.long),
                timesteps=torch.tensor(self.sound.timesteps),
                mse_loss_indexes=torch.tensor(self.sound.mse_loss_indexes, dtype=torch.long),
                token_shapes=list(self.sound.token_shapes),
                tokens=self.sound.tokens,
                condition_mask=list(self.sound.condition_mask),
                noisy_frame_indexes=list(self.sound.noisy_frame_indexes),
            )

        if self._use_mrope and len(self.position_ids) > 0 and isinstance(self.position_ids[0], torch.Tensor):
            mrope_tensors: list[torch.Tensor] = self.position_ids  # type: ignore[assignment]
            position_ids = torch.cat(mrope_tensors, dim=1)  # [3,actual_seq_len]
        else:
            position_ids = torch.tensor(self.position_ids)  # [seq_len]

        return PackedSequence(
            sequence_length=sequence_length,
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            text_ids=torch.tensor(self.text_ids, dtype=torch.long),
            text_indexes=torch.tensor(self.text_indexes, dtype=torch.long),
            position_ids=position_ids,
            vision=vision,
            sound=sound,
        )

    def to_cuda(self) -> None:
        """Move every tensor field (and modality sub-objects) to CUDA in-place."""
        for attr in ("text_ids", "text_indexes", "position_ids"):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.cuda())
        for modality in (self.vision, self.sound):
            if modality is not None:
                _modality_to_cuda(modality)


def _modality_to_cuda(modality: ModalityData) -> None:
    for attr in ("sequence_indexes", "timesteps", "mse_loss_indexes"):
        val = getattr(modality, attr)
        if isinstance(val, torch.Tensor):
            setattr(modality, attr, val.cuda())
    modality.tokens = [t.cuda() for t in modality.tokens]
    modality.condition_mask = [m.cuda() for m in modality.condition_mask]
    modality.noisy_frame_indexes = [i.cuda() for i in modality.noisy_frame_indexes]


@dataclass
class SequencePlan:
    """Plan describing which modalities are present in a sample."""

    has_text: bool
    has_vision: bool = False
    condition_frame_indexes_vision: list[int] = field(default_factory=list)
    has_sound: bool = False

    def as_dict(self) -> dict:
        return {
            "has_text": self.has_text,
            "has_vision": self.has_vision,
            "has_sound": self.has_sound,
            "condition_frame_indexes_vision": self.condition_frame_indexes_vision,
        }


def _pack_text_tokens(
    packed_seq: PackedSequence,
    text_ids: List[int],
    special_tokens: Dict[str, int],
    curr_rope_id: int,
    has_generation: bool,
    use_float_positions: bool = False,
) -> Tuple[int, int, int]:
    """Pack text tokens into the sequence."""
    assert isinstance(packed_seq.text_ids, list), "PackedSequence must be in build mode"
    assert isinstance(packed_seq.text_indexes, list)
    assert isinstance(packed_seq.position_ids, list)

    curr = packed_seq.curr

    if "bos_token_id" in special_tokens:
        shifted_text_ids = [special_tokens["bos_token_id"]] + text_ids
    else:
        shifted_text_ids = text_ids

    split_len = 0

    packed_seq.text_ids.extend(shifted_text_ids)
    packed_seq.text_indexes.extend(range(curr, curr + len(shifted_text_ids)))

    curr += len(shifted_text_ids)
    split_len += len(shifted_text_ids)

    packed_seq.text_ids.append(special_tokens["eos_token_id"])
    packed_seq.text_indexes.append(curr)
    curr += 1
    split_len += 1

    if has_generation:
        packed_seq.text_ids.append(special_tokens["start_of_generation"])
        packed_seq.text_indexes.append(curr)
        curr += 1
        split_len += 1

    if packed_seq._use_mrope:
        text_mrope_ids, packed_seq._mrope_temporal_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=split_len,
            temporal_offset=packed_seq._mrope_temporal_offset,
            use_float_positions=use_float_positions,
        )
        packed_seq.position_ids.append(text_mrope_ids)
    else:
        packed_seq.position_ids.extend(range(curr_rope_id, curr_rope_id + split_len))
    packed_seq.attn_modes.append("causal")
    packed_seq.split_lens.append(split_len)

    packed_seq.curr = curr
    return curr_rope_id + split_len, split_len, split_len


def _pack_vision_tokens(
    packed_seq: PackedSequence,
    input_vision_tokens: torch.Tensor,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    curr_rope_id: int,
    latent_patch_size: int = 1,
    vision_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
) -> int:
    """Pack vision tokens into the sequence."""
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    vision_split_len = 0

    if packed_seq.vision is None:
        packed_seq.vision = ModalityData()

    assert isinstance(packed_seq.vision.sequence_indexes, list)
    assert isinstance(packed_seq.vision.mse_loss_indexes, list)
    assert isinstance(packed_seq.vision.timesteps, list)
    assert isinstance(packed_seq.vision.tokens, list)

    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    if latent_patch_size < 1:
        raise ValueError(f"latent_patch_size must be >= 1, got {latent_patch_size}")
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    packed_seq.vision.token_shapes.append((latent_t, patch_h, patch_w))
    packed_seq.vision.tokens.append(input_vision_tokens)

    num_vision_tokens = latent_t * patch_h * patch_w
    packed_seq.vision.sequence_indexes.extend(range(curr, curr + num_vision_tokens))

    condition_set = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    assert isinstance(packed_seq.vision.condition_mask, list)

    vision_condition_mask = torch.zeros(
        (latent_t, 1, 1), device=input_vision_tokens.device, dtype=input_vision_tokens.dtype
    )
    for frame_idx in condition_set:
        vision_condition_mask[frame_idx, 0, 0] = 1.0
    packed_seq.vision.condition_mask.append(vision_condition_mask)

    vision_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in condition_set],
        device=input_vision_tokens.device,
        dtype=torch.long,
    )
    assert isinstance(packed_seq.vision.noisy_frame_indexes, list)
    packed_seq.vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

    frame_token_stride = patch_h * patch_w
    for frame_idx in range(latent_t):
        if frame_idx in condition_set:
            continue
        frame_start = curr + frame_idx * frame_token_stride
        frame_end = frame_start + frame_token_stride
        packed_seq.vision.mse_loss_indexes.extend(range(frame_start, frame_end))
        if isinstance(input_timestep, torch.Tensor):
            frame_ts = input_timestep[frame_idx].item()
        else:
            frame_ts = input_timestep
        packed_seq.vision.timesteps.extend([frame_ts] * frame_token_stride)

    curr += num_vision_tokens
    vision_split_len += num_vision_tokens

    if packed_seq._use_mrope:
        effective_fps = vision_fps if enable_fps_modulation else None
        vision_mrope_ids, packed_seq._mrope_temporal_offset = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=packed_seq._mrope_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression_factor,
        )
        packed_seq.position_ids.append(vision_mrope_ids)
    else:
        packed_seq.position_ids.extend([curr_rope_id] * vision_split_len)

    packed_seq.curr = curr
    return vision_split_len


def _pack_sound_tokens(
    packed_seq: PackedSequence,
    input_sound_tokens: torch.Tensor,
    input_timestep: float,
    curr_rope_id: int,
    sound_temporal_offset: int | float = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    sound_fps: float | None = None,
) -> int:
    """Pack sound/audio tokens into the sequence. All sound frames are noisy (no conditioning)."""
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    _, sound_split_len = input_sound_tokens.shape

    if packed_seq.sound is None:
        packed_seq.sound = ModalityData()

    assert isinstance(packed_seq.sound.sequence_indexes, list)
    assert isinstance(packed_seq.sound.mse_loss_indexes, list)
    assert isinstance(packed_seq.sound.timesteps, list)
    assert isinstance(packed_seq.sound.tokens, list)
    assert isinstance(packed_seq.sound.condition_mask, list)
    assert isinstance(packed_seq.sound.noisy_frame_indexes, list)

    packed_seq.sound.token_shapes.append((sound_split_len, 1, 1))
    packed_seq.sound.sequence_indexes.extend(range(curr, curr + sound_split_len))
    packed_seq.sound.tokens.append(input_sound_tokens)

    packed_seq.sound.condition_mask.append(
        torch.zeros((sound_split_len, 1), device=input_sound_tokens.device, dtype=input_sound_tokens.dtype)
    )
    packed_seq.sound.noisy_frame_indexes.append(
        torch.arange(sound_split_len, device=input_sound_tokens.device, dtype=torch.long)
    )

    packed_seq.sound.mse_loss_indexes.extend(range(curr, curr + sound_split_len))
    packed_seq.sound.timesteps.extend([input_timestep] * sound_split_len)

    if packed_seq._use_mrope:
        effective_fps = sound_fps if enable_fps_modulation else None
        sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=sound_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,
            start_frame_offset=0,
        )
        packed_seq.position_ids.append(sound_mrope_ids)
    else:
        packed_seq.position_ids.extend([curr_rope_id] * sound_split_len)

    packed_seq.curr = curr + sound_split_len
    return sound_split_len


def pack_input_sequence(
    sequence_plan: SequencePlan,
    text_tokens: list[int],
    input_timestep: torch.Tensor,
    special_tokens: dict[str, int],
    num_vision_items: int = 1,
    x0_tokens_vision: Optional[List[torch.Tensor]] = None,
    fps_vision: Optional[torch.Tensor] = None,
    x0_tokens_sound: Optional[List[torch.Tensor]] = None,
    fps_sound: Optional[torch.Tensor] = None,
    latent_patch_size: int = 1,
    skip_text_tokens: bool = False,
    include_end_of_generation_token: bool = False,
    position_embedding_type: str = "3d_rope",
    unified_3d_mrope_reset_spatial_ids: bool = True,
    unified_3d_mrope_temporal_modality_margin: int = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    initial_mrope_temporal_offset: int | float = 0,
) -> PackedSequence:
    """Pack a single sample's text + vision + sound tokens into the joint sequence layout."""
    assert special_tokens is not None, "Special tokens must be provided"
    assert isinstance(input_timestep, torch.Tensor), "input_timestep must be a tensor"
    if input_timestep.is_cuda:
        raise ValueError("input_timestep must be on CPU, not CUDA")
    if isinstance(text_tokens, torch.Tensor):
        raise ValueError("text_tokens must be a list, not a tensor")
    if not skip_text_tokens:
        assert sequence_plan.has_text, "sequence_plan must have has_text=True when skip_text_tokens=False"

    packed_seq = PackedSequence()
    packed_seq._use_mrope = position_embedding_type == "unified_3d_mrope"
    packed_seq._mrope_reset_spatial = unified_3d_mrope_reset_spatial_ids
    packed_seq._mrope_temporal_offset = initial_mrope_temporal_offset

    _ts = input_timestep.flatten()
    input_timestep_val = _ts[0].item() if _ts.numel() == 1 else _ts

    curr_rope_id = 0
    sample_len = 0
    idx_vision = 0  # walks across multi-vision items within this single sample

    if sequence_plan.has_text and not skip_text_tokens:
        has_generation_for_sample = sequence_plan.has_vision or sequence_plan.has_sound
        curr_rope_id, _, text_sample_len = _pack_text_tokens(
            packed_seq,
            text_tokens,
            special_tokens,
            curr_rope_id,
            has_generation=has_generation_for_sample,
            use_float_positions=enable_fps_modulation,
        )
        sample_len += text_sample_len
        packed_seq._mrope_temporal_offset += unified_3d_mrope_temporal_modality_margin

    vision_start_temporal_offset = packed_seq._mrope_temporal_offset

    if sequence_plan.has_vision:
        vision_split_len = 0
        for item_idx in range(num_vision_items):
            input_vision_tokens = x0_tokens_vision[idx_vision]

            vision_fps: float | None = None
            if (
                enable_fps_modulation
                and fps_vision is not None
                and idx_vision < len(fps_vision)
            ):
                vision_fps = float(fps_vision[idx_vision].item())

            idx_vision += 1

            if num_vision_items > 1 and item_idx < num_vision_items - 1:
                latent_t = input_vision_tokens.shape[2]
                item_condition_frames = list(range(latent_t))
            else:
                item_condition_frames = sequence_plan.condition_frame_indexes_vision

            item_split_len = _pack_vision_tokens(
                packed_seq=packed_seq,
                input_vision_tokens=input_vision_tokens,
                condition_frame_indexes_vision=item_condition_frames,
                input_timestep=input_timestep_val,
                curr_rope_id=curr_rope_id,
                latent_patch_size=latent_patch_size,
                vision_fps=vision_fps,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
                temporal_compression_factor=temporal_compression_factor,
            )
            vision_split_len += item_split_len
        sample_len += vision_split_len
    else:
        vision_split_len = 0

    if sequence_plan.has_sound:
        input_sound_tokens = x0_tokens_sound[0]

        sound_fps: float | None = None
        if enable_fps_modulation and fps_sound is not None and len(fps_sound) > 0:
            sound_fps = float(fps_sound[0].item())

        sound_split_len = _pack_sound_tokens(
            packed_seq=packed_seq,
            input_sound_tokens=input_sound_tokens,
            input_timestep=input_timestep_val,
            curr_rope_id=curr_rope_id,
            sound_temporal_offset=vision_start_temporal_offset,
            enable_fps_modulation=enable_fps_modulation,
            base_fps=base_fps,
            sound_fps=sound_fps,
        )
        sample_len += sound_split_len
    else:
        sound_split_len = 0

    eov_len = 0
    has_any_generation = sequence_plan.has_vision or sequence_plan.has_sound
    if include_end_of_generation_token and has_any_generation:
        assert isinstance(packed_seq.text_ids, list)
        assert isinstance(packed_seq.text_indexes, list)
        assert isinstance(packed_seq.position_ids, list)

        packed_seq.text_ids.append(special_tokens["end_of_generation"])
        packed_seq.text_indexes.append(packed_seq.curr)

        if packed_seq._use_mrope:
            eov_dtype = torch.float32 if enable_fps_modulation else torch.long
            eov_mrope_ids = torch.full((3, 1), packed_seq._mrope_temporal_offset, dtype=eov_dtype)
            packed_seq.position_ids.append(eov_mrope_ids)  # type: ignore[arg-type]
            packed_seq._mrope_temporal_offset += 1
        else:
            packed_seq.position_ids.append(curr_rope_id)  # type: ignore[arg-type]

        packed_seq.curr += 1
        eov_len = 1
        sample_len += 1

    combined_split_len = vision_split_len + sound_split_len + eov_len
    packed_seq.attn_modes.append("full")
    packed_seq.split_lens.append(combined_split_len)
    packed_seq.sample_lens.append(sample_len)

    return packed_seq.finalize()


# ============================================================================
# Pipeline output + IO helpers
# ============================================================================


_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."


@dataclass
class Cosmos3OmniPipelineOutput(BaseOutput):
    """Output dataclass for :class:`Cosmos3OmniDiffusersPipeline`.

    Attributes:
        video: List of decoded video tensors, one per generated sample,
            each of shape ``[C, T, H, W]`` in ``[0, 1]`` (or raw latents when
            ``output_type="latent"``).
        sound: List of decoded audio waveforms of shape ``[C, N]``, one per
            sample.  ``None`` when ``enable_sound=False``.
    """

    video: list
    sound: Optional[list] = None


def save_img_or_video(sample, save_fp_wo_ext, fps=24, quality=10):
    """Save a 4D ``[C, T, H, W]`` sample as a JPEG (T=1) or MP4 (T>1)."""
    from PIL import Image as PILImage

    assert sample.ndim == 4, "Only support 4D tensor [C, T, H, W]"

    if torch.is_floating_point(sample):
        sample = sample.clamp(0, 1)
    else:
        assert sample.dtype == torch.uint8, "Only support uint8 tensor"
        sample = sample.float().div(255)

    np_arr = sample.cpu().float().numpy()  # [C, T, H, W] in [0, 1]
    if np_arr.shape[1] == 1:
        img = (np_arr.squeeze(1).transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, C] uint8 for PIL
        PILImage.fromarray(img, mode="RGB").save(f"{save_fp_wo_ext}.jpg", format="JPEG", quality=85)
    else:
        # export_to_video scales float [0, 1] ndarrays to uint8 internally — don't pre-scale.
        # macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
        frames = list(np_arr.transpose(1, 2, 3, 0))  # list of [H, W, C] float frames
        export_to_video(frames, f"{save_fp_wo_ext}.mp4", fps=fps, quality=quality, macro_block_size=1)


def save_wav(waveform: torch.Tensor, path, sample_rate: int) -> None:
    """Save a decoded waveform ``[C, N]`` or ``[N]`` as a WAV file.

    Args:
        waveform: Audio tensor of shape ``[C, N]`` (multi-channel) or ``[N]`` (mono).
        path: Destination file path (``str`` or :class:`~pathlib.Path`).  The ``.wav``
            extension is expected but not enforced.
        sample_rate: Sample rate in Hz.
    """
    import soundfile as sf  # type: ignore[import-not-found]

    audio_np = waveform.clamp(-1.0, 1.0).to(dtype=torch.float32).cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # soundfile expects [N, C]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio_np, sample_rate)


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

        self.llm_special_tokens = {
            "start_of_generation": text_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "end_of_generation": text_tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            "eos_token_id": text_tokenizer.eos_token_id,
        }

        # Prompt-augmentation templates: appended to the user-supplied prompt and negative prompt
        # inside `encode_prompt` so the LLM sees the same metadata the model was trained with.
        self.duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        self.image_resolution_template = "This image is of {height}x{width} resolution."
        self.video_resolution_template = "This video is of {height}x{width} resolution."

        # Default negative prompts used when the caller doesn't supply one.
        self.text2image_negative_prompt = ""
        self.text2video_negative_prompt = (
            "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
            "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
            "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
            "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
            "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
            "Overall, the video is of poor quality."
        )
        self.image2video_negative_prompt = (
            "The video captures a series of frames showing macroblocking artifacts, chromatic aberration, "
            "high-frequency noise, and rolling shutter distortion. It includes static with no motion, motion blur, "
            "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
            "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
            "movements, low frame rate, bit-depth compression artifacts, color banding, unnatural transitions, "
            "outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual "
            "noise, and flickering. Avoid moiré patterns, edge halos, and temporal aliasing. Furthermore, the content "
            "defies common sense, generating illogical scenarios, nonsensical entities, absurd character behaviors, "
            "and conceptual paradoxes that violate basic human reasoning and everyday reality. The video looks like a "
            "surreal or glitchy hallucination. Overall, the video is of poor quality."
        )

    def _encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] → normalized latents [B,z_dim,T//4,H//16,W//16]. Bit-for-bit
        matches Wan2pt2VAEInterface; no autocast (WanVAE was trained with is_amp=False)."""
        in_dtype = x.dtype
        dtype = self._vae_dtype
        mean = self._vae_latents_mean.to(device=x.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=x.device, dtype=dtype)
        raw_mu = self.vae.encode(x.to(dtype)).latent_dist.mode()
        return ((raw_mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)).to(in_dtype)

    def _decode_video(self, z: torch.Tensor) -> torch.Tensor:
        """[B,z_dim,T_lat,H_lat,W_lat] → raw pixels [B,3,T,H,W]."""
        in_dtype = z.dtype
        dtype = self._vae_dtype
        mean = self._vae_latents_mean.to(device=z.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=z.device, dtype=dtype)
        z_raw = z.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        return self.vae.decode(z_raw).sample.to(in_dtype)

    def tokenize_caption(
        self,
        caption: str,
        is_video: bool = False,
        use_system_prompt: bool = False,
    ) -> list[int]:
        """Tokenize a text caption into token IDs using the Qwen2 chat template.
        Returns:
            List of token IDs representing the full chat-formatted caption.
        """
        conversations = []
        # Optionally prepend a system prompt that tells the model whether it is generating
        # an image or a video. This changes the conditioning context for the LLM.
        if use_system_prompt:
            _system_prompt = _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE
            conversations.append({"role": "system", "content": _system_prompt})
        conversations.append({"role": "user", "content": caption})

        tokenizer_output = self.text_tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
        )
        return tokenizer_output

    def apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: List[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        start_noisy_index = 0
        flattened_noisy_frame_indexes = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            assert noisy_indexes_i.numel() <= token_shape_i[0]
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            noisy_indexes_i = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
            noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())
            start_noisy_index += math.prod(token_shape_i)
        flattened_noisy_frame_indexes = torch.cat(flattened_noisy_frame_indexes, dim=0)
        assert packed_tokens.dim() == 2
        assert packed_timestep_embeds.dim() == 2
        assert packed_timestep_embeds.shape[1] == packed_tokens.shape[1]
        assert packed_timestep_embeds.shape[0] <= packed_tokens.shape[0]
        assert flattened_noisy_frame_indexes.dim() == 1
        assert flattened_noisy_frame_indexes.shape[0] == packed_timestep_embeds.shape[0]
        flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(-1).expand(
            -1,
            packed_tokens.shape[1],
        )
        return packed_tokens.scatter_add(
            dim=0,
            index=flattened_noisy_frame_indexes,
            src=packed_timestep_embeds,
        )

    def patchify_and_pack_latents(
        self,
        latent_patch_size: int,
        latent_channel: int,
        tokens_vision: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = latent_patch_size
        packed_latent = []
        original_latent_shapes = []
        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)  # [C,T,H,W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def unpatchify_and_unpack_latents(
        self,
        latent_patch_size: int,
        latent_channel: int,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        p = latent_patch_size
        unpatchified_latents = []
        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            if original_latent_shapes is not None:
                _, h_orig, w_orig = original_latent_shapes[i]
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                h_orig, w_orig = h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c
            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                output_tensor[:, noisy_frame_indexes] = latent
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def decode_vision(
        self,
        patch_latent_dim: int,
        latent_patch_size: int,
        latent_channel: int,
        packed_seq,
        last_hidden_state: torch.Tensor,
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """Decode vision predictions from last_hidden_state. Returns preds_vision list."""
        vision = packed_seq.vision
        has_noisy_vision = (
            vision is not None
            and vision.tokens is not None
            and isinstance(vision.mse_loss_indexes, torch.Tensor)
            and vision.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_vision:
            preds_vision = torch.zeros(
                [1, patch_latent_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )
            preds_vision = self.transformer.vae2llm(preds_vision)
            preds_vision = self.transformer.llm2vae(preds_vision)
            if vision is not None and vision.tokens is not None:
                preds_vision_list = [torch.zeros_like(tok) for tok in vision.tokens]
                preds_vision_list[0] = preds_vision_list[0] + 0.0 * preds_vision.sum()
            else:
                preds_vision_list = [preds_vision]
        else:
            assert vision is not None
            assert isinstance(vision.mse_loss_indexes, torch.Tensor)
            assert vision.noisy_frame_indexes is not None
            preds_vision = self.transformer.llm2vae(last_hidden_state[vision.mse_loss_indexes])
            preds_vision_list = self.unpatchify_and_unpack_latents(
                latent_patch_size,
                latent_channel,
                preds_vision,
                token_shapes_vision=vision.token_shapes,
                noisy_frame_indexes_vision=vision.noisy_frame_indexes,
                original_latent_shapes=original_latent_shapes,
            )
        return preds_vision_list

    def _pack_sound_latents(
        self,
        tokens_sound: list,
        token_shapes_sound: list,
    ) -> torch.Tensor:
        """Pack per-sample sound latents into a single 2-D tensor.

        Args:
            tokens_sound: List of ``[C, T]`` tensors, one per sample.
            token_shapes_sound: List of ``(T, 1, 1)`` tuples (from packed_seq.sound).

        Returns:
            ``[total_T, C]`` packed tensor.
        """
        packed = []
        for sound, shape in zip(tokens_sound, token_shapes_sound):
            T = shape[0]
            packed.append(sound[:, :T].permute(1, 0))  # [C, T] → [T, C]
        return torch.cat(packed, dim=0)  # [total_T, C]

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: list,
        noisy_frame_indexes_sound: list,
    ) -> list:
        """Unpack packed sound predictions back to per-sample ``[C, T]`` tensors.

        Args:
            packed_preds: ``[total_noisy_T, C]`` predictions at noisy positions.
            token_shapes_sound: List of ``(T, 1, 1)`` tuples per sample.
            noisy_frame_indexes_sound: List of ``[T_noisy]`` index tensors per sample.

        Returns:
            List of ``[C, T]`` tensors (zeros at conditioned positions).
        """
        sound_dim = self.transformer.config.sound_dim
        unpacked = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            output = torch.zeros(
                (sound_dim, T),
                device=packed_preds.device,
                dtype=packed_preds.dtype,
            )
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[:, noisy_idxs] = packed_preds[start_idx : start_idx + t_n].T
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    def encode_sound_tokens(
        self,
        timestep_scale: float,
        packed_seq,
        hidden_states: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Project sound tokens into ``hidden_states`` in-place.

        Projects sound latents into ``hidden_states`` via ``sound2llm``, modality embed, and timestep embeddings.

        Steps:
        1. Pack latents: list of ``[C, T]`` → ``[total_T, C]``
        2. Project: ``sound2llm`` + ``sound_modality_embed``
        3. Add timestep embeddings to noisy frames
        4. Scatter into ``hidden_states`` at ``sound.sequence_indexes``
        """
        if packed_seq.sound is None or packed_seq.sound.tokens is None:
            return

        sound = packed_seq.sound
        assert sound.token_shapes is not None
        assert isinstance(sound.sequence_indexes, torch.Tensor)
        assert isinstance(sound.timesteps, torch.Tensor)
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)

        packed_tokens_sound = self._pack_sound_latents(sound.tokens, sound.token_shapes)
        packed_tokens_sound = packed_tokens_sound.to(target_dtype)

        packed_tokens_sound = self.transformer.sound2llm(packed_tokens_sound) + self.transformer.sound_modality_embed

        if sound.mse_loss_indexes.numel() > 0:
            timesteps_sound = sound.timesteps * timestep_scale
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_sound = self.transformer.time_embedder(
                    self.transformer.time_proj(timesteps_sound)
                )
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
            packed_tokens_sound = self.apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound.noisy_frame_indexes,
                token_shapes=sound.token_shapes,
            )

        hidden_states[sound.sequence_indexes] = packed_tokens_sound

    def decode_sound_tokens(
        self,
        packed_seq,
        last_hidden_state: torch.Tensor,
    ) -> list:
        """Decode sound predictions from transformer hidden states.

        Extracts sound predictions from hidden states via ``llm2sound`` and unpacks them back to latent shape.
        Includes a dummy forward path for graph-consistency when no noisy tokens.

        Returns:
            List of ``[C, T]`` tensors, one per sample.
        """
        sound = packed_seq.sound
        has_noisy_sound = (
            sound is not None
            and sound.tokens is not None
            and isinstance(sound.mse_loss_indexes, torch.Tensor)
            and sound.mse_loss_indexes.numel() > 0
        )

        if not has_noisy_sound:
            sound_dim = self.transformer.config.sound_dim
            dummy = torch.zeros(
                [1, sound_dim],
                device=last_hidden_state.device,
                dtype=last_hidden_state.dtype,
            )
            dummy = self.transformer.sound2llm(dummy) + self.transformer.sound_modality_embed
            dummy = self.transformer.llm2sound(dummy)
            if sound is not None and sound.tokens is not None:
                preds = [torch.zeros_like(tok) for tok in sound.tokens]
                preds[0] = preds[0] + 0.0 * dummy.sum()
            else:
                preds = [dummy]
            return preds

        assert sound is not None
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)
        preds_packed = self.transformer.llm2sound(last_hidden_state[sound.mse_loss_indexes])
        return self._unpack_sound_latents(preds_packed, sound.token_shapes, sound.noisy_frame_indexes)

    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a sound latent ``[C, T]`` to a waveform ``[audio_ch, N]``.

        Adds/removes the batch dimension expected by the sound tokenizer decoder.
        """
        assert self.sound_tokenizer is not None
        decoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        waveform = self.sound_tokenizer.decode(latent.unsqueeze(0).to(decoder_dtype))  # [1, audio_ch, N]
        return waveform.squeeze(0)  # [audio_ch, N]

    def normalize_video_databatch_inplace(
        self,
        input_video_key: str,
        data_batch: dict,
        input_key: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        input_key = input_video_key if input_key is None else input_key
        if input_key in data_batch:
            if data_batch.get("is_preprocessed", False) is True:
                for i in range(len(data_batch[input_key])):
                    assert torch.is_floating_point(data_batch[input_key][i])
                    assert torch.all((data_batch[input_key][i] >= -1.0001) & (data_batch[input_key][i] <= 1.0001))
            else:
                for i in range(len(data_batch[input_key])):
                    item = data_batch[input_key][i]
                    if isinstance(item, torch.Tensor):
                        item = [item]
                    assert item[0].dtype == torch.uint8
                    data_batch[input_key][i] = torch.stack(item).to(device=device, dtype=dtype) / 127.5 - 1.0
                data_batch["is_preprocessed"] = True

    def augment_image_dim_inplace(
        self,
        input_image_key: str,
        data_batch: dict,
        input_key: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        input_key = input_image_key if input_key is None else input_key
        if input_key in data_batch:
            if data_batch.get("is_preprocessed", False) is True:
                for i in range(len(data_batch[input_key])):
                    assert data_batch[input_key][i].shape[2] == 1
                return
            else:
                new_image_tensor_list = []
                for i in range(len(data_batch[input_key])):
                    for img_tensor in data_batch[input_key][i]:
                        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2).contiguous()
                        if img_tensor.dtype == torch.uint8:
                            img_tensor = img_tensor.to(device=device, dtype=dtype) / 127.5 - 1.0
                        new_image_tensor_list.append(img_tensor)
                data_batch[input_key] = new_image_tensor_list
                data_batch["is_preprocessed"] = True

    def remove_padding_from_latent(
        self,
        spatial_compression_factor: int,
        x0_tokens_vision: list[torch.Tensor],
        frame_size: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        cropped_latents = []
        for i in range(len(x0_tokens_vision)):
            fs = frame_size[i]
            if fs.dim() == 2:
                fs = fs[0]
            orig_h = int(fs[2].item())
            orig_w = int(fs[3].item())
            orig_h_latent = orig_h // spatial_compression_factor
            orig_w_latent = orig_w // spatial_compression_factor
            cropped_latents.append(x0_tokens_vision[i][:, :, :, :orig_h_latent, :orig_w_latent].contiguous())
        return cropped_latents

    def get_data_and_condition(
        self,
        input_image_key: str,
        input_video_key: str,
        data_batch: dict,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[int, List[torch.Tensor], Optional[torch.Tensor]]:
        """Encode vision input and return ``(num_vision_items, x0_tokens_vision, fps_vision)``."""
        assert (input_image_key in data_batch) != (input_video_key in data_batch)
        is_img = input_image_key in data_batch
        sample_vision_list = data_batch[input_image_key if is_img else input_video_key]

        # Detect multi-vision (e.g. image + video conditioning) within this single sample.
        has_multiple_vision = any(isinstance(v, (list, tuple)) and len(v) > 1 for v in sample_vision_list)
        if has_multiple_vision:
            num_vision_items = sum(len(v) for v in sample_vision_list)
            media_key = input_video_key if not is_img else input_image_key
            data_batch[media_key] = [item.unsqueeze(0) for sublist in sample_vision_list for item in sublist]
            if data_batch[media_key][0].dtype == torch.float32 and not is_img:
                data_batch["is_preprocessed"] = True
        else:
            num_vision_items = 1

        self.normalize_video_databatch_inplace(input_video_key, data_batch, device=device, dtype=dtype)
        self.augment_image_dim_inplace(input_image_key, data_batch, device=device, dtype=dtype)
        raw_state_vision = data_batch[input_image_key if is_img else input_video_key]
        x0_tokens_vision = [
            self._encode_video(raw_state_vision_i).contiguous().float() for raw_state_vision_i in raw_state_vision
        ]

        frame_size = data_batch.get("image_size", None)
        if frame_size is not None:
            x0_tokens_vision = self.remove_padding_from_latent(
                self.vae.config.scale_factor_spatial, x0_tokens_vision, frame_size
            )

        fps_raw = data_batch.get("conditioning_fps", None)
        if isinstance(fps_raw, list):
            fps_raw = torch.stack(fps_raw).flatten()
        fps_vision = fps_raw.to(device=device, dtype=dtype) if fps_raw is not None else None

        return num_vision_items, x0_tokens_vision, fps_vision

    def prepare_latents(
        self,
        prompt: str,
        cond_tokens: list[int],
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        condition_frame_indexes: Optional[List[int]] = None,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        input_caption_key: str = "ai_caption",
        input_video_key: str = "video",
        input_image_key: str = "images",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_sound: bool = False,
    ) -> tuple[
        torch.Tensor,
        SequencePlan,
        int,
        List[torch.Tensor],
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Build conditioning + initial noise for a single sample.

        Returns:
            ``(latents, sequence_plan, num_vision_items, x0_tokens_vision,
            fps_vision, x0_tokens_sound, fps_sound)``.
        """
        is_image = num_frames == 1

        conditioning_frames = None
        if image is not None:
            conditioning_frames = self._load_image_as_tensor(image, height, width)

        image_size = [torch.tensor([[height, width, height, width]], dtype=torch.float32, device=device)]

        if is_image:
            img_tensor = (
                conditioning_frames.unsqueeze(0).to(device=device, dtype=dtype)
                if conditioning_frames is not None
                else torch.zeros(1, 3, 1, height, width, dtype=dtype, device=device)
            )
            sequence_plan = SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])
            data_batch = {
                input_image_key: [img_tensor],
                "image_size": image_size,
                "is_preprocessed": True,
                "fps": torch.tensor([float(fps)], device=device),
                "conditioning_fps": torch.tensor([float(fps)], device=device),
                "num_frames": torch.tensor([num_frames], device=device),
                "sequence_plan": [sequence_plan],
                input_caption_key: [prompt],
            }
        else:
            cond_indexes = (
                condition_frame_indexes
                if condition_frame_indexes is not None
                else ([0] if conditioning_frames is not None else [])
            )
            if conditioning_frames is not None:
                video_data = torch.zeros(1, 3, num_frames, height, width, dtype=dtype)
                t_fill = min(conditioning_frames.shape[1], num_frames)
                video_data[0, :, :t_fill] = conditioning_frames[:, :t_fill].to(dtype=dtype)
                if t_fill < num_frames:
                    video_data[0, :, t_fill:] = video_data[0, :, t_fill - 1 : t_fill].expand(
                        -1, num_frames - t_fill, -1, -1
                    )
                video_tensor = video_data.to(device=device)
            else:
                video_tensor = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            sequence_plan = SequencePlan(
                has_text=True, has_vision=True, condition_frame_indexes_vision=list(cond_indexes)
            )
            data_batch = {
                input_video_key: [video_tensor],
                "image_size": image_size,
                "is_preprocessed": True,
                "fps": torch.tensor([float(fps)], device=device),
                "conditioning_fps": torch.tensor([float(fps)], device=device),
                "num_frames": torch.tensor([num_frames], device=device),
                "sequence_plan": [sequence_plan],
                input_caption_key: [prompt],
            }

        # --- Inject sound into sequence_plan ---
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
            sequence_plan.has_sound = True

        num_vision_items, x0_tokens_vision, fps_vision = self.get_data_and_condition(
            input_image_key, input_video_key, data_batch, device=device, dtype=dtype
        )

        # Run pack_input_sequence with a dummy timestep to extract the condition_mask used for noise blending.
        mask_timestep = torch.zeros((1,), dtype=torch.float32)
        packed_seq = pack_input_sequence(
            sequence_plan=sequence_plan,
            text_tokens=cond_tokens,
            input_timestep=mask_timestep,
            special_tokens=self.llm_special_tokens,
            num_vision_items=num_vision_items,
            x0_tokens_vision=x0_tokens_vision,
            fps_vision=fps_vision,
            x0_tokens_sound=x0_tokens_sound,
            fps_sound=fps_sound,
            latent_patch_size=self.transformer.config.latent_patch_size,
            include_end_of_generation_token=self.transformer.config.joint_attn_implementation == "flex",
            position_embedding_type=self.transformer.config.position_embedding_type,
            unified_3d_mrope_reset_spatial_ids=self.transformer.config.unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.transformer.config.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.transformer.config.enable_fps_modulation,
            base_fps=float(self.transformer.config.base_fps),
            temporal_compression_factor=self.vae.config.scale_factor_temporal,
        )

        assert packed_seq.vision is not None
        assert packed_seq.vision.condition_mask is not None
        assert isinstance(packed_seq.vision.condition_mask, list)
        assert x0_tokens_vision is not None

        if latents is not None:
            return (
                latents.to(device=device, dtype=dtype),
                sequence_plan,
                num_vision_items,
                x0_tokens_vision,
                fps_vision,
                x0_tokens_sound,
                fps_sound,
            )

        noise_vision_list: list[torch.Tensor] = []
        for x0_token, cond_mask in zip(x0_tokens_vision, packed_seq.vision.condition_mask, strict=True):
            pure_noise = randn_tensor(tuple(x0_token.shape), generator=generator, device=device, dtype=dtype)
            noise_vision_list.append(
                cond_mask * x0_token.to(device=device, dtype=dtype) + (1.0 - cond_mask) * pure_noise
            )

        initial_noise = torch.cat([t.reshape(-1) for t in noise_vision_list])

        # Append sound noise (all noisy: cond_mask = 0 everywhere)
        if enable_sound and packed_seq.sound is not None:
            assert isinstance(packed_seq.sound.condition_mask, list)
            for x0_sound, cond_mask_sound in zip(x0_tokens_sound, packed_seq.sound.condition_mask):
                pure_noise_sound = randn_tensor(tuple(x0_sound.shape), generator=generator, device=device, dtype=dtype)
                noise_sound = cond_mask_sound.T * x0_sound + (1.0 - cond_mask_sound.T) * pure_noise_sound
                initial_noise = torch.cat([initial_noise, noise_sound.reshape(-1)])

        return (
            initial_noise,
            sequence_plan,
            num_vision_items,
            x0_tokens_vision,
            fps_vision,
            x0_tokens_sound,
            fps_sound,
        )

    def encode_text(
        self,
        hidden_size: int,
        packed_seq,
    ) -> tuple[torch.Tensor, torch.dtype]:
        """Embed text tokens. Returns (hidden_states [N_total, H], target_dtype)."""
        packed_text_embedding = self.transformer.model.embed_tokens(packed_seq.text_ids)
        hidden_states = packed_text_embedding.new_zeros(size=(packed_seq.sequence_length, hidden_size))
        hidden_states[packed_seq.text_indexes] = packed_text_embedding
        return hidden_states, packed_text_embedding.dtype

    def encode_vision(
        self,
        timestep_scale: float,
        latent_patch_size: int,
        latent_channel: int,
        packed_seq,
        hidden_states: torch.Tensor,
        target_dtype: torch.dtype,
        fps: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, int, int]] | None:
        """Project vision tokens into hidden_states in-place. Returns original_latent_shapes."""
        if packed_seq.vision is None or packed_seq.vision.tokens is None:
            return None
        vision = packed_seq.vision
        assert vision.tokens is not None
        assert vision.token_shapes is not None
        assert isinstance(vision.sequence_indexes, torch.Tensor)
        assert isinstance(vision.timesteps, torch.Tensor)
        assert isinstance(vision.mse_loss_indexes, torch.Tensor)

        packed_tokens_vision, original_latent_shapes = self.patchify_and_pack_latents(
            latent_patch_size, latent_channel, vision.tokens, vision.token_shapes
        )
        packed_tokens_vision = self.transformer.vae2llm(packed_tokens_vision)

        if vision.mse_loss_indexes.numel() > 0:
            timesteps_vision = vision.timesteps * timestep_scale
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_vision = self.transformer.time_embedder(
                    self.transformer.time_proj(timesteps_vision)
                )
            packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
            packed_tokens_vision = self.apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision.noisy_frame_indexes,
                token_shapes=vision.token_shapes,
            )

        hidden_states[vision.sequence_indexes] = packed_tokens_vision
        return original_latent_shapes

    def _load_image_as_tensor(self, image, target_h: int, target_w: int) -> torch.Tensor:
        """Load image from PIL, path, URL, or tensor; returns [3, 1, H, W] in [-1, 1]."""
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image_str = str(image)
            if image_str.startswith("http://") or image_str.startswith("https://"):
                import io
                import urllib.request

                with urllib.request.urlopen(image_str) as resp:
                    img_bytes = resp.read()
                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                with open(image_str, "rb") as f:
                    pil_img = PILImage.open(f).convert("RGB")
            img_t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float()
        elif hasattr(image, "convert"):  # PIL.Image
            img_t = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float()
        elif isinstance(image, torch.Tensor):
            img_t = image.float()
            if img_t.dim() == 4:
                img_t = img_t.squeeze(0)
            # if already normalized to [-1, 1], skip the /127.5-1 step below
            if img_t.max() <= 1.1:
                img_4d = img_t.unsqueeze(0)
                orig_h, orig_w = img_4d.shape[2], img_4d.shape[3]
                scale = max(target_w / orig_w, target_h / orig_h)
                resize_h = int(math.ceil(scale * orig_h))
                resize_w = int(math.ceil(scale * orig_w))
                img_4d = TF.resize(img_4d, [resize_h, resize_w])
                img_4d = TF.center_crop(img_4d, [target_h, target_w])
                return img_4d.squeeze(0).unsqueeze(1)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        img_4d = img_t.unsqueeze(0)  # [1, 3, H, W]  (uint8-range [0, 255])
        orig_h, orig_w = img_4d.shape[2], img_4d.shape[3]
        scale = max(target_w / orig_w, target_h / orig_h)
        resize_h = int(math.ceil(scale * orig_h))
        resize_w = int(math.ceil(scale * orig_w))
        img_4d = TF.resize(img_4d, [resize_h, resize_w])
        img_4d = TF.center_crop(img_4d, [target_h, target_w])
        img_4d = img_4d / 127.5 - 1.0  # normalize after resize, matching load_conditioning_image
        return img_4d.squeeze(0).unsqueeze(1)  # [3, 1, H, W]

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

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        use_system_prompt: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Apply prompt-augmentation templates and tokenize cond/uncond prompts via the Qwen2 chat template.

        When ``negative_prompt`` is ``None``, a mode-specific default is chosen
        (text2image / text2video / image2video). The duration and resolution
        templates are then appended to both prompts so the LLM sees the same
        metadata it was trained with.

        Returns:
            ``(cond_text_tokens, uncond_text_tokens)`` — token-ID lists for this sample.
        """
        is_image = num_frames == 1

        if negative_prompt is None:
            if image is not None:
                negative_prompt = self.image2video_negative_prompt
            elif is_image:
                negative_prompt = self.text2image_negative_prompt
            else:
                negative_prompt = self.text2video_negative_prompt

        resolution_template = self.image_resolution_template if is_image else self.video_resolution_template

        def _apply_templates(text: str) -> str:
            if not is_image:
                text = text.rstrip(".") + ". " + self.duration_template.format(duration=num_frames / fps, fps=fps)
            text = text.rstrip(".") + ". " + resolution_template.format(height=height, width=width)
            return text

        prompt = _apply_templates(prompt)
        negative_prompt = _apply_templates(negative_prompt)

        cond_tokens = self.tokenize_caption(prompt, is_video=False, use_system_prompt=use_system_prompt)
        uncond_tokens = self.tokenize_caption(negative_prompt, is_video=False, use_system_prompt=use_system_prompt)
        return cond_tokens, uncond_tokens

    def decode_latents(self, vision_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode latents to pixel tensors of shape [C, T, H, W] in [0, 1]."""
        frames = []
        for vision_latent in vision_list:
            vision = self._decode_video(vision_latent.cuda())  # [1, C, T, H, W]
            frames.append(((1.0 + vision) / 2).clamp(0, 1).squeeze(0))
        return frames

    def _postprocess_latents(
        self,
        latents: torch.Tensor,
        num_vision_items: int,
        x0_tokens_vision: List[torch.Tensor],
        x0_tokens_sound: Optional[List[torch.Tensor]],
        enable_sound: bool,
        output_type: str,
    ) -> tuple[list[torch.Tensor], Optional[list[torch.Tensor]]]:
        """Extract vision/sound slices from the flat denoised latent, then decode.

        Returns ``(video, sound)``: ``video`` is a list of ``[C, T, H, W]`` pixel
        tensors in ``[0, 1]`` (or raw latents when ``output_type == "latent"``);
        ``sound`` is ``None`` unless ``enable_sound`` was set.
        """
        result_vision: list[torch.Tensor] = []
        offset = 0
        for j in range(num_vision_items):
            vision_shape = x0_tokens_vision[j].shape
            vision_dim = int(torch.prod(torch.tensor(vision_shape)))
            if j == num_vision_items - 1:
                result_vision.append(latents[offset : offset + vision_dim].reshape(vision_shape))
            offset += vision_dim

        result_sound: Optional[list] = None
        if enable_sound and x0_tokens_sound is not None:
            sound_shape = x0_tokens_sound[0].shape  # [sound_dim, T_sound]
            sound_dim_flat = int(torch.prod(torch.tensor(sound_shape)))
            sound_latent = latents[offset : offset + sound_dim_flat].reshape(sound_shape)
            result_sound = [self.decode_sound(sound_latent)]

        if output_type == "latent":
            return result_vision, result_sound
        return self.decode_latents(result_vision), result_sound

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
        condition_frame_indexes: Optional[List[int]] = None,
        enable_sound: bool = False,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "video",
        return_dict: bool = True,
        use_system_prompt: bool = False,
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

        # 2. Encode prompt (applies metadata templates and selects mode-specific default negative prompt)
        cond_tokens, uncond_tokens = self.encode_prompt(
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
        (
            latents,
            sequence_plan,
            num_vision_items,
            x0_tokens_vision,
            fps_vision,
            x0_tokens_sound,
            fps_sound,
        ) = self.prepare_latents(
            prompt=prompt,
            cond_tokens=cond_tokens,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            condition_frame_indexes=condition_frame_indexes,
            latents=latents,
            generator=generator,
            device=device,
            dtype=dtype,
            enable_sound=enable_sound,
        )

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Hoist per-step constants out of the loop
        config = self.transformer.config
        hidden_size = config.hidden_size
        latent_patch_size = config.latent_patch_size
        latent_channel = config.latent_channel
        patch_latent_dim = config.patch_latent_dim
        timestep_scale = config.timestep_scale
        assert config.use_moe
        include_eog = config.joint_attn_implementation == "flex"
        has_sound = x0_tokens_sound is not None and sequence_plan.has_sound

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                torch._inductor.cudagraph_mark_step_begin()
                timestep = t.reshape(1, 1)

                # Split flat latents → per-modality noisy tensors for this step
                noise_x_vision: list[torch.Tensor] = []
                offset = 0
                for j in range(num_vision_items):
                    vision_shape = x0_tokens_vision[j].shape
                    vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                    noise_x_vision.append(latents[offset : offset + vision_dim].reshape(vision_shape))
                    offset += vision_dim
                noise_x_sound: Optional[list] = None
                if has_sound:
                    sound_shape = x0_tokens_sound[0].shape
                    sound_dim_flat = int(torch.prod(torch.tensor(sound_shape)))
                    noise_x_sound = [latents[offset : offset + sound_dim_flat].reshape(sound_shape)]

                # --- Conditional pass ---
                packed_seq = pack_input_sequence(
                    sequence_plan=sequence_plan,
                    text_tokens=cond_tokens,
                    input_timestep=timestep.cpu(),
                    special_tokens=self.llm_special_tokens,
                    num_vision_items=num_vision_items,
                    x0_tokens_vision=noise_x_vision,
                    fps_vision=fps_vision,
                    x0_tokens_sound=noise_x_sound,
                    fps_sound=fps_sound if has_sound else None,
                    latent_patch_size=latent_patch_size,
                    include_end_of_generation_token=include_eog,
                    position_embedding_type=config.position_embedding_type,
                    unified_3d_mrope_reset_spatial_ids=config.unified_3d_mrope_reset_spatial_ids,
                    unified_3d_mrope_temporal_modality_margin=config.unified_3d_mrope_temporal_modality_margin,
                    enable_fps_modulation=config.enable_fps_modulation,
                    base_fps=float(config.base_fps),
                    temporal_compression_factor=self.vae.config.scale_factor_temporal,
                )
                if packed_seq.vision is not None:
                    packed_seq.vision.tokens = [x.to(device=device, dtype=dtype) for x in noise_x_vision]
                if noise_x_sound is not None and packed_seq.sound is not None:
                    packed_seq.sound.tokens = [x.to(device=device, dtype=dtype) for x in noise_x_sound]
                packed_seq.to_cuda()

                hidden_states, target_dtype = self.encode_text(hidden_size, packed_seq)
                original_latent_shapes = self.encode_vision(
                    timestep_scale, latent_patch_size, latent_channel, packed_seq, hidden_states, target_dtype, fps=fps_vision
                )
                if noise_x_sound is not None:
                    self.encode_sound_tokens(timestep_scale, packed_seq, hidden_states, target_dtype)

                und_len = packed_seq.split_lens[0]
                und_out, gen_out = self.transformer(
                    hidden_states[:und_len], hidden_states[und_len:], position_ids=packed_seq.position_ids
                )
                last_hidden_state = torch.cat([und_out, gen_out], dim=0)

                preds_vision = self.decode_vision(
                    patch_latent_dim, latent_patch_size, latent_channel, packed_seq, last_hidden_state, original_latent_shapes
                )
                preds_sound = self.decode_sound_tokens(packed_seq, last_hidden_state) if noise_x_sound is not None else None

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
                packed_seq = pack_input_sequence(
                    sequence_plan=sequence_plan,
                    text_tokens=uncond_tokens,
                    input_timestep=timestep.cpu(),
                    special_tokens=self.llm_special_tokens,
                    num_vision_items=num_vision_items,
                    x0_tokens_vision=noise_x_vision,
                    fps_vision=fps_vision,
                    x0_tokens_sound=noise_x_sound,
                    fps_sound=fps_sound if has_sound else None,
                    latent_patch_size=latent_patch_size,
                    include_end_of_generation_token=include_eog,
                    position_embedding_type=config.position_embedding_type,
                    unified_3d_mrope_reset_spatial_ids=config.unified_3d_mrope_reset_spatial_ids,
                    unified_3d_mrope_temporal_modality_margin=config.unified_3d_mrope_temporal_modality_margin,
                    enable_fps_modulation=config.enable_fps_modulation,
                    base_fps=float(config.base_fps),
                    temporal_compression_factor=self.vae.config.scale_factor_temporal,
                )
                if packed_seq.vision is not None:
                    packed_seq.vision.tokens = [x.to(device=device, dtype=dtype) for x in noise_x_vision]
                if noise_x_sound is not None and packed_seq.sound is not None:
                    packed_seq.sound.tokens = [x.to(device=device, dtype=dtype) for x in noise_x_sound]
                packed_seq.to_cuda()

                hidden_states, target_dtype = self.encode_text(hidden_size, packed_seq)
                original_latent_shapes = self.encode_vision(
                    timestep_scale, latent_patch_size, latent_channel, packed_seq, hidden_states, target_dtype, fps=fps_vision
                )
                if noise_x_sound is not None:
                    self.encode_sound_tokens(timestep_scale, packed_seq, hidden_states, target_dtype)

                und_len = packed_seq.split_lens[0]
                und_out, gen_out = self.transformer(
                    hidden_states[:und_len], hidden_states[und_len:], position_ids=packed_seq.position_ids
                )
                last_hidden_state = torch.cat([und_out, gen_out], dim=0)

                preds_vision = self.decode_vision(
                    patch_latent_dim, latent_patch_size, latent_channel, packed_seq, last_hidden_state, original_latent_shapes
                )
                preds_sound = self.decode_sound_tokens(packed_seq, last_hidden_state) if noise_x_sound is not None else None

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
                latents = self.scheduler.step(
                    velocity_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

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
            num_vision_items=num_vision_items,
            x0_tokens_vision=x0_tokens_vision,
            x0_tokens_sound=x0_tokens_sound,
            enable_sound=enable_sound,
            output_type=output_type,
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, sound)
        return Cosmos3OmniPipelineOutput(video=video, sound=sound)
