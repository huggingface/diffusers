# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GenerationDataClean:
    batch_size: int
    is_image_batch: bool
    raw_state_vision: Optional[List[torch.Tensor]] = None
    x0_tokens_vision: Optional[List[torch.Tensor]] = None
    fps_vision: Optional[torch.Tensor] = None
    num_vision_items_per_sample: Optional[List[int]] = None
    x0_tokens_action: Optional[List[torch.Tensor]] = None
    fps_action: Optional[torch.Tensor] = None
    action_domain_id: Optional[List[torch.Tensor]] = None
    raw_action_dim: Optional[List[Optional[torch.Tensor]]] = None
    x0_tokens_sound: Optional[List[torch.Tensor]] = None
    fps_sound: Optional[torch.Tensor] = None


# A packed sequence is a plain dict at runtime; the alias exists only for type hints.
SequencePack = dict[str, Any]


# ------------------------------------
# Internal helpers
# ------------------------------------


def _pad_to_N(N: int, x: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] <= N
    padded = x.new_zeros((N, *x.shape[1:]))
    padded[: x.shape[0]] = x
    return padded


def _compute_mode_indices_and_offsets(
    split_lens: list[int],
    attn_modes: list[str],
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (token indices into the packed sequence, per-segment start offsets)
    for every split whose ``attn_mode`` equals ``mode``."""
    indices: list[int] = []
    offsets: list[int] = [0]
    next_offset = 0
    start = 0
    for split_len, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == mode:
            indices.extend(range(start, start + split_len))
            next_offset += split_len
            offsets.append(next_offset)
        start += split_len
    return (
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
    )


def _init_sequence_pack(
    sample_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    device: torch.device,
) -> dict:
    """Compute the metadata fields (offsets, mode indices, max lengths) that
    every ``SequencePack`` carries."""
    max_sample_len = max(sample_lens)
    max_causal_len = max((sl for sl, m in zip(split_lens, attn_modes) if m == "causal"), default=0)
    max_full_len = max((sl for sl, m in zip(split_lens, attn_modes) if m == "full"), default=0)
    sample_lens_cu = torch.tensor([0] + sample_lens, device=device, dtype=torch.int32)
    sample_offsets = torch.cumsum(sample_lens_cu, dim=0, dtype=torch.int32)
    causal_indices, causal_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "causal", device)
    full_indices, full_only_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "full", device)
    return {
        "sample_offsets": sample_offsets,
        "max_sample_len": max_sample_len,
        "max_causal_len": max_causal_len,
        "max_full_len": max_full_len,
        "_causal_indices": causal_indices,
        "_full_indices": full_indices,
        "_causal_seq_offsets": causal_seq_offsets,
        "_full_only_seq_offsets": full_only_seq_offsets,
        "_num_causal_tokens": len(causal_indices),
        "_num_full_tokens": len(full_indices),
    }


class SplitInfo:
    """Per-batch attention metadata consumed by the transformer."""

    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
    ):
        assert sum(sample_lens) == sum(split_lens)
        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == "causal":
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == "full":
                max_full_len = max(max_full_len, split_len)
        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)
        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens


def build_packed_sequence(
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
) -> tuple[SequencePack, SplitInfo]:
    """Scatter ``packed_sequence`` into causal/full mode splits and return the
    pack alongside the attention metadata needed by the transformer."""
    assert sum(sample_lens) == packed_sequence.shape[0]
    meta = _init_sequence_pack(sample_lens, split_lens, attn_modes, packed_sequence.device)
    pack: SequencePack = {
        **meta,
        "max_num_tokens": sum(sample_lens),
        "causal_seq": packed_sequence[meta["_causal_indices"]],
        "full_only_seq": packed_sequence[meta["_full_indices"]],
        "is_sharded": False,
    }
    attention_meta = SplitInfo(split_lens=split_lens, attn_modes=attn_modes, sample_lens=sample_lens)
    return pack, attention_meta


# ------------------------------------
# Public API
# ------------------------------------
#
# A SequencePack is a plain dict carrying token tensors split by attention mode:
#   - ``causal_seq``: tokens that participate in causal attention (understanding).
#   - ``full_only_seq``: tokens that participate in full attention (generation).
# Plus the metadata produced by ``_init_sequence_pack`` (sample offsets, mode
# indices, padded max lengths, ``is_sharded``).
#
# ``is_sharded=True`` denotes a context-parallel local shard where the tensors
# are stored back-to-back as ``[causal | full_only]`` and must be sliced linearly
# rather than scattered via the saved mode indices.


def from_mode_splits(
    causal_seq: torch.Tensor,
    full_only_seq: torch.Tensor,
    orig: SequencePack,
    is_sharded: bool | None = None,
) -> SequencePack:
    """Build a new pack from mode splits, copying metadata from ``orig``.

    ``is_sharded=None`` inherits from ``orig``; pass ``True`` to mark a
    context-parallel local shard.
    """
    out = dict(orig)
    out["causal_seq"] = causal_seq
    out["full_only_seq"] = full_only_seq
    out["is_sharded"] = orig.get("is_sharded", False) if is_sharded is None else is_sharded
    return out


def zeros_like(orig: SequencePack, shape: Tuple[int, ...] | torch.Size | None = None) -> SequencePack:
    """Build a pack with zero tensors and the same metadata as ``orig``.

    If ``shape`` is provided its leading dim must be ``-1`` (token count is
    preserved from ``orig``); trailing dims override ``orig``'s.
    """
    if shape is None:
        shape_causal = orig["causal_seq"].shape
        shape_full = orig["full_only_seq"].shape
    else:
        assert len(shape) >= 1 and shape[0] == -1
        shape_causal = (orig["causal_seq"].shape[0],) + tuple(shape)[1:]
        shape_full = (orig["full_only_seq"].shape[0],) + tuple(shape)[1:]
    causal_seq = torch.zeros(shape_causal, device=orig["causal_seq"].device, dtype=orig["causal_seq"].dtype)
    full_only_seq = torch.zeros(shape_full, device=orig["full_only_seq"].device, dtype=orig["full_only_seq"].dtype)
    return from_mode_splits(causal_seq, full_only_seq, orig)


def from_joint(packed_sequence: torch.Tensor, metadata_source: SequencePack) -> SequencePack:
    """Split a single packed tensor of shape ``[seq_len, ...]`` into mode splits.

    When ``metadata_source`` is a context-parallel shard the input is assumed to
    be laid out as ``[causal | full_only]`` and is sliced linearly; otherwise
    tokens are scattered via the saved mode indices and the splits are zero-padded
    to the max lengths recorded in ``metadata_source``.
    """
    if metadata_source["is_sharded"]:
        n_causal = metadata_source["causal_seq"].shape[0]
        causal_seq = packed_sequence[:n_causal]
        full_only_seq = packed_sequence[n_causal:]
    else:
        causal_seq = packed_sequence[metadata_source["_causal_indices"]]
        full_only_seq = packed_sequence[metadata_source["_full_indices"]]
        max_causal_len = metadata_source["causal_seq"].shape[0]
        max_full_len = metadata_source["full_only_seq"].shape[0]
        causal_seq = _pad_to_N(max_causal_len, causal_seq)
        full_only_seq = _pad_to_N(max_full_len, full_only_seq)
    return from_mode_splits(causal_seq, full_only_seq, metadata_source)


def from_und_gen_splits(und_seq: torch.Tensor, gen_seq: torch.Tensor, orig: SequencePack) -> SequencePack:
    """Build a new pack from und/gen splits (und == causal, gen == full)."""
    return from_mode_splits(und_seq, gen_seq, orig)


def get_und_seq(pack: SequencePack) -> torch.Tensor:
    """Return the understanding (causal) token tensor."""
    return pack["causal_seq"]


def set_und_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """Overwrite the understanding (causal) token tensor in-place."""
    pack["causal_seq"] = value


def get_gen_seq(pack: SequencePack) -> torch.Tensor:
    """Return the generating (full-only) token tensor."""
    return pack["full_only_seq"]


def set_gen_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """Overwrite the generating (full-only) token tensor in-place."""
    pack["full_only_seq"] = value


def get_device_and_dtype(pack: SequencePack) -> Tuple[torch.device, torch.dtype]:
    """Return ``(device, dtype)`` of the causal token tensor."""
    return pack["causal_seq"].device, pack["causal_seq"].dtype


def get_all_seq(pack: SequencePack) -> torch.Tensor:
    """Reassemble both mode splits into a single ``[seq_len, ...]`` tensor.

    Not supported on context-parallel shards.
    """
    if pack["is_sharded"]:
        raise NotImplementedError("get_all_seq is not supported in context parallel sharded mode")
    causal_seq = pack["causal_seq"]
    full_only_seq = pack["full_only_seq"]
    causal_indices = pack["_causal_indices"]
    full_indices = pack["_full_indices"]
    out = causal_seq.new_zeros(int(causal_indices.shape[0] + full_indices.shape[0]), *causal_seq.shape[1:])
    if causal_seq.shape[0] > 0:
        out[causal_indices] = causal_seq[: causal_indices.shape[0]]
    if full_only_seq.shape[0] > 0:
        out[full_indices] = full_only_seq[: full_indices.shape[0]]
    return out


def get_causal_seq(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(causal_seq, _causal_seq_offsets)``."""
    return pack["causal_seq"], pack["_causal_seq_offsets"]


def get_full_only_seq(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(full_only_seq, _full_only_seq_offsets)``."""
    return pack["full_only_seq"], pack["_full_only_seq_offsets"]


# ============================================================================
# 3D mRoPE position ID utilities
# ============================================================================


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for text tokens.

    For text tokens, all three axes (temporal, height, width) share the same
    monotonically increasing position IDs, starting from ``temporal_offset``.

    Args:
        num_tokens: Number of text tokens.
        temporal_offset: Current temporal offset to start from.
        use_float_positions: If True, generate float position IDs.

    Returns:
        Tuple of position IDs tensor of shape (3, num_tokens) and updated temporal offset.
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
    """Generate 3D mRoPE position IDs for VAE vision tokens (image/video latents).

    Args:
        grid_t: Number of temporal frames in the latent grid.
        grid_h: Height of the latent grid (after patchification).
        grid_w: Width of the latent grid (after patchification).
        temporal_offset: Current temporal offset.
        reset_spatial_indices: If True, spatial indices start from 0 for each vision segment.
        fps: Frames per second. If None, FPS modulation is disabled.
        base_fps: Base FPS for normalization.
        temporal_compression_factor: VAE temporal compression factor.
        base_temporal_compression_factor: Base temporal compression factor.
        start_frame_offset: Offset added to frame indices before FPS scaling.

    Returns:
        Tuple of position IDs tensor of shape (3, grid_t * grid_h * grid_w) and updated offset.
    """
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
# Data structures for sequence packing
# ============================================================================


@dataclass
class ModalityData:
    """Unified container for a single generation modality's data.

    This dataclass serves dual purposes:
    1. During packing: Acts as a builder, accumulating data in lists
    2. After finalize(): Holds finalized tensors ready for model consumption
    """

    sequence_indexes: list[int] | torch.Tensor = field(default_factory=list)
    timesteps: list[float] | torch.Tensor = field(default_factory=list)
    mse_loss_indexes: list[int] | torch.Tensor = field(default_factory=list)
    token_shapes: list = field(default_factory=list)

    tokens: list[torch.Tensor] = field(default_factory=list)
    condition_mask: list[torch.Tensor] = field(default_factory=list)
    noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)
    domain_id: list[torch.Tensor] = field(default_factory=list)
    raw_action_dim: list[torch.Tensor | None] | None = field(default_factory=list)


@dataclass
class PackedSequence:
    """Unified sequence container - works as builder during packing and final output."""

    # Sequence structure
    sample_lens: list[int] = field(default_factory=list)
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    is_image_batch: bool = False
    sequence_length: int = 0

    # Build-time tracking
    curr: int = 0

    # Text modality (list during build, tensor after finalize)
    text_ids: list[int] | torch.Tensor = field(default_factory=list)
    text_indexes: list[int] | torch.Tensor = field(default_factory=list)
    position_ids: list[int] | torch.Tensor = field(default_factory=list)

    # Loss computation - Cross Entropy (text)
    label_ids: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_indexes: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_weights: list[float] | torch.Tensor | None = field(default_factory=list)

    # Build-time mRoPE tracking
    _use_mrope: bool = False
    _mrope_temporal_offset: int | float = 0
    _mrope_reset_spatial: bool = True

    # Temporal causal
    null_action_supertokens: bool = False

    # Generation modalities
    vision: ModalityData | None = None
    action: ModalityData | None = None
    sound: ModalityData | None = None

    def finalize(
        self,
        gen_data_clean: "GenerationDataClean",
    ) -> "PackedSequence":
        """Convert all lists to tensors and compute derived values."""
        sequence_length = sum(self.sample_lens)
        sample_lens = self.sample_lens.copy()
        split_lens = self.split_lens.copy()
        attn_modes = self.attn_modes.copy()

        label_ids: torch.Tensor | None = None
        ce_loss_indexes: torch.Tensor | None = None
        ce_loss_weights: torch.Tensor | None = None
        if self.label_ids and len(self.label_ids) > 0:
            label_ids = torch.tensor(self.label_ids)
            ce_loss_indexes = torch.tensor(self.ce_loss_indexes)
            ce_loss_weights = torch.tensor(self.ce_loss_weights)

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

        action: ModalityData | None = None
        if self.action is not None and len(self.action.sequence_indexes) > 0:
            action = ModalityData(
                sequence_indexes=torch.tensor(self.action.sequence_indexes, dtype=torch.long),
                timesteps=torch.tensor(self.action.timesteps),
                mse_loss_indexes=torch.tensor(self.action.mse_loss_indexes, dtype=torch.long),
                token_shapes=list(self.action.token_shapes),
                tokens=self.action.tokens,
                condition_mask=list(self.action.condition_mask),
                noisy_frame_indexes=list(self.action.noisy_frame_indexes),
                domain_id=(
                    gen_data_clean.action_domain_id
                    if gen_data_clean.action_domain_id is not None
                    else [torch.zeros(1, dtype=torch.long)] * len(self.action.token_shapes)
                ),
                raw_action_dim=gen_data_clean.raw_action_dim,
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
            is_image_batch=gen_data_clean.is_image_batch,
            text_ids=torch.tensor(self.text_ids, dtype=torch.long),
            text_indexes=torch.tensor(self.text_indexes, dtype=torch.long),
            position_ids=position_ids,
            label_ids=label_ids,
            ce_loss_indexes=ce_loss_indexes,
            ce_loss_weights=ce_loss_weights,
            vision=vision,
            action=action,
            sound=sound,
            null_action_supertokens=self.null_action_supertokens,
        )

    def to_cuda(self) -> None:
        """Move every tensor field (and modality sub-objects) to CUDA in-place."""
        for attr in ("text_ids", "text_indexes", "position_ids", "label_ids", "ce_loss_indexes", "ce_loss_weights"):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.cuda())
        for modality in (self.vision, self.action, self.sound):
            if modality is not None:
                _modality_to_cuda(modality)


def _modality_to_cuda(modality: "ModalityData") -> None:
    for attr in ("sequence_indexes", "timesteps", "mse_loss_indexes"):
        val = getattr(modality, attr)
        if isinstance(val, torch.Tensor):
            setattr(modality, attr, val.cuda())
    modality.tokens = [t.cuda() for t in modality.tokens]
    modality.condition_mask = [m.cuda() for m in modality.condition_mask]
    modality.noisy_frame_indexes = [i.cuda() for i in modality.noisy_frame_indexes]
    modality.domain_id = [d.cuda() for d in modality.domain_id]
    if modality.raw_action_dim is not None:
        modality.raw_action_dim = [d.cuda() if d is not None else None for d in modality.raw_action_dim]


@dataclass
class SequencePlan:
    """Plan describing which modalities are present in a sample."""

    has_text: bool
    has_vision: bool = False
    condition_frame_indexes_vision: list[int] = field(default_factory=list)
    has_action: bool = False
    condition_frame_indexes_action: list[int] = field(default_factory=list)
    has_sound: bool = False
    condition_frame_indexes_sound: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "has_text": self.has_text,
            "has_vision": self.has_vision,
            "has_action": self.has_action,
            "has_sound": self.has_sound,
            "condition_frame_indexes_vision": self.condition_frame_indexes_vision,
            "condition_frame_indexes_action": self.condition_frame_indexes_action,
            "condition_frame_indexes_sound": self.condition_frame_indexes_sound,
        }


# ============================================================================
# Helper functions for packing sequences
# ============================================================================


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
    assert isinstance(packed_seq.label_ids, list)
    assert isinstance(packed_seq.ce_loss_indexes, list)
    assert isinstance(packed_seq.ce_loss_weights, list)

    curr = packed_seq.curr

    if "bos_token_id" in special_tokens:
        shifted_text_ids = [special_tokens["bos_token_id"]] + text_ids
    else:
        shifted_text_ids = text_ids

    split_len = 0

    packed_seq.text_ids.extend(shifted_text_ids)
    packed_seq.text_indexes.extend(range(curr, curr + len(shifted_text_ids)))

    packed_seq.ce_loss_indexes.extend(range(curr, curr + len(shifted_text_ids)))
    packed_seq.ce_loss_weights.extend([1.0] * len(shifted_text_ids))
    packed_seq.label_ids.extend(text_ids[1:] + [special_tokens["eos_token_id"]])

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


def _pack_action_tokens(
    packed_seq: PackedSequence,
    input_action_tokens: torch.Tensor,
    condition_frame_indexes_action: list[int],
    input_timestep: float,
    curr_rope_id: int,
    action_temporal_offset: int | float = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    action_fps: float | None = None,
    base_temporal_compression_factor: int | None = None,
) -> int:
    """Pack action tokens into the sequence."""
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    action_split_len = input_action_tokens.shape[0]

    if packed_seq.action is None:
        packed_seq.action = ModalityData()

    assert isinstance(packed_seq.action.sequence_indexes, list)
    assert isinstance(packed_seq.action.mse_loss_indexes, list)
    assert isinstance(packed_seq.action.timesteps, list)
    assert isinstance(packed_seq.action.tokens, list)

    action_indexes = list(range(curr, curr + action_split_len))
    packed_seq.action.sequence_indexes.extend(action_indexes)
    packed_seq.action.token_shapes.append((action_split_len,))
    packed_seq.action.tokens.append(input_action_tokens)

    condition_set = {idx for idx in condition_frame_indexes_action if 0 <= idx < action_split_len}
    assert isinstance(packed_seq.action.condition_mask, list)

    action_condition_mask = torch.zeros(
        (action_split_len, 1), device=input_action_tokens.device, dtype=input_action_tokens.dtype
    )
    for frame_idx in condition_set:
        action_condition_mask[frame_idx, 0] = 1.0
    packed_seq.action.condition_mask.append(action_condition_mask)

    action_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(action_split_len) if idx not in condition_set],
        device=input_action_tokens.device,
        dtype=torch.long,
    )
    assert isinstance(packed_seq.action.noisy_frame_indexes, list)
    packed_seq.action.noisy_frame_indexes.append(action_noisy_frame_indexes)

    frame_token_stride = 1
    for frame_idx in range(action_split_len):
        if frame_idx in condition_set:
            continue
        frame_start = curr + frame_idx * frame_token_stride
        frame_end = frame_start + frame_token_stride
        packed_seq.action.mse_loss_indexes.extend(range(frame_start, frame_end))
        packed_seq.action.timesteps.extend([input_timestep] * frame_token_stride)

    if packed_seq._use_mrope:
        effective_fps = action_fps if enable_fps_modulation else None
        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=action_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,
            base_temporal_compression_factor=base_temporal_compression_factor,
            start_frame_offset=1,
        )
        packed_seq.position_ids.append(action_mrope_ids)
    else:
        packed_seq.position_ids.extend([curr_rope_id] * action_split_len)

    packed_seq.curr = curr + action_split_len
    return action_split_len


def _pack_sound_tokens(
    packed_seq: PackedSequence,
    input_sound_tokens: torch.Tensor,
    condition_frame_indexes_sound: list[int],
    input_timestep: float,
    curr_rope_id: int,
    sound_temporal_offset: int | float = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    sound_fps: float | None = None,
) -> int:
    """Pack sound/audio tokens into the sequence."""
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    _, sound_split_len = input_sound_tokens.shape

    if packed_seq.sound is None:
        packed_seq.sound = ModalityData()

    assert isinstance(packed_seq.sound.sequence_indexes, list)
    assert isinstance(packed_seq.sound.mse_loss_indexes, list)
    assert isinstance(packed_seq.sound.timesteps, list)
    assert isinstance(packed_seq.sound.tokens, list)

    packed_seq.sound.token_shapes.append((sound_split_len, 1, 1))
    packed_seq.sound.sequence_indexes.extend(range(curr, curr + sound_split_len))
    packed_seq.sound.tokens.append(input_sound_tokens)

    condition_set = {idx for idx in condition_frame_indexes_sound if 0 <= idx < sound_split_len}
    assert isinstance(packed_seq.sound.condition_mask, list)

    sound_condition_mask = torch.zeros(
        (sound_split_len, 1), device=input_sound_tokens.device, dtype=input_sound_tokens.dtype
    )
    for frame_idx in condition_set:
        sound_condition_mask[frame_idx, 0] = 1.0
    packed_seq.sound.condition_mask.append(sound_condition_mask)

    sound_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(sound_split_len) if idx not in condition_set],
        device=input_sound_tokens.device,
        dtype=torch.long,
    )
    assert isinstance(packed_seq.sound.noisy_frame_indexes, list)
    packed_seq.sound.noisy_frame_indexes.append(sound_noisy_frame_indexes)

    for frame_idx in range(sound_split_len):
        if frame_idx in condition_set:
            continue
        frame_start = curr + frame_idx
        frame_end = frame_start + 1
        packed_seq.sound.mse_loss_indexes.extend(range(frame_start, frame_end))
        packed_seq.sound.timesteps.extend([input_timestep])

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


def _pack_supertokens_temporal_causal(
    packed_seq: "PackedSequence",
    input_vision_tokens: torch.Tensor,
    input_action_tokens: torch.Tensor | None,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    curr_rope_id: int,
    latent_patch_size: int,
    temporal_compression_factor: int,
    action_dim: int,
    vision_fps: float | None = None,
    action_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
) -> tuple[int, bool]:
    """Pack vision and action tokens in interleaved supertoken order for temporal causal attention.

    Buffer layout: [action_t0, vision_t0, action_t1, vision_t1, ..., action_{T-1}, vision_{T-1}]
    """
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    tcf = temporal_compression_factor
    patches_per_frame = patch_h * patch_w
    supertoken_len = tcf + patches_per_frame

    if packed_seq.vision is None:
        packed_seq.vision = ModalityData()
    if packed_seq.action is None:
        packed_seq.action = ModalityData()

    assert isinstance(packed_seq.vision.sequence_indexes, list)
    assert isinstance(packed_seq.vision.mse_loss_indexes, list)
    assert isinstance(packed_seq.vision.timesteps, list)
    assert isinstance(packed_seq.vision.tokens, list)
    assert isinstance(packed_seq.vision.condition_mask, list)
    assert isinstance(packed_seq.action.sequence_indexes, list)
    assert isinstance(packed_seq.action.mse_loss_indexes, list)
    assert isinstance(packed_seq.action.timesteps, list)
    assert isinstance(packed_seq.action.tokens, list)
    assert isinstance(packed_seq.action.condition_mask, list)

    device = input_vision_tokens.device
    dtype = input_vision_tokens.dtype

    null_tokens = torch.zeros(tcf, action_dim, device=device, dtype=dtype)
    if input_action_tokens is not None:
        if input_action_tokens.dim() == 3:
            real_actions = input_action_tokens.squeeze(0)
        else:
            real_actions = input_action_tokens
        if latent_t == 1:
            all_action_tokens = real_actions
        else:
            all_action_tokens = torch.cat([null_tokens, real_actions], dim=0)
    else:
        all_action_tokens = torch.zeros(latent_t * tcf, action_dim, device=device, dtype=dtype)

    null_action_flag = not (latent_t == 1 and input_action_tokens is not None)

    packed_seq.vision.token_shapes.append((latent_t, patch_h, patch_w))
    packed_seq.vision.tokens.append(input_vision_tokens)

    condition_set_vision = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    for fidx in condition_set_vision:
        vision_condition_mask[fidx, 0, 0] = 1.0
    packed_seq.vision.condition_mask.append(vision_condition_mask)

    vision_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in condition_set_vision],
        device=device,
        dtype=torch.long,
    )
    packed_seq.vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

    packed_seq.action.token_shapes.append((latent_t * tcf,))
    packed_seq.action.tokens.append(all_action_tokens)

    action_condition_mask = torch.ones((latent_t * tcf, 1), device=device, dtype=dtype)
    packed_seq.action.condition_mask.append(action_condition_mask)

    curr = packed_seq.curr
    total_split_len = 0

    if packed_seq._use_mrope:
        temporal_offset = packed_seq._mrope_temporal_offset
        effective_action_fps = action_fps if enable_fps_modulation else None
        effective_vision_fps = vision_fps if enable_fps_modulation else None

        fps_active = effective_action_fps is not None
        t_dtype = torch.float32 if fps_active else torch.long
        t_offset = float(temporal_offset) if fps_active else int(temporal_offset)
        null_t = torch.full((tcf,), t_offset, dtype=t_dtype)
        null_hw = torch.zeros(tcf, dtype=t_dtype)
        null_ids = torch.stack([null_t, null_hw, null_hw])  # [3,tcf]

        def _real_action_ids(n_frames: int, start_frame_offset: int) -> torch.Tensor:
            flat, _ = get_3d_mrope_ids_vae_tokens(
                grid_t=n_frames * tcf,
                grid_h=1,
                grid_w=1,
                temporal_offset=temporal_offset,
                reset_spatial_indices=packed_seq._mrope_reset_spatial,
                fps=effective_action_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,
                base_temporal_compression_factor=tcf,
                start_frame_offset=start_frame_offset,
            )
            return flat.reshape(3, n_frames, tcf)

        if latent_t > 1:
            null_ids_3d = null_ids.reshape(3, 1, tcf)
            real_ids_3d = _real_action_ids(latent_t - 1, start_frame_offset=1)
            action_ids_3d = torch.cat([null_ids_3d, real_ids_3d], dim=1)
        elif input_action_tokens is None:
            action_ids_3d = null_ids.reshape(3, 1, tcf)
        else:
            action_ids_3d = _real_action_ids(1, start_frame_offset=0)

        vision_ids_flat, new_offset = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_vision_fps,
            base_fps=base_fps,
            temporal_compression_factor=tcf,
        )
        vision_ids_3d = vision_ids_flat.reshape(3, latent_t, patches_per_frame)

        interleaved_ids = torch.cat([action_ids_3d, vision_ids_3d], dim=2).reshape(3, latent_t * supertoken_len)
        packed_seq.position_ids.append(interleaved_ids)
        packed_seq._mrope_temporal_offset = new_offset

    for frame_t in range(latent_t):
        action_indexes = list(range(curr, curr + tcf))
        packed_seq.action.sequence_indexes.extend(action_indexes)
        curr += tcf
        total_split_len += tcf

        if not packed_seq._use_mrope:
            packed_seq.position_ids.extend([curr_rope_id] * tcf)

        frame_indexes = list(range(curr, curr + patches_per_frame))
        packed_seq.vision.sequence_indexes.extend(frame_indexes)
        curr += patches_per_frame
        total_split_len += patches_per_frame

        if not packed_seq._use_mrope:
            packed_seq.position_ids.extend([curr_rope_id] * patches_per_frame)

        if frame_t not in condition_set_vision:
            packed_seq.vision.mse_loss_indexes.extend(frame_indexes)
            frame_ts = input_timestep[frame_t].item() if isinstance(input_timestep, torch.Tensor) else input_timestep
            packed_seq.vision.timesteps.extend([frame_ts] * patches_per_frame)

    packed_seq.curr = curr
    return total_split_len, null_action_flag


# ============================================================================
# Main packing functions
# ============================================================================


def pack_input_sequence(
    sequence_plans: list[SequencePlan],
    input_text_indexes: list[list[int]],
    gen_data_clean: GenerationDataClean,
    input_timesteps: torch.Tensor,
    special_tokens: dict[str, int],
    max_num_tokens: int | None = None,
    latent_patch_size: int = 1,
    skip_text_tokens: bool = False,
    include_end_of_generation_token: bool = False,
    position_embedding_type: str = "3d_rope",
    unified_3d_mrope_reset_spatial_ids: bool = True,
    unified_3d_mrope_temporal_modality_margin: int = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    video_temporal_causal: bool = False,
    action_dim: int = 32,
    initial_mrope_temporal_offset: int | float = 0,
) -> PackedSequence:
    """Pack a sequence of input strings and VAE latents into a packed tensor format.

    Args:
        sequence_plans: List of SequencePlan items describing which modalities are present.
        input_text_indexes: List of text token ID sequences.
        gen_data_clean: GenerationDataClean containing vision, action, and sound tensors.
        input_timesteps: Diffusion timesteps for each sample. Shape (B,) or (B, 1).
        special_tokens: Dictionary containing special token IDs.
        max_num_tokens: Maximum number of tokens (unused, kept for API compatibility).
        latent_patch_size: Patch size used by the network to pack latents.
        skip_text_tokens: If True, skip packing text tokens.
        include_end_of_generation_token: If True, append end-of-generation token.
        position_embedding_type: Position embedding type for vision tokens.
        unified_3d_mrope_reset_spatial_ids: If True, spatial indices start from 0 per segment.
        unified_3d_mrope_temporal_modality_margin: Temporal margin between text and vision.
        enable_fps_modulation: If True, scale temporal position IDs based on video FPS.
        base_fps: Base FPS for normalization.
        temporal_compression_factor: VAE temporal compression factor.
        video_temporal_causal: If True, pack vision and action as interleaved supertokens.
        action_dim: Action token dimension for temporal causal packing.
        initial_mrope_temporal_offset: Initial temporal offset for AR inference.

    Returns:
        PackedSequence containing all packed tensors and metadata.
    """
    del max_num_tokens

    assert special_tokens is not None, "Special tokens must be provided"
    assert isinstance(input_timesteps, torch.Tensor), "input_timesteps must be a tensor"
    if input_timesteps.is_cuda:
        raise ValueError("input_timesteps must be on CPU, not CUDA")
    if isinstance(input_text_indexes, torch.Tensor):
        raise ValueError("input_text_tokens must be a list, not a tensor")

    packed_seq = PackedSequence()
    packed_seq._use_mrope = position_embedding_type == "unified_3d_mrope"
    packed_seq._mrope_reset_spatial = unified_3d_mrope_reset_spatial_ids

    idx_text = 0
    idx_vision = 0
    idx_action = 0
    idx_sound = 0
    null_action_flags: list[bool] = []

    if not skip_text_tokens:
        for plan in sequence_plans:
            assert plan.has_text, "All sequence plans must have has_text=True when skip_text_tokens=False"

    for sample_idx, sequence_plan in enumerate(sequence_plans):
        curr_rope_id = 0
        sample_len = 0

        packed_seq._mrope_temporal_offset = initial_mrope_temporal_offset

        _ts = input_timesteps[sample_idx]
        input_timestep = _ts.item() if _ts.numel() == 1 else _ts

        if sequence_plan.has_text and not skip_text_tokens:
            text_ids = input_text_indexes[idx_text]
            idx_text += 1

            has_generation_for_sample = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
            curr_rope_id, _, text_sample_len = _pack_text_tokens(
                packed_seq,
                text_ids,
                special_tokens,
                curr_rope_id,
                has_generation=has_generation_for_sample,
                use_float_positions=enable_fps_modulation,
            )
            sample_len += text_sample_len
            packed_seq._mrope_temporal_offset += unified_3d_mrope_temporal_modality_margin

        vision_start_temporal_offset = packed_seq._mrope_temporal_offset

        if video_temporal_causal and sequence_plan.has_vision:
            assert position_embedding_type == "unified_3d_mrope", (
                "video_temporal_causal=True requires position_embedding_type='unified_3d_mrope'"
            )
            input_vision_tokens = gen_data_clean.x0_tokens_vision[idx_vision]
            idx_vision += 1

            vision_fps = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_vision is not None
                and idx_vision - 1 < len(gen_data_clean.fps_vision)
            ):
                vision_fps = float(gen_data_clean.fps_vision[idx_vision - 1].item())

            input_action_tokens_tc: torch.Tensor | None = None
            action_fps_tc: float | None = None
            if sequence_plan.has_action:
                input_action_tokens_tc = gen_data_clean.x0_tokens_action[idx_action]
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps_tc = float(gen_data_clean.fps_action[idx_action].item())
                idx_action += 1

            supertoken_split_len, null_flag = _pack_supertokens_temporal_causal(
                packed_seq=packed_seq,
                input_vision_tokens=input_vision_tokens,
                input_action_tokens=input_action_tokens_tc,
                condition_frame_indexes_vision=sequence_plan.condition_frame_indexes_vision,
                input_timestep=input_timestep,
                curr_rope_id=curr_rope_id,
                latent_patch_size=latent_patch_size,
                temporal_compression_factor=temporal_compression_factor,
                action_dim=action_dim,
                vision_fps=vision_fps,
                action_fps=action_fps_tc,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
            )
            null_action_flags.append(null_flag)
            sample_len += supertoken_split_len
            vision_split_len = supertoken_split_len
            action_split_len = 0

        else:
            if sequence_plan.has_vision:
                num_vis = (
                    gen_data_clean.num_vision_items_per_sample[sample_idx]
                    if gen_data_clean.num_vision_items_per_sample is not None
                    else 1
                )

                vision_split_len = 0
                for item_idx in range(num_vis):
                    input_vision_tokens = gen_data_clean.x0_tokens_vision[idx_vision]

                    vision_fps: float | None = None
                    if (
                        enable_fps_modulation
                        and gen_data_clean.fps_vision is not None
                        and idx_vision < len(gen_data_clean.fps_vision)
                    ):
                        vision_fps = float(gen_data_clean.fps_vision[idx_vision].item())

                    idx_vision += 1

                    if num_vis > 1 and item_idx < num_vis - 1:
                        latent_t = input_vision_tokens.shape[2]
                        item_condition_frames = list(range(latent_t))
                    else:
                        item_condition_frames = sequence_plan.condition_frame_indexes_vision

                    item_split_len = _pack_vision_tokens(
                        packed_seq=packed_seq,
                        input_vision_tokens=input_vision_tokens,
                        condition_frame_indexes_vision=item_condition_frames,
                        input_timestep=input_timestep,
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

            if sequence_plan.has_action:
                input_action_tokens = gen_data_clean.x0_tokens_action[idx_action]

                action_fps: float | None = None
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps = float(gen_data_clean.fps_action[idx_action].item())

                idx_action += 1

                action_split_len = _pack_action_tokens(
                    packed_seq=packed_seq,
                    input_action_tokens=input_action_tokens,
                    condition_frame_indexes_action=sequence_plan.condition_frame_indexes_action,
                    input_timestep=input_timestep,
                    curr_rope_id=curr_rope_id,
                    action_temporal_offset=vision_start_temporal_offset,
                    enable_fps_modulation=enable_fps_modulation,
                    base_fps=base_fps,
                    action_fps=action_fps,
                    base_temporal_compression_factor=temporal_compression_factor,
                )
                sample_len += action_split_len
            else:
                action_split_len = 0

        if sequence_plan.has_sound:
            input_sound_tokens = gen_data_clean.x0_tokens_sound[idx_sound]

            sound_fps: float | None = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_sound is not None
                and idx_sound < len(gen_data_clean.fps_sound)
            ):
                sound_fps = float(gen_data_clean.fps_sound[idx_sound].item())

            idx_sound += 1

            sound_split_len = _pack_sound_tokens(
                packed_seq=packed_seq,
                input_sound_tokens=input_sound_tokens,
                condition_frame_indexes_sound=sequence_plan.condition_frame_indexes_sound,
                input_timestep=input_timestep,
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
        has_any_generation = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
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

        combined_split_len = vision_split_len + action_split_len + sound_split_len + eov_len
        packed_seq.attn_modes.append("full")
        packed_seq.split_lens.append(combined_split_len)
        packed_seq.sample_lens.append(sample_len)

    if null_action_flags:
        assert len(set(null_action_flags)) == 1, (
            f"Inconsistent null_action_supertokens across samples: {null_action_flags}."
        )
        packed_seq.null_action_supertokens = null_action_flags[0]

    return packed_seq.finalize(gen_data_clean=gen_data_clean)


def build_sequence_plans_from_data_batch(
    data_batch: dict,
    input_video_key,
    input_image_key: str,
) -> list[SequencePlan]:
    """Build or retrieve sequence plans from a data batch dictionary.

    This function extracts sequence plans from the data batch if they exist,
    otherwise creates default SequencePlan objects for each sample in the batch.

    Args:
        data_batch: Dictionary containing the data batch from the dataloader.
        input_video_key: Key for video tensors in the batch.
        input_image_key: Key for image tensors in the batch.

    Returns:
        List of SequencePlan objects, one per sample in the batch.
    """
    # NOTE: this function is ONLY intended for backward compatibility.
    # For new modalities, please generate the sequence_plan in the dataset class.

    if "sequence_plan" in data_batch:
        return data_batch["sequence_plan"]

    assert "action" not in data_batch or data_batch["action"] is None, "Action data SHOULD have sequence_plans!"
    assert "sound" not in data_batch or data_batch["sound"] is None, "Sound data SHOULD have sequence_plans!"

    batch_size = 0
    for key in [input_video_key, input_image_key]:
        if key in data_batch:
            val = data_batch[key]
            if isinstance(val, torch.Tensor):
                batch_size = val.shape[0]
                break
            elif isinstance(val, list):
                batch_size = len(val)
                break

    if batch_size == 0:
        raise ValueError(
            f"Cannot determine batch size from data_batch. Expected {input_video_key}, {input_image_key}, or similar key."
        )

    return [
        SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=[],
        )
        for _ in range(batch_size)
    ]
