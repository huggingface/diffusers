# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Diffusers integration additions are licensed under the Apache License, Version 2.0.
#
# This file carries the third-party notices for the whole RFD4-Proteina integration; the
# autoencoder, scheduler and pipeline point here rather than repeating them.
#
# Adapted from RFD4-Proteina, revision 0e05aece83eafc4bf06dad7f9a7e5d50922ec83f. The model and
# sampling code is BSD-3-Clause. Through it, these components are also adapted:
#   * pair-biased attention (`RFD4ProteinaPairAttention`): MIT, Copyright (c) 2022 MattMcPartlon
#   * adaptive normalization and SwiGLU (`RFD4ProteinaAdaptiveNorm`, the swish-gated linear in
#     `RFD4ProteinaTransition`): MIT, Copyright (c) 2024 Phil Wang, from alphafold3-pytorch,
#     https://github.com/lucidrains/alphafold3-pytorch
#   * pair transition and triangular multiplicative update: Apache-2.0,
#     Copyright 2021 AlQuraishi Laboratory, Copyright 2021 DeepMind Technologies Limited.
#     Apache-2.0 is the license this repository ships in LICENSE.
# These were adapted to Diffusers model interfaces, attention processors and checkpoint layouts;
# the original implementations are not imported. AtomWorks remains a separate dependency under its
# own license, and model weights have their own distribution terms. The notices below preserve the
# upstream terms and do not relicense the upstream material.
#
# ---------------------------------------------------------------------------------------------
# RFD4-Proteina
# BSD 3-Clause License
#
# Copyright (c) 2026, Institute for Protein Design, University of Washington
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions
#   and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this list of
#   conditions and the following disclaimer in the documentation and/or other materials provided
#   with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ---------------------------------------------------------------------------------------------
# Pair-biased attention
# MIT License
#
# Copyright (c) 2022 MattMcPartlon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ---------------------------------------------------------------------------------------------
# Adaptive normalization and SwiGLU
# MIT License
#
# Copyright (c) 2024 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""RFD4-Proteina denoiser operating on prepared molecular tensors."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import BaseOutput, apply_lora_scale
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin, get_parameter_dtype


# Annotation keys of the prepared conditioning tensors, as written by the RFD4 input preparation.
CONDITION_KEYS = {
    "sequence": "condition_sequence_1_residue",
    "coordinates": "condition_coordinate_1_atom",
    "internal_coordinates": "condition_InternalCoordinate_1_atom",
    "coordinate_group": "condition_CoordinateGroup_1_atom",
    "rasa": "condition_rasa-binned_1_atom",
    "secondary_structure": "condition_secondary-structure_1_residue",
    "hbond": "condition_hbond_1_atom",
    "pistack": "condition_pistack_1_atom",
    "pication": "condition_pication_1_atom",
    "saltbridge": "condition_saltbridge_1_atom",
    "hydrophobic": "condition_hydrophobic_1_atom",
    "metal_coord": "condition_metal-coord_1_atom",
    "hotspot": "condition_hotspot_1_atom",
    "index_group": "condition_index-group_1_token",
    "cath": "condition_cath-label_1_residue",
    "centroid": "condition_ChainCentroid_1_atom",
}

SEQUENCE_PROBABILITY_TOKENS = (
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL A C G U DA DC DG DT".split()
)

NORMS = {"layer_norm": nn.LayerNorm, "rms_norm": nn.RMSNorm}


def prepare_coord_group_seq(
    features: Mapping[str, Any], generators: Sequence[torch.Generator] | None = None
) -> torch.Tensor | None:
    """Randomly relabel the coordinate groups of each design.

    Draws from one generator per batch row when given, and from the global stream otherwise.
    """
    values = features.get(CONDITION_KEYS["coordinate_group"])
    if values is None:
        return None
    values = values.long().clone()
    rows = list(generators) if generators is not None else [None] * values.shape[0]
    for row, generator in enumerate(rows):
        present = values[row] != -1
        unique, inverse = torch.unique(values[row][present], return_inverse=True)
        device = generator.device if generator is not None else values.device
        permutation = torch.randperm(len(unique), device=device, generator=generator)
        values[row][present] = permutation.to(values.device)[inverse]
    return values.float()


def get_rfd4_time_embedding(values: torch.Tensor, width: int, max_positions: int = 2000) -> torch.Tensor:
    """Sinusoidal embedding of times in [0, 1], in the original model's operation order.

    This is the same function as [`~models.embeddings.get_timestep_embedding`] called with `flip_sin_to_cos=False`,
    `downscale_freq_shift=1` and `max_period=max_positions`, and differs from it only in float32 rounding: this scales
    the time by `max_positions` before multiplying by the frequencies, where the shared helper scales afterwards.
    Because the sine arguments reach `max_positions` radians, that reordering moves the embedding by up to 1.2e-4,
    about a thousand times float32 eps.

    Over the released 60-step recipe that compounds to 0.17 angstrom of maximum coordinate deviation (0.004 angstrom
    mean) with identical sequence output, so it is small but not noise, and it would cost the exact parity this port
    holds against the original runtime. Kept deliberately.
    """
    values = values.float() * max_positions
    half = width // 2
    frequencies = torch.exp(
        torch.arange(half, dtype=torch.float32, device=values.device) * (-math.log(max_positions) / (half - 1))
    )
    result = torch.cat((torch.sin(values[:, None] * frequencies), torch.cos(values[:, None] * frequencies)), dim=-1)
    return F.pad(result, (0, width % 2))


def align_chirality(
    chirality_vectors: torch.Tensor,
    features: Mapping[str, Any],
) -> torch.Tensor:
    """Orient packed stereochemical vectors towards the current noisy backbone.

    The released checkpoint sees the chirality feature rotated onto the noisy neighbour directions rather than in its
    canonical frame, so the two are different inputs to the same weights. Needs `chiral_frame_atoms` `(batch, tokens,
    slots, 2 + K)` and `chiral_slot_owner` `(batch, tokens, slots)` alongside the `chirality` feature.

    Args:
        chirality_vectors (`torch.Tensor`): Packed unit vectors `(batch, tokens, 3 * slots)`.
        features (`Mapping`): Prepared tensors, including the noisy state under `x_t`.

    Returns:
        `torch.Tensor`: Aligned vectors; slots without a frame keep their input value.
    """
    missing = sorted({"chiral_frame_atoms", "chiral_slot_owner"} - features.keys())
    if missing:
        raise KeyError(
            f"The chirality feature needs {missing} in the prepared features. The released checkpoint was trained "
            "with chirality vectors oriented towards the noisy backbone; without these tensors the model would "
            "silently receive a different conditioning signal."
        )
    batch_size, num_tokens, width = chirality_vectors.shape
    num_slots = width // 3
    anchors = features["x_t"]["bb_ca"].float()
    valid = features["padding_mask"].bool()
    frames = features["chiral_frame_atoms"]
    rows, tokens, slots = torch.where((frames[..., 0] >= 0) & valid[..., None])
    if rows.numel() == 0:
        return chirality_vectors * valid[..., None]
    origins, neighbors = frames[rows, tokens, slots, :2], frames[rows, tokens, slots, 2:]
    present = neighbors >= 0
    neighbors = neighbors.clamp_min(0)
    packed = chirality_vectors.reshape(batch_size, num_tokens, num_slots, 3)
    supplied = packed[rows[:, None], neighbors, slots[:, None]].float()
    origin = anchors[rows[:, None], origins].mean(-2, keepdim=True)
    noisy = anchors[rows[:, None], neighbors] - origin
    with torch.autocast(device_type=chirality_vectors.device.type, enabled=False):
        # Owner ids map every slot, including copies, onto the frame that defines it; zero is "no frame".
        frame_index = torch.zeros(batch_size, num_tokens * num_slots, device=packed.device, dtype=torch.long)
        frame_index[rows, tokens * num_slots + slots] = torch.arange(1, len(rows) + 1, device=packed.device)
        owners = features["chiral_slot_owner"]
        slot_frames = frame_index[torch.arange(batch_size, device=packed.device)[:, None, None], owners.clamp_min(0)]
        supplied = torch.where(present[..., None], supplied, 0.0)
        noisy = torch.where(present[..., None], noisy, 0.0)
        noisy = noisy / torch.linalg.vector_norm(noisy, dim=-1, keepdim=True).clamp_min(1e-8)
        # Fit proper rotations by SVD of the covariance, applied as `source @ rotation`. Where the fit is not unique the
        # arbitrary optimum SVD returns is used, matching the reference implementation.
        u, singular, vh = torch.linalg.svd((supplied.transpose(-1, -2) @ noisy).float())
        signs = torch.ones_like(singular)
        signs[..., -1] = torch.linalg.det(u @ vh).sign()
        rotation = (u * signs[..., None, :]) @ vh
        rotations = torch.cat([torch.eye(3, device=packed.device)[None], rotation])
        aligned = torch.einsum("bnsx,bnsxy->bnsy", packed.float(), rotations[slot_frames])
        aligned = torch.where((owners >= 0)[..., None], aligned, packed)
    return aligned.reshape_as(chirality_vectors) * valid[..., None]


def feature_dimensions(
    t_emb_dim: int,
    latent_dim: int,
    token_type_emb_dim: int,
    recycle_conditioning_dim: int,
    cath_emb_dim: int,
    seq_sep_dim: int,
    xt_pair_dist_dim: int,
) -> dict[str, int]:
    """Width of every denoiser input feature. Several are set by the checkpoint configuration, so the table is
    built per model."""
    return {
        "time_emb_bb_ca": t_emb_dim,
        "time_emb_local_latents": t_emb_dim,
        "recycle_remaining": recycle_conditioning_dim,
        "token_type": token_type_emb_dim,
        "xt_bb_ca": 3,
        "xt_local_latents": latent_dim,
        "rasa_binned": 37,
        "chain_centroid_xyz": 3,
        "secondary_structure": 16,
        "hbond_modernized": 74,
        "pistack": 37,
        "pication": 74,
        "saltbridge": 37,
        "hydrophobic": 37,
        "metal_coord": 37,
        "hotspot": 37,
        "sequence_probabilities": 56,
        "na_ss_state": 2,
        "coord_group_seq": 37,
        "atom_charge": 16,
        "atomic_element": 16,
        "chirality": 18,
        "cond_a37coors_nm": 148,
        "cond_a37coors_internal_nm": 148,
        "token_bond_path_length": seq_sep_dim,
        "xt_bb_ca_pair_dists": xt_pair_dist_dim,
        "molecule_idx_pair": 1,
        "molecule_entity_pair": 1,
        "index_group_pair": 1,
        "bond_mask": 1,
        "bond_order": 5,
        "is_aromatic": 1,
        "na_ss_pair": 2,
        "canonical_polymer_ordering": 1,
        "cath_code": cath_emb_dim * 3,
    }


def _zeros(features: Mapping[str, Any], dim: int, pair: bool = False) -> torch.Tensor:
    mask = features["padding_mask"]
    shape = (*mask.shape, mask.shape[-1], dim) if pair else (*mask.shape, dim)
    return torch.zeros(shape, device=mask.device)


def _time_embedding_bb_ca(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return get_rfd4_time_embedding(features["t"]["bb_ca"], dim)


def _time_embedding_local_latents(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return get_rfd4_time_embedding(features["t"]["local_latents"], dim)


def _recycle_remaining(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    # One-hot over the passes still to come, saturating at the trained width. All-zero means the conditioning is
    # absent, which is distinct from "no passes remain".
    mask = features["padding_mask"]
    remaining = features.get("recycle_remaining")
    if remaining is None:
        return torch.zeros((mask.shape[0], dim), device=mask.device)
    index = torch.full((mask.shape[0],), min(int(remaining), dim - 1), device=mask.device)
    return F.one_hot(index.long(), dim).float()


def _xt_bb_ca(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return features["x_t"]["bb_ca"]


def _xt_local_latents(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return features["x_t"]["local_latents"]


def _cond_a37coors_nm(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    values = features.get(CONDITION_KEYS["coordinates"])
    if values is None:
        return _zeros(features, dim)
    mask = torch.isfinite(values).any(dim=-1)
    coords = torch.nan_to_num(values) / 10.0 * mask[..., None]
    return torch.cat((coords.flatten(2), mask.float()), dim=-1)


def _cond_a37coors_internal_nm(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    values = features.get(CONDITION_KEYS["internal_coordinates"])
    groups = features.get(CONDITION_KEYS["coordinate_group"])
    if values is None or groups is None:
        return _zeros(features, dim)
    mask = (groups != -1) & (groups != 0)
    coords = torch.nan_to_num(values) / 10.0 * mask[..., None]
    return torch.cat((coords.flatten(2), mask.float()), dim=-1)


def _rasa_binned(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    values = features.get(CONDITION_KEYS["rasa"])
    return _zeros(features, dim) if values is None else values.float() * features["atom37_mask"].float()


def _chain_centroid_xyz(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    values = features.get(CONDITION_KEYS["centroid"])
    if values is None:
        return _zeros(features, dim)
    finite = torch.isfinite(values).all(dim=-1, keepdim=True)
    return torch.where(finite, values.float() / 10.0, torch.zeros_like(values, dtype=torch.float32))


def _atom_condition(
    features: Mapping[str, Any], dim: int, name: str, categorical: bool = False, required: bool = True
) -> torch.Tensor:
    value = features.get(CONDITION_KEYS[name])
    if value is None:
        if required:
            raise KeyError(f"required conditioning annotation for {name} is missing")
        return _zeros(features, dim)
    mask = features["atom37_mask"].float()
    if name in {"hbond", "pication"}:
        first = ((value == 1) | (value == 3)).float() * mask
        second = ((value == 2) | (value == 3)).float() * mask
        return torch.cat((first, second), dim=-1)
    return ((value != 0).float() if categorical else value.float()) * mask


def _hbond_modernized(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _atom_condition(features, dim, "hbond")


def _pistack(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _atom_condition(features, dim, "pistack")


def _pication(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _atom_condition(features, dim, "pication")


def _saltbridge(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _atom_condition(features, dim, "saltbridge", categorical=True)


def _hydrophobic(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _atom_condition(features, dim, "hydrophobic")


def _metal_coord(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _atom_condition(features, dim, "metal_coord", categorical=True)


def _hotspot(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _atom_condition(features, dim, "hotspot", required=False)


def _sequence_probabilities(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    by_token = {}
    for key, value in features.items():
        if "sequence-probability" in key and key.endswith("_1_residue"):
            by_token.setdefault(key.rsplit("-", 1)[-1].removesuffix("_1_residue"), value)
    missing = torch.full(features["padding_mask"].shape, torch.nan, device=features["padding_mask"].device)
    probabilities = torch.stack(
        [by_token.get(token, missing) for token in SEQUENCE_PROBABILITY_TOKENS], dim=-1
    ).float()
    present = ~probabilities.isnan()
    return torch.cat((probabilities.masked_fill(~present, 0), present.float()), dim=-1)


def _na_ss_state(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    if "na_ss_state" not in features:
        return _zeros(features, dim)
    return torch.stack((features["na_ss_state"].float(), features["na_ss_state_mask"].float()), dim=-1)


def _coord_group_seq(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    values = features.get("_prepared_coord_group_seq")
    return _zeros(features, dim) if values is None else values


def _chirality(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return features["chirality"].float() if "chirality" in features else _zeros(features, dim)


def _token_bond_path_length(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return F.one_hot(features["token_bond_path_length"].long().clamp(0, dim - 1), dim).float()


def _pair_match(values: torch.Tensor) -> torch.Tensor:
    return (values[:, :, None] == values[:, None, :]).float().unsqueeze(-1)


def _molecule_idx_pair(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _pair_match(features["molecule_iid"])


def _molecule_entity_pair(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _pair_match(features["molecule_entity_id"])


def _index_group_pair(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _pair_match(features.get(CONDITION_KEYS["index_group"]))


def _pair_values(features: Mapping[str, Any], name: str, dim: int) -> torch.Tensor:
    return features[name] if name in features else _zeros(features, dim, pair=True)[..., 0]


def _bond_mask(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _pair_values(features, "bond_mask", dim).float().unsqueeze(-1)


def _is_aromatic(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _pair_values(features, "is_aromatic", dim).float().unsqueeze(-1)


def _canonical_polymer_ordering(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return _pair_values(features, "canonical_polymer_ordering", dim).float().unsqueeze(-1)


def _bond_order(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return F.one_hot(_pair_values(features, "bond_order", dim).long().clamp(0, 4), 5).float()


def _na_ss_pair(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    if "na_ss_pair" not in features:
        return _zeros(features, dim, pair=True)
    return torch.stack((features["na_ss_pair"].float(), features["na_ss_pair_mask"].float()), dim=-1)


# Features without learned weights: plain functions of the prepared tensors and the feature width.
FEATURE_FUNCTIONS = {
    "time_emb_bb_ca": _time_embedding_bb_ca,
    "time_emb_local_latents": _time_embedding_local_latents,
    "recycle_remaining": _recycle_remaining,
    "xt_bb_ca": _xt_bb_ca,
    "xt_local_latents": _xt_local_latents,
    "cond_a37coors_nm": _cond_a37coors_nm,
    "cond_a37coors_internal_nm": _cond_a37coors_internal_nm,
    "rasa_binned": _rasa_binned,
    "chain_centroid_xyz": _chain_centroid_xyz,
    "hbond_modernized": _hbond_modernized,
    "pistack": _pistack,
    "pication": _pication,
    "saltbridge": _saltbridge,
    "hydrophobic": _hydrophobic,
    "metal_coord": _metal_coord,
    "hotspot": _hotspot,
    "sequence_probabilities": _sequence_probabilities,
    "na_ss_state": _na_ss_state,
    "coord_group_seq": _coord_group_seq,
    "chirality": _chirality,
    "token_bond_path_length": _token_bond_path_length,
    "molecule_idx_pair": _molecule_idx_pair,
    "molecule_entity_pair": _molecule_entity_pair,
    "index_group_pair": _index_group_pair,
    "bond_mask": _bond_mask,
    "is_aromatic": _is_aromatic,
    "canonical_polymer_ordering": _canonical_polymer_ordering,
    "bond_order": _bond_order,
    "na_ss_pair": _na_ss_pair,
}


class _TokenTypeEmbedding(nn.Embedding):
    def __init__(self, dim: int, num_token_types: int, **kwargs) -> None:
        super().__init__(num_token_types, dim)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        return super().forward(features["token_type"].long())


class _SecondaryStructureEmbedding(nn.Embedding):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(4, dim, padding_idx=0)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        values = features.get(CONDITION_KEYS["secondary_structure"])
        return _zeros(features, self.embedding_dim) if values is None else super().forward(values.long())


class _AtomChargeEmbedding(nn.Embedding):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(11, dim)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        if "charge" not in features:
            return _zeros(features, self.embedding_dim)
        values = (features["charge"][:, :, 1].long() + 4).clamp(0, 10)
        return super().forward(values) * (features["token_type"] == 3)[..., None]


class _AtomicElementEmbedding(nn.Embedding):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(128, dim)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        if "element" not in features:
            return _zeros(features, self.embedding_dim)
        values = features["element"][:, :, 1].long().clamp(0, self.num_embeddings - 1)
        return super().forward(values) * (features["token_type"] == 3)[..., None]


class _CathEmbedding(nn.Module):
    def __init__(self, dim: int, cath_emb_dim: int, **kwargs) -> None:
        super().__init__()
        self.dim = dim
        self.c = nn.Embedding(6, cath_emb_dim)
        self.a = nn.Embedding(44, cath_emb_dim)
        self.t = nn.Embedding(1473, cath_emb_dim)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        values = features.get(CONDITION_KEYS["cath"])
        if values is None:
            return _zeros(features, self.dim)
        values = values.long() + 1
        return torch.cat((self.c(values[..., 0]), self.a(values[..., 1]), self.t(values[..., 2])), dim=-1)


class _PairDistances(nn.Module):
    """One-hot bins of the pairwise distances of the noisy backbone. It has no weights; it needs the bin range."""

    def __init__(self, dim: int, xt_pair_dist_min: float, xt_pair_dist_max: float, **kwargs) -> None:
        super().__init__()
        self.dim, self.limits = dim, (xt_pair_dist_min, xt_pair_dist_max)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        coords = features["x_t"]["bb_ca"]
        distances = torch.linalg.norm(coords[:, :, None] - coords[:, None, :], dim=-1)
        limits = torch.linspace(*self.limits, self.dim - 1, device=coords.device)
        return F.one_hot(torch.bucketize(distances, limits), self.dim).float()


# The features that need a module: the ones with learned weights, and the one with a setting of its own.
FEATURE_MODULES = {
    "token_type": _TokenTypeEmbedding,
    "secondary_structure": _SecondaryStructureEmbedding,
    "atom_charge": _AtomChargeEmbedding,
    "atomic_element": _AtomicElementEmbedding,
    "cath_code": _CathEmbedding,
    "xt_bb_ca_pair_dists": _PairDistances,
}


class RFD4ProteinaFeatureFactory(nn.Module):
    """Concatenates named features and projects them to `width`, computing the projection in the model dtype.

    Each configured feature is resolved once, to a function (`FEATURE_FUNCTIONS`) or to a module (`FEATURE_MODULES`,
    kept in `self.feature_modules`).
    """

    def __init__(
        self,
        names: Sequence[str] | None,
        width: int,
        mode: Literal["seq", "pair"],
        t_emb_dim: int,
        latent_dim: int,
        token_type_emb_dim: int,
        num_token_types: int,
        recycle_conditioning_dim: int,
        cath_emb_dim: int,
        seq_sep_dim: int,
        xt_pair_dist_dim: int,
        xt_pair_dist_min: float,
        xt_pair_dist_max: float,
        normalize: bool = False,
        compact: bool = False,
    ) -> None:
        super().__init__()
        self.mode, self.width = mode, width
        self.compact = compact
        self.names = list(names or ())
        dimensions = feature_dimensions(
            t_emb_dim,
            latent_dim,
            token_type_emb_dim,
            recycle_conditioning_dim,
            cath_emb_dim,
            seq_sep_dim,
            xt_pair_dist_dim,
        )
        self.feature_modules = nn.ModuleDict()
        self.feature_fns = []
        for name in self.names:
            if name not in dimensions:
                raise ValueError(f"unsupported RFD4 feature: {name}")
            if name in FEATURE_FUNCTIONS:
                self.feature_fns.append(partial(FEATURE_FUNCTIONS[name], dim=dimensions[name]))
            else:
                module = FEATURE_MODULES[name](
                    dimensions[name],
                    num_token_types=num_token_types,
                    cath_emb_dim=cath_emb_dim,
                    xt_pair_dist_min=xt_pair_dist_min,
                    xt_pair_dist_max=xt_pair_dist_max,
                )
                self.feature_modules[name] = module
                self.feature_fns.append(module)
        if self.names:
            self.ln_out = nn.RMSNorm(width, eps=1e-5) if normalize else nn.Identity()
            self.linear_out = nn.Linear(sum(dimensions[name] for name in self.names), width, bias=False)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        mask = features["padding_mask"].bool()
        if not self.names:
            shape = (*mask.shape, mask.shape[-1], self.width) if self.mode == "pair" else (*mask.shape, self.width)
            return torch.zeros(shape, device=mask.device, dtype=features["compute_dtype"])
        spatial = (mask.shape[-1],) if self.mode == "seq" else (mask.shape[-1], mask.shape[-1])
        tensors = [feature(features) for feature in self.feature_fns]
        if self.compact and all(value.ndim == 2 for value in tensors):
            active = mask.any(dim=-1)
        else:
            tensors = [
                value.reshape(value.shape[0], *((1,) * len(spatial)), -1).expand(-1, *spatial, -1)
                if value.ndim == 2
                else value
                for value in tensors
            ]
            active = mask if self.mode == "seq" else mask[:, :, None] & mask[:, None, :]
        stacked = torch.cat(tensors, dim=-1) * active[..., None]
        compute_dtype = features["compute_dtype"]
        # The projection may be kept in float32 while the model runs in lower precision. A quantized weight is an integer
        # tensor, so its input follows the compute dtype instead.
        weight_dtype = get_parameter_dtype(self.linear_out)
        projected = self.linear_out(stacked.to(weight_dtype if weight_dtype.is_floating_point else compute_dtype))
        return (self.ln_out(projected) * active[..., None]).to(compute_dtype)


class RFD4ProteinaTransition(nn.Module):
    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        self.swish_linear = nn.Sequential(nn.Linear(width, 2 * width * expansion, bias=False))
        self.linear_out = nn.Linear(width * expansion, width, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        left, gate = self.swish_linear(values).chunk(2, dim=-1)
        return self.linear_out(F.silu(gate) * left) * mask[..., None]


class RFD4ProteinaAdaptiveNorm(nn.Module):
    def __init__(self, width: int, condition_width: int, norm_type: str) -> None:
        super().__init__()
        self.norm = NORMS[norm_type](width, eps=1e-5, elementwise_affine=False)
        self.norm_cond = NORMS[norm_type](condition_width, eps=1e-5)
        self.to_gamma = nn.Sequential(nn.Linear(condition_width, width), nn.Sigmoid())
        self.to_beta = nn.Linear(condition_width, width, bias=False)

    def forward(self, values: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values)
        normalized_condition = self.norm_cond(condition)
        gamma, beta = self.to_gamma(normalized_condition), self.to_beta(normalized_condition)
        if condition.ndim == 2 and values.ndim > 2:
            shape = (condition.shape[0], *((1,) * (values.ndim - 2)), values.shape[-1])
            gamma, beta = gamma.reshape(shape), beta.reshape(shape)
        return (normalized * gamma + beta) * mask[..., None]


class RFD4ProteinaPairAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "RFD4ProteinaPairAttention",
        hidden_states: torch.Tensor,
        pairs: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, width = hidden_states.shape
        hidden_states = attn.node_norm(hidden_states)
        query, key = attn.to_q(hidden_states), attn.to_k(hidden_states)
        query, key = attn.norm_q(query), attn.norm_k(key)
        value, gate = attn.to_v(hidden_states), attn.to_g(hidden_states)
        query, key, value, gate = (
            tensor.view(batch, tokens, attn.heads, attn.head_width) for tensor in (query, key, value, gate)
        )
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        if attn.use_pair:
            bias = attn.to_bias(attn.pair_norm(pairs)).permute(0, 3, 1, 2).masked_fill(~pair_mask, -1e4)
        else:
            bias = torch.zeros(batch, attn.heads, tokens, tokens, device=query.device, dtype=query.dtype)
            bias = bias.masked_fill(~pair_mask, -1e4)
        output = dispatch_attention_fn(
            query, key, value, attn_mask=bias, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        output = (output * gate.sigmoid()).reshape(batch, tokens, width)
        return attn.to_out(output)


class RFD4ProteinaPairAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = RFD4ProteinaPairAttnProcessor
    _available_processors = [RFD4ProteinaPairAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, width: int, pair_width: int, heads: int, norm_type: str, use_pair: bool) -> None:
        super().__init__()
        self.heads, self.head_width, self.use_pair = heads, width // heads, use_pair
        self.node_norm = NORMS[norm_type](width, eps=1e-5)
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_g = nn.Linear(width, width)
        self.norm_q = NORMS[norm_type](width, eps=1e-5)
        self.norm_k = NORMS[norm_type](width, eps=1e-5)
        self.to_out = nn.Linear(width, width)
        if use_pair:
            self.to_bias = nn.Linear(pair_width, heads, bias=False)
            self.pair_norm = NORMS[norm_type](pair_width, eps=1e-5)
        self.set_processor(RFD4ProteinaPairAttnProcessor())

    def forward(self, values: torch.Tensor, pairs: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
        return self.processor(self, values, pairs, mask)


class RFD4ProteinaAdaptiveAttention(nn.Module):
    def __init__(
        self, width: int, pair_width: int, heads: int, condition_width: int, norm_type: str, use_pair: bool
    ) -> None:
        super().__init__()
        self.adaln = RFD4ProteinaAdaptiveNorm(width, condition_width, norm_type)
        self.mha = RFD4ProteinaPairAttention(width, pair_width, heads, norm_type, use_pair)
        self.scale_output = nn.Sequential(nn.Linear(condition_width, width), nn.Sigmoid())

    def forward(
        self, values: torch.Tensor, pairs: torch.Tensor | None, condition: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.mha(self.adaln(values, condition, mask), pairs, mask)
        return output * self.scale_output(condition) * mask[..., None]


class RFD4ProteinaAdaptiveTransition(nn.Module):
    def __init__(self, width: int, condition_width: int, norm_type: str) -> None:
        super().__init__()
        self.adaln = RFD4ProteinaAdaptiveNorm(width, condition_width, norm_type)
        self.transition = RFD4ProteinaTransition(width)
        self.scale_output = nn.Sequential(nn.Linear(condition_width, width), nn.Sigmoid())

    def forward(self, values: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.transition(self.adaln(values, condition, mask), mask)
        return output * self.scale_output(condition) * mask[..., None]


class RFD4ProteinaTrunkLayer(nn.Module):
    """One attention + transition block, conditioned on `condition` through adaptive normalization."""

    def __init__(
        self, token_dim: int, pair_repr_dim: int, nheads: int, dim_cond: int, norm_type: str, use_pair: bool
    ) -> None:
        super().__init__()
        self.mhba = RFD4ProteinaAdaptiveAttention(token_dim, pair_repr_dim, nheads, dim_cond, norm_type, use_pair)
        self.transition = RFD4ProteinaAdaptiveTransition(token_dim, dim_cond, norm_type)

    def forward(
        self, values: torch.Tensor, pairs: torch.Tensor | None, condition: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        values = values * mask[..., None]
        values = (values + self.mhba(values, pairs, condition, mask)) * mask[..., None]
        return (values + self.transition(values, condition, mask)) * mask[..., None]


class RFD4ProteinaTriangleMultiplication(nn.Module):
    def __init__(self, width: int, hidden: int, outgoing: bool) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.linear_a_p = nn.Linear(width, hidden, bias=False)
        self.linear_a_g = nn.Linear(width, hidden, bias=False)
        self.linear_b_p = nn.Linear(width, hidden, bias=False)
        self.linear_b_g = nn.Linear(width, hidden, bias=False)
        self.linear_g = nn.Linear(width, width, bias=False)
        self.linear_z = nn.Linear(hidden, width, bias=False)
        self.layer_norm_in = nn.LayerNorm(width, eps=1e-5, bias=False)
        self.layer_norm_out = nn.LayerNorm(hidden, eps=1e-5, bias=False)

    def forward(self, pairs: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.layer_norm_in(pairs)
        left = self.linear_a_p(normalized) * self.linear_a_g(normalized).sigmoid() * pair_mask[..., None]
        right = self.linear_b_p(normalized) * self.linear_b_g(normalized).sigmoid() * pair_mask[..., None]
        if self.outgoing:
            output = torch.matmul(left.permute(0, 3, 1, 2), right.permute(0, 3, 2, 1))
        else:
            output = torch.matmul(left.permute(0, 3, 2, 1), right.permute(0, 3, 1, 2))
        output = self.linear_z(self.layer_norm_out(output.permute(0, 2, 3, 1)))
        return output * self.linear_g(normalized).sigmoid()


class RFD4ProteinaPairTransition(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(width, eps=1e-5)
        self.linear_1 = nn.Linear(width, 2 * width)
        self.linear_2 = nn.Linear(2 * width, width)

    def forward(self, pairs: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.relu(self.linear_1(self.layer_norm(pairs)))) * pair_mask[..., None]


class RFD4ProteinaPairUpdate(nn.Module):
    def __init__(self, token_dim: int, pair_repr_dim: int, tri_mult_c: int) -> None:
        super().__init__()
        self.layer_norm_in = nn.LayerNorm(token_dim)
        self.linear_x = nn.Linear(token_dim, 2 * pair_repr_dim, bias=False)
        self.tri_mult_out = RFD4ProteinaTriangleMultiplication(pair_repr_dim, tri_mult_c, True)
        self.tri_mult_in = RFD4ProteinaTriangleMultiplication(pair_repr_dim, tri_mult_c, False)
        self.transition_out = RFD4ProteinaPairTransition(pair_repr_dim)

    def forward(self, values: torch.Tensor, pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pair_mask = mask[:, :, None] & mask[:, None, :]
        left, right = self.linear_x(self.layer_norm_in(values * mask[..., None])).chunk(2, dim=-1)
        pairs = (pairs + left[:, None] * right[:, :, None]) * pair_mask[..., None]
        pairs = (pairs + self.tri_mult_out(pairs, pair_mask)) * pair_mask[..., None]
        pairs = (pairs + self.tri_mult_in(pairs, pair_mask)) * pair_mask[..., None]
        return (pairs + self.transition_out(pairs, pair_mask)) * pair_mask[..., None]


class RFD4ProteinaPairBuilder(nn.Module):
    def __init__(
        self,
        feats_pair_repr: Sequence[str],
        feats_pair_cond: Sequence[str],
        pair_repr_dim: int,
        dim_cond: int,
        feature_kwargs: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.init_repr_factory = RFD4ProteinaFeatureFactory(
            feats_pair_repr, pair_repr_dim, "pair", **feature_kwargs, normalize=True
        )
        self.cond_factory = None
        self.adaln = None
        if feats_pair_cond:
            self.cond_factory = RFD4ProteinaFeatureFactory(
                feats_pair_cond, dim_cond, "pair", **feature_kwargs, normalize=True, compact=True
            )
            self.adaln = RFD4ProteinaAdaptiveNorm(pair_repr_dim, dim_cond, "rms_norm")

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        pairs = self.init_repr_factory(features)
        if self.cond_factory is not None:
            mask = features["padding_mask"].bool()
            pairs = self.adaln(pairs, self.cond_factory(features), mask[:, :, None] & mask[:, None, :])
        return pairs


@dataclass
class RFD4ProteinaTransformerOutput(BaseOutput):
    """Denoiser predictions before scheduler integration.

    Args:
        backbone (`torch.Tensor`): Backbone velocity shaped `(batch, tokens, 3)`, in nanometers per unit time.
        local_latents (`torch.Tensor`): Latent velocity shaped `(batch, tokens, latent_dim)`.
        sequence_logits (`torch.Tensor`, *optional*): Residue logits shaped `(batch, tokens, 147)`.
        insertion_logits (`torch.Tensor`, *optional*): Insertion assignment logits shaped `(batch, tokens, tokens)`.
    """

    backbone: torch.Tensor
    local_latents: torch.Tensor | None = None
    sequence_logits: torch.Tensor | None = None
    insertion_logits: torch.Tensor | None = None


class RFD4ProteinaTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    """RFD4 velocity denoiser with paired backbone and local-latent modalities.

    The model runs in its own dtype: load it with `dtype=torch.bfloat16` for reduced-precision inference. The sampling
    state passed to `forward` stays in float32; feature embeddings are computed in float32 and cast at the input
    projection.

    Args:
        nlayers (`int`, defaults to `14`): Number of trunk layers.
        token_dim (`int`, defaults to `768`): Token representation width.
        nheads (`int`, defaults to `12`): Attention heads per layer.
        dim_cond (`int`, defaults to `256`): Width of the conditioning vector (time, recycle index, CATH label).
        pair_repr_dim (`int`, defaults to `384`): Pair representation width.
        latent_dim (`int`, defaults to `32`): Width of the per-token local latent.
        token_type_emb_dim (`int`, defaults to `32`): Token-type embedding width.
        cath_emb_dim (`int`, defaults to `32`): Width of each CATH class/architecture/topology embedding.
        t_emb_dim (`int`, defaults to `256`): Sinusoidal time embedding width.
        seq_sep_dim (`int`, defaults to `64`): Number of bond-path-length bins.
        xt_pair_dist_dim (`int`, defaults to `30`): Number of backbone pair-distance bins.
        xt_pair_dist_min (`float`, defaults to `0.1`), xt_pair_dist_max (`float`, defaults to `3.0`):
            Pair-distance range in nanometers.
        feats_seq, feats_cond_seq, feats_pair_repr, feats_pair_cond (`list[str]`):
            Names of the features projected into the token, conditioning, pair and pair-conditioning inputs.
        update_pair_repr_every_n (`int`, defaults to `2`):
            How often the pair representation is updated from the tokens between layers.
        tri_mult_c (`int`, defaults to `256`): Hidden width of the triangle multiplication in those updates.
        num_token_types (`int`, defaults to `7`): Size of the token-type vocabulary: protein, RNA, DNA, atomized,
            unknown, glycan and non-canonical amino acid.
        recycle_conditioning_dim (`int`, defaults to `1`): Width of the `recycle_remaining` one-hot, when that
            feature is in `feats_cond_seq`.
        num_sequence_classes (`int`, defaults to `147`): Size of the residue vocabulary the sequence readout predicts.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["RFD4ProteinaTrunkLayer"]
    _repeated_blocks = ["RFD4ProteinaTrunkLayer", "RFD4ProteinaPairUpdate"]
    _skip_layerwise_casting_patterns = ["norm", "ln_out", "embedding"]
    # Conditioning and normalization stay in full precision: the sampler integrates sixty steps and
    # amplifies rounding carried in the time and recycle embeddings.
    _keep_in_fp32_modules = ["cond_factory", "norm", "ln_out"]

    @register_to_config
    def __init__(
        self,
        nlayers: int = 14,
        token_dim: int = 768,
        nheads: int = 12,
        dim_cond: int = 256,
        pair_repr_dim: int = 384,
        latent_dim: int = 32,
        token_type_emb_dim: int = 32,
        cath_emb_dim: int = 32,
        t_emb_dim: int = 256,
        seq_sep_dim: int = 64,
        xt_pair_dist_dim: int = 30,
        xt_pair_dist_min: float = 0.1,
        xt_pair_dist_max: float = 3.0,
        feats_seq: tuple[str, ...] = (
            "xt_bb_ca",
            "xt_local_latents",
            "token_type",
            "sequence_probabilities",
            "cond_a37coors_nm",
            "cond_a37coors_internal_nm",
            "rasa_binned",
            "secondary_structure",
            "hbond_modernized",
            "pistack",
            "pication",
            "saltbridge",
            "hydrophobic",
            "metal_coord",
            "na_ss_state",
            "hotspot",
            "atom_charge",
            "atomic_element",
            "coord_group_seq",
            "chirality",
            "chain_centroid_xyz",
        ),
        feats_cond_seq: tuple[str, ...] = (
            "time_emb_bb_ca",
            "time_emb_local_latents",
            "cath_code",
            "recycle_remaining",
        ),
        feats_pair_repr: tuple[str, ...] = (
            "xt_bb_ca_pair_dists",
            "bond_mask",
            "bond_order",
            "is_aromatic",
            "index_group_pair",
            "molecule_idx_pair",
            "molecule_entity_pair",
            "token_bond_path_length",
            "canonical_polymer_ordering",
            "na_ss_pair",
        ),
        feats_pair_cond: tuple[str, ...] = ("time_emb_bb_ca", "time_emb_local_latents"),
        update_pair_repr_every_n: int = 2,
        tri_mult_c: int = 256,
        num_sequence_classes: int = 147,
        num_token_types: int = 7,
        recycle_conditioning_dim: int = 1,
    ) -> None:
        super().__init__()
        feature_kwargs = {
            "t_emb_dim": t_emb_dim,
            "latent_dim": latent_dim,
            "token_type_emb_dim": token_type_emb_dim,
            "num_token_types": num_token_types,
            "recycle_conditioning_dim": recycle_conditioning_dim,
            "cath_emb_dim": cath_emb_dim,
            "seq_sep_dim": seq_sep_dim,
            "xt_pair_dist_dim": xt_pair_dist_dim,
            "xt_pair_dist_min": xt_pair_dist_min,
            "xt_pair_dist_max": xt_pair_dist_max,
        }
        self.nlayers = nlayers
        self.gradient_checkpointing = False
        self.init_repr_factory = RFD4ProteinaFeatureFactory(feats_seq, token_dim, "seq", **feature_kwargs)
        self.cond_factory = RFD4ProteinaFeatureFactory(feats_cond_seq, dim_cond, "seq", **feature_kwargs)
        self.transition_c_1 = RFD4ProteinaTransition(dim_cond, 2)
        self.transition_c_2 = RFD4ProteinaTransition(dim_cond, 2)
        self.pair_repr_builder = RFD4ProteinaPairBuilder(
            feats_pair_repr, feats_pair_cond, pair_repr_dim, dim_cond, feature_kwargs
        )
        self.transformer_layers = nn.ModuleList(
            RFD4ProteinaTrunkLayer(token_dim, pair_repr_dim, nheads, dim_cond, "rms_norm", True)
            for _ in range(nlayers)
        )
        # Keyed by the index of the trunk layer each update follows, so only those layers have an entry.
        self.pair_update_layers = nn.ModuleDict(
            {
                str(index): RFD4ProteinaPairUpdate(token_dim, pair_repr_dim, tri_mult_c)
                for index in range(0, nlayers - 1, update_pair_repr_every_n)
            }
        )
        # The released checkpoint conditions the trunk on the passes still to come, so the conditioning vector is rebuilt
        # for every pass instead of once.
        self.condition_on_recycle_remaining = "recycle_remaining" in feats_cond_seq
        self.s_cycle_norm = nn.RMSNorm(token_dim, eps=1e-5)
        self.s_cycle_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.z_cycle_norm = nn.RMSNorm(pair_repr_dim, eps=1e-5)
        self.z_cycle_proj = nn.Linear(pair_repr_dim, pair_repr_dim, bias=False)
        self.local_latents_linear = nn.Sequential(
            nn.RMSNorm(token_dim, eps=1e-5), nn.Linear(token_dim, latent_dim, bias=False)
        )
        self.ca_linear = nn.Sequential(nn.RMSNorm(token_dim, eps=1e-5), nn.Linear(token_dim, 3, bias=False))
        self.sequence_readout = nn.Sequential(
            nn.RMSNorm(token_dim, eps=1e-5), nn.Linear(token_dim, num_sequence_classes, bias=False)
        )
        self.insertion_q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.insertion_k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self._insertion_head_scale = token_dim**-0.5

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        bb_ca: torch.Tensor,
        local_latents: torch.Tensor,
        bb_ca_time: torch.Tensor,
        local_latents_time: torch.Tensor,
        padding_mask: torch.Tensor,
        token_type: torch.Tensor | None = None,
        atom37_mask: torch.Tensor | None = None,
        conditioning: dict[str, Any] | None = None,
        recycle_depth: int = 0,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> RFD4ProteinaTransformerOutput | tuple[torch.Tensor]:
        """Predict velocities from prepared molecular tensors without advancing the sampler.

        Args:
            bb_ca (`torch.Tensor`): Noisy backbone coordinates `(batch, tokens, 3)` in nanometers.
            local_latents (`torch.Tensor`): Noisy local latents `(batch, tokens, latent_dim)`.
            bb_ca_time (`torch.Tensor`): `(batch,)` backbone time in [0, 1].
            local_latents_time (`torch.Tensor`): `(batch,)` local-latent time in [0, 1].
            padding_mask (`torch.Tensor`): `(batch, tokens)`, true where a token is real rather than padding.
            token_type (`torch.Tensor`, *optional*): `(batch, tokens)` token-type codes.
            atom37_mask (`torch.Tensor`, *optional*): `(batch, tokens, 37)`, true where an atom slot is used.
            conditioning (`dict[str, Any]`, *optional*): The remaining topology and conditioning tensors,
                under the keys the preprocessor emits. Which of them are read is fixed by the checkpoint's `feats_seq`,
                `feats_cond_seq`, `feats_pair_repr` and `feats_pair_cond`; a key a configured feature asks for but that
                is absent leaves that feature unconditioned rather than raising.
            recycle_depth (`int`, defaults to `0`): Number of trunk passes; zero still executes one pass.
            attention_kwargs (`dict`, *optional*): Only `scale` is read, as the LoRA scale for this call.
            return_dict (`bool`, defaults to `True`): Return a typed output, or a tuple of non-None fields.

        Returns:
            [`RFD4ProteinaTransformerOutput`] or `tuple`: Backbone/local-latent velocities and optional
            sequence/insertion logits, before scheduler integration.
        """
        features = {
            **(conditioning or {}),
            "padding_mask": padding_mask,
            "x_t": {"bb_ca": bb_ca, "local_latents": local_latents},
            "t": {"bb_ca": bb_ca_time, "local_latents": local_latents_time},
            "compute_dtype": self.dtype,
        }
        for name, value in (("token_type", token_type), ("atom37_mask", atom37_mask)):
            if value is not None:
                features[name] = value
        mask = padding_mask.bool()
        emitted = mask & ~features["is_in_context_conditioning"].bool()
        # The released checkpoint sees the chirality feature rotated onto the noisy neighbour frame, so the feature the
        # trunk reads is replaced before any layer runs.
        if "chirality" in features:
            features["chirality"] = align_chirality(features["chirality"].float(), features)

        cycles = max(1, recycle_depth)
        values = self.init_repr_factory(features) * mask[..., None]
        pairs = self.pair_repr_builder(features)
        previous_values, previous_pairs = torch.zeros_like(values), torch.zeros_like(pairs)
        for cycle in range(cycles):
            values_in = values + self.s_cycle_proj(self.s_cycle_norm(previous_values))
            pairs_in = (pairs + self.z_cycle_proj(self.z_cycle_norm(previous_pairs))).contiguous()
            # Before the first pass the remaining-passes feature is absent (all-zero), which is distinct from "no passes
            # remain". Later passes carry the passes still to come.
            if cycle == 0 or self.condition_on_recycle_remaining:
                condition = self.cond_factory(
                    {
                        **features,
                        "recycle_remaining": None if cycle == 0 else cycles - cycle - 1,
                    }
                )
                condition = self.transition_c_2(self.transition_c_1(condition, mask), mask)
            hidden, hidden_pairs = values_in, pairs_in
            for index, layer in enumerate(self.transformer_layers):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden = self._gradient_checkpointing_func(layer, hidden, hidden_pairs, condition, mask)
                else:
                    hidden = layer(hidden, hidden_pairs, condition, mask)
                if str(index) in self.pair_update_layers:
                    hidden_pairs = self.pair_update_layers[str(index)](hidden, hidden_pairs, mask)
            previous_values, previous_pairs = hidden, hidden_pairs
        values = previous_values

        backbone = self.ca_linear(values) * emitted[..., None]
        local_latents = (
            self.local_latents_linear(values) * (emitted & features["supports_local_latent"].bool())[..., None]
        )
        sequence_logits = self.sequence_readout(values) * mask[..., None]
        query, key = self.insertion_q_proj(values), self.insertion_k_proj(values)
        insertion_logits = torch.bmm(query, key.transpose(1, 2)) * self._insertion_head_scale
        output = RFD4ProteinaTransformerOutput(
            backbone=backbone,
            local_latents=local_latents,
            sequence_logits=sequence_logits,
            insertion_logits=insertion_logits,
        )
        return output if return_dict else output.to_tuple()
