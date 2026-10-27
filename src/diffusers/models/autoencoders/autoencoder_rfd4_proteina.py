# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Diffusers integration additions are licensed under the Apache License, Version 2.0.
#
# Adapted from RFD4-Proteina (BSD-3-Clause), with MIT and Apache-2.0 components.
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Copyright (c) 2022 MattMcPartlon
# Copyright (c) 2024 Phil Wang
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Full third-party copyright notices, license conditions and disclaimers for the RFD4-Proteina
# integration are in diffusers/models/transformers/transformer_rfd4_proteina.py.
"""RFD4-Proteina autoencoder: per-token local latents <-> atom37 coordinates and residue identities."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.accelerate_utils import apply_forward_hook
from ...utils.torch_utils import randn_tensor
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin, get_parameter_dtype


NORMS = {"layer_norm": nn.LayerNorm, "rms_norm": nn.RMSNorm}


# Copied from diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaTransition
class RFD4ProteinaTransition(nn.Module):
    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        self.swish_linear = nn.Sequential(nn.Linear(width, 2 * width * expansion, bias=False))
        self.linear_out = nn.Linear(width * expansion, width, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        left, gate = self.swish_linear(values).chunk(2, dim=-1)
        return self.linear_out(F.silu(gate) * left) * mask[..., None]


# Copied from diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaAdaptiveNorm
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


def feature_dimensions(
    token_type_emb_dim: int, latent_dim: int, n_residue_types: int, charge_emb_dim: int
) -> dict[str, int]:
    """Width of every decoder input feature. Unlike `NORMS` this cannot be a constant: four of the
    widths are set by the checkpoint configuration, so the table is built per model."""
    return {
        "token_type": token_type_emb_dim,
        "chain_idx_seq": 1,
        "x1_a37coors_nm": 148,
        "x1_a37coors_nm_rel": 148,
        "ca_coors_nm": 3,
        "z_latent_seq": latent_dim,
        "cond_residue_type": n_residue_types,
        "incident_bond_type_counts": 37 * 10,
        "atom37_element_charge_embedding": 37 * (8 + charge_emb_dim),
    }


# Copied from diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaPairAttnProcessor
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


# Copied from diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaPairAttention
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


# Copied from diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaAdaptiveAttention
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


# Copied from diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaAdaptiveTransition
class RFD4ProteinaAdaptiveTransition(nn.Module):
    def __init__(self, width: int, condition_width: int, norm_type: str) -> None:
        super().__init__()
        self.adaln = RFD4ProteinaAdaptiveNorm(width, condition_width, norm_type)
        self.transition = RFD4ProteinaTransition(width)
        self.scale_output = nn.Sequential(nn.Linear(condition_width, width), nn.Sigmoid())

    def forward(self, values: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.transition(self.adaln(values, condition, mask), mask)
        return output * self.scale_output(condition) * mask[..., None]


# Copied from diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaTrunkLayer
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


def _autoencoder_zeros(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    mask = features["padding_mask"]
    return torch.zeros((*mask.shape, dim), device=mask.device)


def _chain_idx_seq(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return features["chain_iid"].float().unsqueeze(-1)


def _ca_coors_nm(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return features["ca_coors_nm"]


def _z_latent_seq(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    return features["z_latent"]


def _incident_bond_type_counts(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    counts = features["incident_bond_type_counts"].float()
    return (counts * features["atom37_mask"].unsqueeze(-1)).flatten(start_dim=-2)


def _cond_residue_type(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    values = features.get("condition_sequence_1_residue")
    if values is None:
        return _autoencoder_zeros(features, dim)
    valid = values != 0
    return F.one_hot((values * valid).long(), dim).float() * valid[..., None]


def _x1_a37coors_nm(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    coords, mask = features["coords_nm"], features["atom37_mask"]
    coords = coords * mask[..., None]
    return torch.cat((coords.flatten(2), mask.float()), dim=-1)


def _x1_a37coors_nm_rel(features: Mapping[str, Any], dim: int) -> torch.Tensor:
    coords, mask = features["coords_nm"], features["atom37_mask"]
    coords = coords * mask[..., None]
    coords = (coords - coords[:, :, 1, None]) * mask[..., None]
    return torch.cat((coords.flatten(2), mask.float()), dim=-1)


# Features without learned weights: plain functions of the prepared tensors and the feature width.
AUTOENCODER_FEATURE_FUNCTIONS = {
    "chain_idx_seq": _chain_idx_seq,
    "ca_coors_nm": _ca_coors_nm,
    "z_latent_seq": _z_latent_seq,
    "incident_bond_type_counts": _incident_bond_type_counts,
    "cond_residue_type": _cond_residue_type,
    "x1_a37coors_nm": _x1_a37coors_nm,
    "x1_a37coors_nm_rel": _x1_a37coors_nm_rel,
}


class _TokenTypeEmbedding(nn.Embedding):
    def __init__(self, dim: int, num_token_types: int, **kwargs) -> None:
        super().__init__(num_token_types, dim)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        return super().forward(features["token_type"].long())


class _ElementChargeEmbedding(nn.Module):
    def __init__(self, dim: int, charge_emb_dim: int, **kwargs) -> None:
        super().__init__()
        self.element = nn.Embedding(119, 8)
        self.charge = nn.Embedding(11, charge_emb_dim)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        mask = features["atom37_mask"].bool()
        elements = features["ground_truth_element"].long().masked_fill(~mask, 0)
        # Formal charges are stored signed; the embedding covers -4..6 saturated.
        charges = (features["charge"].long().masked_fill(~mask, 0) + 4).clamp(0, 10)
        encoded = torch.cat((self.element(elements), self.charge(charges)), dim=-1)
        return (encoded * mask.unsqueeze(-1)).flatten(start_dim=-2)


# The features with learned weights.
AUTOENCODER_FEATURE_MODULES = {
    "token_type": _TokenTypeEmbedding,
    "atom37_element_charge_embedding": _ElementChargeEmbedding,
}


class RFD4ProteinaAutoencoderFeatureFactory(nn.Module):
    """Concatenates named features and projects them to `width`. Each feature is resolved once, to a function
    (`AUTOENCODER_FEATURE_FUNCTIONS`) or to a module with weights (`AUTOENCODER_FEATURE_MODULES`, kept in
    `self.feature_modules`)."""

    def __init__(
        self,
        names: Sequence[str],
        width: int,
        token_type_emb_dim: int,
        latent_dim: int,
        n_residue_types: int,
        num_token_types: int,
        charge_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        dimensions = feature_dimensions(token_type_emb_dim, latent_dim, n_residue_types, charge_emb_dim)
        self.feature_modules = nn.ModuleDict()
        self.feature_fns = []
        for name in names:
            if name not in dimensions:
                raise ValueError(f"unsupported RFD4 autoencoder feature: {name}")
            if name in AUTOENCODER_FEATURE_FUNCTIONS:
                self.feature_fns.append(partial(AUTOENCODER_FEATURE_FUNCTIONS[name], dim=dimensions[name]))
            else:
                module = AUTOENCODER_FEATURE_MODULES[name](
                    dimensions[name], num_token_types=num_token_types, charge_emb_dim=charge_emb_dim
                )
                self.feature_modules[name] = module
                self.feature_fns.append(module)
        self.linear_out = nn.Linear(sum(dimensions[name] for name in names), width, bias=False)

    def forward(self, features: Mapping[str, Any]) -> torch.Tensor:
        mask = features["padding_mask"].bool()
        stacked = torch.cat([feature(features) for feature in self.feature_fns], dim=-1) * mask[..., None]
        compute_dtype = features["compute_dtype"]
        # The projection may be kept in float32 while the model runs in lower precision. A quantized weight is an integer
        # tensor, so its input follows the compute dtype instead.
        weight_dtype = get_parameter_dtype(self.linear_out)
        projected = self.linear_out(stacked.to(weight_dtype if weight_dtype.is_floating_point else compute_dtype))
        return (projected * mask[..., None]).to(compute_dtype)


class RFD4ProteinaEncoder(nn.Module):
    """Maps atom37 coordinates to the mean and log-scale of the per-token local latent distribution."""

    def __init__(
        self,
        nlayers: int,
        token_dim: int,
        nheads: int,
        dim_cond: int,
        pair_repr_dim: int,
        token_type_emb_dim: int,
        latent_dim: int,
        feats_seq: Sequence[str],
        n_residue_types: int,
        num_token_types: int,
    ) -> None:
        super().__init__()
        vocabulary = (n_residue_types, num_token_types)
        self.init_repr_factory = RFD4ProteinaAutoencoderFeatureFactory(
            feats_seq, token_dim, token_type_emb_dim, latent_dim, *vocabulary
        )
        self.dim_cond = dim_cond
        self.transition_c_1 = RFD4ProteinaTransition(dim_cond, 2)
        self.transition_c_2 = RFD4ProteinaTransition(dim_cond, 2)
        self.transformer_layers = nn.ModuleList(
            RFD4ProteinaTrunkLayer(token_dim, pair_repr_dim, nheads, dim_cond, "layer_norm", False)
            for _ in range(nlayers)
        )
        self.latent_decoder_mean_n_log_scale = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, 2 * latent_dim, bias=False)
        )

    def forward(
        self, features: Mapping[str, Any], generator: torch.Generator | list[torch.Generator] | None = None
    ) -> dict[str, torch.Tensor]:
        batch_size = features["padding_mask"].shape[0]
        generators = (
            None if generator is None else ([generator] if isinstance(generator, torch.Generator) else generator)
        )
        if generators is not None and len(generators) != batch_size:
            raise ValueError(f"Provide one autoencoder generator per batch row ({batch_size}).")
        padding = features["padding_mask"].bool()
        attention = padding & ~features["is_in_context_conditioning"].bool()
        emit = attention & features["supports_local_latent"].bool()
        # The autoencoder is not conditioned: the conditioning vector is zero.
        condition = torch.zeros(
            (*padding.shape, self.dim_cond), device=padding.device, dtype=features["compute_dtype"]
        )
        condition = self.transition_c_2(self.transition_c_1(condition, padding), padding)
        values = self.init_repr_factory(features) * padding[..., None]
        for layer in self.transformer_layers:
            values = layer(values, None, condition, attention)
        # The head that decides the sampled latent is kept in float32; give it matching input.
        head_dtype = get_parameter_dtype(self.latent_decoder_mean_n_log_scale[1])
        mean, log_scale = self.latent_decoder_mean_n_log_scale(values.to(head_dtype)).chunk(2, dim=-1)
        mean, log_scale = mean * emit[..., None], log_scale * emit[..., None]
        noise = (
            torch.randn_like(log_scale)
            if generators is None
            else randn_tensor(log_scale.shape, generator=generators, device=log_scale.device, dtype=log_scale.dtype)
        )
        latent = (mean + noise * log_scale.exp()) * emit[..., None]
        return {"latent": latent, "mean": mean, "log_scale": log_scale}


class RFD4ProteinaDecoder(nn.Module):
    """Maps backbone coordinates and local latents to atom37 coordinates and residue identities."""

    def __init__(
        self,
        nlayers: int,
        token_dim: int,
        nheads: int,
        dim_cond: int,
        pair_repr_dim: int,
        token_type_emb_dim: int,
        latent_dim: int,
        n_residue_types: int,
        num_token_types: int,
        feats_seq: Sequence[str],
        atom37_mask: Sequence[Sequence[bool]],
        allowed_residue_types: Sequence[Sequence[bool]],
    ) -> None:
        super().__init__()
        self.init_repr_factory = RFD4ProteinaAutoencoderFeatureFactory(
            feats_seq, token_dim, token_type_emb_dim, latent_dim, n_residue_types, num_token_types
        )
        self.dim_cond = dim_cond
        self.transition_c_1 = RFD4ProteinaTransition(dim_cond, 2)
        self.transition_c_2 = RFD4ProteinaTransition(dim_cond, 2)
        self.transformer_layers = nn.ModuleList(
            RFD4ProteinaTrunkLayer(token_dim, pair_repr_dim, nheads, dim_cond, "layer_norm", False)
            for _ in range(nlayers)
        )
        self.logit_linear = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, n_residue_types, bias=False))
        self.struct_linear = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 37 * 3, bias=False))
        self.atom_presence_linear = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 37, bias=False))
        self.register_buffer("_atom37_mask", torch.tensor(atom37_mask, dtype=torch.bool), persistent=False)
        self.register_buffer(
            "_allowed_residue_types", torch.tensor(allowed_residue_types, dtype=torch.bool), persistent=False
        )

    def forward(self, features: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        padding = features["padding_mask"].bool()
        attention = padding & ~features["is_in_context_conditioning"].bool()
        emit = attention & features["supports_local_latent"].bool()
        # The autoencoder is not conditioned: the conditioning vector is zero.
        condition = torch.zeros(
            (*padding.shape, self.dim_cond), device=padding.device, dtype=features["compute_dtype"]
        )
        condition = self.transition_c_2(self.transition_c_1(condition, padding), padding)
        values = self.init_repr_factory(features) * padding[..., None]
        for layer in self.transformer_layers:
            values = layer(values, None, condition, attention)
        logits = self.logit_linear(values) * emit[..., None]
        if "token_type" in features:
            token_type = features["token_type"].long()
            allowed = self._allowed_residue_types.to(token_type.device)[token_type]
            logits = logits.masked_fill(~(allowed | ~emit[..., None]), torch.finfo(logits.dtype).min)
        coordinates = self.struct_linear(values).reshape(*values.shape[:2], 37, 3).float() * emit[..., None, None]
        ca = features["ca_coors_nm"]
        coordinates[..., 1, :] = 0
        coordinates = coordinates + ca[:, :, None]
        coordinates = coordinates * emit[..., None, None]
        residue_type = logits.argmax(dim=-1) * emit
        # A residue the sequence condition names is kept as given; index 0 is the mask token, which leaves the
        # decision to the decoder.
        sequence = features.get("condition_sequence_1_residue")
        if sequence is not None:
            residue_type = torch.where((sequence != 0) & emit, sequence, residue_type)
        atom_mask = self._atom37_mask.to(residue_type.device)[residue_type] & emit[..., None]
        # The decoder predicts which of the residue's slots are actually occupied, so a modified residue can drop atoms
        # its canonical form would have.
        present = self.atom_presence_linear(values) > 0
        present[..., 1] = True  # every emitted token keeps its reference atom
        # Atoms that belong to a conditioned coordinate group are required, whatever the presence head says.
        groups = features.get("condition_CoordinateGroup_1_atom")
        if groups is not None:
            present = present | (groups >= 0)
        atom_mask = atom_mask & present
        return {
            "coordinates": coordinates * atom_mask[..., None],
            "residue_type": residue_type * emit,
            "emit_mask": emit,
            "atom_mask": atom_mask,
        }


@dataclass
class RFD4ProteinaEncoderOutput(BaseOutput):
    """Sampled latent and Gaussian parameters, each shaped `(batch, tokens, latent_dim)`.

    Args:
        latent (`torch.Tensor`): Sample `mean + noise * exp(log_scale)`, zero outside emitted latent tokens.
        mean (`torch.Tensor`): Gaussian mean, zero outside emitted latent tokens.
        log_scale (`torch.Tensor`): Log standard deviation, zero outside emitted latent tokens.
    """

    latent: torch.Tensor
    mean: torch.Tensor | None = None
    log_scale: torch.Tensor | None = None


@dataclass
class RFD4ProteinaDecoderOutput(BaseOutput):
    """Decoded molecular coordinates and discrete atom37 topology.

    Args:
        coordinates (`torch.Tensor`): Coordinates `(batch, tokens, 37, 3)` in nanometers; non-emitted atoms are zero.
        residue_type (`torch.Tensor`): Residue vocabulary indices shaped `(batch, tokens)`.
        emit_mask (`torch.Tensor`): Boolean emitted-token mask shaped `(batch, tokens)`.
        atom_mask (`torch.Tensor`): Boolean valid-atom mask shaped `(batch, tokens, 37)`.
    """

    coordinates: torch.Tensor
    residue_type: torch.Tensor | None = None
    emit_mask: torch.Tensor | None = None
    atom_mask: torch.Tensor | None = None


class RFD4ProteinaAutoencoder(ModelMixin, ConfigMixin, AttentionMixin):
    """Paired molecular encoder and atom37 decoder.

    Coordinates supplied to and returned by this model are in nanometers.

    Args:
        latent_dim (`int`, defaults to `32`): Width of the per-token local latent.
        n_residue_types (`int`, defaults to `147`): Residue vocabulary size.
        num_token_types (`int`, defaults to `7`): Size of the token-type vocabulary: protein, RNA, DNA, atomized,
            unknown, glycan and non-canonical amino acid.
        allowed_residue_types (`list[list[bool]]`): `(num_token_types, n_residue_types)` table of the residue classes
            each token type may decode to.
        nlayers (`int`, defaults to `12`): Trunk layers in each of the encoder and decoder.
        token_dim (`int`, defaults to `768`): Token representation width.
        nheads (`int`, defaults to `12`): Attention heads per layer.
        dim_cond (`int`, defaults to `128`): Width of the (unused, zero) conditioning vector.
        pair_repr_dim (`int`, defaults to `256`): Pair width the trunk layers are constructed with. The released
            autoencoder uses no pair bias, so this holds no parameters.
        token_type_emb_dim (`int`, defaults to `32`): Token-type embedding width.
        encoder_feats_seq, decoder_feats_seq (`list[str]`): Features projected into the encoder and decoder inputs.
        atom37_mask (`list[list[bool]]`): Valid atom37 slots for each residue type, shaped `(n_residue_types, 37)`.
    """

    _no_split_modules = ["RFD4ProteinaTrunkLayer"]
    _repeated_blocks = ["RFD4ProteinaTrunkLayer"]
    _skip_layerwise_casting_patterns = ["norm", "embedding"]
    # The head producing the latent mean and log-scale decides the sampled latent, so it stays fp32.
    _keep_in_fp32_modules = ["latent_decoder_mean_n_log_scale", "norm"]
    # Callers reach this model through `encode`/`decode`, never `forward`, so a block-level group that
    # hooked the root would never onload. Naming the two submodules as blocks puts the hooks on modules
    # whose own forward does run, which is how `AutoencoderKL` supports offloading through the same path.
    _group_offload_block_modules = ["encoder", "decoder"]

    @register_to_config
    def __init__(
        self,
        atom37_mask: Sequence[Sequence[bool]],
        allowed_residue_types: Sequence[Sequence[bool]],
        latent_dim: int = 32,
        n_residue_types: int = 147,
        num_token_types: int = 7,
        nlayers: int = 12,
        token_dim: int = 768,
        nheads: int = 12,
        dim_cond: int = 128,
        pair_repr_dim: int = 256,
        token_type_emb_dim: int = 32,
        encoder_feats_seq: tuple[str, ...] = (
            "token_type",
            "x1_a37coors_nm",
            "x1_a37coors_nm_rel",
            "chain_idx_seq",
            "incident_bond_type_counts",
            "atom37_element_charge_embedding",
        ),
        decoder_feats_seq: tuple[str, ...] = ("ca_coors_nm", "z_latent_seq", "token_type", "cond_residue_type"),
    ) -> None:
        super().__init__()
        shared = {
            "nlayers": nlayers,
            "token_dim": token_dim,
            "nheads": nheads,
            "dim_cond": dim_cond,
            "pair_repr_dim": pair_repr_dim,
            "token_type_emb_dim": token_type_emb_dim,
            "latent_dim": latent_dim,
        }
        self.latent_dim = latent_dim
        vocabulary = {"n_residue_types": n_residue_types, "num_token_types": num_token_types}
        self.encoder = RFD4ProteinaEncoder(**shared, **vocabulary, feats_seq=encoder_feats_seq)
        self.decoder = RFD4ProteinaDecoder(
            **shared,
            **vocabulary,
            feats_seq=decoder_feats_seq,
            atom37_mask=atom37_mask,
            allowed_residue_types=allowed_residue_types,
        )

    @property
    def allowed_residue_types(self) -> torch.Tensor:
        """`(num_token_types, n_residue_types)` table of the classes each token type may decode to."""
        return self.decoder._allowed_residue_types

    @apply_forward_hook
    def encode(
        self,
        coords_nm: torch.Tensor,
        padding_mask: torch.Tensor,
        token_type: torch.Tensor | None = None,
        atom37_mask: torch.Tensor | None = None,
        conditioning: dict[str, Any] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
    ) -> RFD4ProteinaEncoderOutput | tuple:
        """Encode prepared coordinates and conditioning into sampled per-token local latents.

        Args:
            coords_nm (`torch.Tensor`): `(batch, tokens, 37, 3)` atom37 coordinates in nanometers.
            padding_mask (`torch.Tensor`): `(batch, tokens)`, true where a token is real rather than padding.
            token_type (`torch.Tensor`, *optional*): `(batch, tokens)` token-type codes.
            atom37_mask (`torch.Tensor`, *optional*): `(batch, tokens, 37)`, true where an atom slot is used.
            conditioning (`dict[str, Any]`, *optional*): The remaining topology and conditioning tensors,
                under the keys the preprocessor emits.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*): One generator per batch row
                for independent stochastic latent samples. Without one, sampling draws from the global stream.
            return_dict (`bool`, defaults to `True`): Return a typed output, or a tuple of its fields.

        Returns:
            [`RFD4ProteinaEncoderOutput`] or `tuple`: Latent, mean and log-scale tensors shaped `(batch, tokens,
            latent_dim)`.
        """
        features = {
            **(conditioning or {}),
            "compute_dtype": self.dtype,
            "coords_nm": coords_nm,
            "padding_mask": padding_mask,
        }
        if token_type is not None:
            features["token_type"] = token_type
        if atom37_mask is not None:
            features["atom37_mask"] = atom37_mask
        output = RFD4ProteinaEncoderOutput(**self.encoder(features, generator=generator))
        return output if return_dict else output.to_tuple()

    @apply_forward_hook
    def decode(
        self,
        z_latent: torch.Tensor,
        ca_coors_nm: torch.Tensor,
        padding_mask: torch.Tensor,
        token_type: torch.Tensor | None = None,
        atom37_mask: torch.Tensor | None = None,
        conditioning: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> RFD4ProteinaDecoderOutput | tuple:
        """Decode per-token latents and backbone positions into atom37 structure.

        Args:
            z_latent (`torch.Tensor`): `(batch, tokens, latent_dim)` per-token local latents.
            ca_coors_nm (`torch.Tensor`): `(batch, tokens, 3)` backbone CA positions in nanometers.
            padding_mask (`torch.Tensor`): `(batch, tokens)`, true where a token is real rather than padding.
            token_type (`torch.Tensor`, *optional*): `(batch, tokens)` token-type codes.
            atom37_mask (`torch.Tensor`, *optional*): `(batch, tokens, 37)`, true where an atom slot is used.
            conditioning (`dict[str, Any]`, *optional*): The remaining topology and conditioning tensors,
                under the keys the preprocessor emits. The sequence condition is read from here.
            return_dict (`bool`, defaults to `True`): Return a typed output, or a tuple of its fields.

        Returns:
            [`RFD4ProteinaDecoderOutput`] or `tuple`: Coordinates `(batch, tokens, 37, 3)` in nanometers, residue types
            `(batch, tokens)`, emission mask `(batch, tokens)` and atom mask `(batch, tokens, 37)`. Non-emitted atom
            coordinates are zero.
        """
        features = {
            **(conditioning or {}),
            "compute_dtype": self.dtype,
            "z_latent": z_latent,
            "ca_coors_nm": ca_coors_nm,
            "padding_mask": padding_mask,
        }
        if token_type is not None:
            features["token_type"] = token_type
        if atom37_mask is not None:
            features["atom37_mask"] = atom37_mask
        output = RFD4ProteinaDecoderOutput(**self.decoder(features))
        return output if return_dict else output.to_tuple()

    def forward(
        self,
        z_latent: torch.Tensor,
        ca_coors_nm: torch.Tensor,
        padding_mask: torch.Tensor,
        token_type: torch.Tensor | None = None,
        atom37_mask: torch.Tensor | None = None,
        conditioning: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> RFD4ProteinaDecoderOutput | tuple:
        """Decode per-token latents and backbone positions; equivalent to calling [`~RFD4ProteinaAutoencoder.decode`].

        Args:
            z_latent (`torch.Tensor`): `(batch, tokens, latent_dim)` per-token local latents.
            ca_coors_nm (`torch.Tensor`): `(batch, tokens, 3)` backbone CA positions in nanometers.
            padding_mask (`torch.Tensor`): `(batch, tokens)`, true where a token is real rather than padding.
            token_type (`torch.Tensor`, *optional*): `(batch, tokens)` token-type codes.
            atom37_mask (`torch.Tensor`, *optional*): `(batch, tokens, 37)`, true where an atom slot is used.
            conditioning (`dict[str, Any]`, *optional*): The remaining topology and conditioning tensors.
            return_dict (`bool`, defaults to `True`): Return a typed output, or a tuple of its fields.

        Returns:
            [`RFD4ProteinaDecoderOutput`] or `tuple`: Decoded coordinates, residue types and atom/emit masks.
        """
        return self.decode(
            z_latent=z_latent,
            ca_coors_nm=ca_coors_nm,
            padding_mask=padding_mask,
            token_type=token_type,
            atom37_mask=atom37_mask,
            conditioning=conditioning,
            return_dict=return_dict,
        )
