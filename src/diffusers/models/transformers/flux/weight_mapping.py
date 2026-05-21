# Copyright 2025 Black Forest Labs, The HuggingFace Team. All rights reserved.
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

from typing import Any

import torch

from ....loaders.weight_mapping import WeightMappingHandler


def swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    """Swap scale and shift in AdaLayerNorm weights (original uses shift,scale; diffusers uses scale,shift)."""
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


# Pattern-based key renaming (substring replacements applied in order)
FLUX_RENAME_PATTERNS: dict[str, str] = {
    # Global key renames
    "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
    "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
    "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
    "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
    "txt_in.": "context_embedder.",
    "img_in.": "x_embedder.",
    "final_layer.linear.": "proj_out.",
    # Double block patterns
    "double_blocks.": "transformer_blocks.",
    ".img_mod.lin.": ".norm1.linear.",
    ".txt_mod.lin.": ".norm1_context.linear.",
    ".img_attn.norm.query_norm.scale": ".attn.norm_q.weight",
    ".img_attn.norm.key_norm.scale": ".attn.norm_k.weight",
    ".txt_attn.norm.query_norm.scale": ".attn.norm_added_q.weight",
    ".txt_attn.norm.key_norm.scale": ".attn.norm_added_k.weight",
    ".img_mlp.0.": ".ff.net.0.proj.",
    ".img_mlp.2.": ".ff.net.2.",
    ".txt_mlp.0.": ".ff_context.net.0.proj.",
    ".txt_mlp.2.": ".ff_context.net.2.",
    ".img_attn.proj.": ".attn.to_out.0.",
    ".txt_attn.proj.": ".attn.to_add_out.",
    # Single block patterns
    "single_blocks.": "single_transformer_blocks.",
    ".modulation.lin.": ".norm.linear.",
    ".norm.query_norm.scale": ".attn.norm_q.weight",
    ".norm.key_norm.scale": ".attn.norm_k.weight",
    ".linear2.": ".proj_out.",
}


# --------------------------------------------------------------------------
# Per-key transforms (split + special), unified.
# --------------------------------------------------------------------------
# Single source of truth. Each entry is
#   (source_substring, [target_substrings], forward_fn, reverse_fn)
# - source/targets include surrounding dots so they only match at module
#   boundaries (e.g. ".img_attn.qkv." matches both "X.img_attn.qkv.weight"
#   and "X.img_attn.qkv.bias" with one entry).
# - len(targets) == 1 -> a unary transform (e.g. AdaLN scale/shift swap).
# - len(targets)  > 1 -> a split transform (forward chunks the tensor).
# - forward_fn(tensor, **ctx) -> list[tensor] of length len(targets).
# - reverse_fn(list[tensor], **ctx) -> tensor.
def _swap_to_list(v, **_):
    return [swap_scale_shift(v)]


def _list_to_swap(vs, **_):
    return swap_scale_shift(vs[0])


def _make_chunk(n):
    return lambda v, **_: torch.chunk(v, n, dim=0)


def _qkvmlp_split(v, inner_dim=3072, **_):
    return torch.split(v, [inner_dim, inner_dim, inner_dim, inner_dim * 4], dim=0)


def _cat0(vs, **_):
    return torch.cat(vs, dim=0)


FLUX_TRANSFORMS = [
    ("final_layer.adaLN_modulation.1.", ["norm_out.linear."], _swap_to_list, _list_to_swap),
    (".img_attn.qkv.", [".attn.to_q.", ".attn.to_k.", ".attn.to_v."], _make_chunk(3), _cat0),
    (
        ".txt_attn.qkv.",
        [".attn.add_q_proj.", ".attn.add_k_proj.", ".attn.add_v_proj."],
        _make_chunk(3),
        _cat0,
    ),
    (".linear1.", [".attn.to_q.", ".attn.to_k.", ".attn.to_v.", ".proj_mlp."], _qkvmlp_split, _cat0),
]


# Backward-compat tables derived from FLUX_TRANSFORMS so the existing
# map_from_diffusers code (and any external readers, including lora.py)
# keep working without changes.
def _wrap_unary(fwd_fn):
    return lambda v: fwd_fn(v)[0]


FLUX_SPECIAL_KEYS: dict[str, dict] = {}
FLUX_QKV_SPLIT_PATTERNS: dict[str, list[str]] = {}
FLUX_QKVMLP_SPLIT_PATTERN: str = ""
FLUX_QKVMLP_TARGETS: list[str] = []
for _src, _tgts, _fwd, _ in FLUX_TRANSFORMS:
    if len(_tgts) == 1:
        for _suffix in ("weight", "bias"):
            FLUX_SPECIAL_KEYS[_src + _suffix] = {
                "target": _tgts[0] + _suffix,
                "transform": _wrap_unary(_fwd),
            }
    elif _src == ".linear1.":
        FLUX_QKVMLP_SPLIT_PATTERN = _src
        FLUX_QKVMLP_TARGETS = list(_tgts)
    else:
        FLUX_QKV_SPLIT_PATTERNS[_src] = list(_tgts)


def _get_inner_dim(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer inner_dim from state_dict weights."""
    for key in state_dict:
        if "single_blocks." in key and ".linear1." in key and key.endswith(".bias"):
            # linear1 contains Q, K, V, MLP fused - Q/K/V each have inner_dim
            # Total size = 3 * inner_dim + mlp_hidden_dim = 3 * inner_dim + 4 * inner_dim = 7 * inner_dim
            total = state_dict[key].shape[0]
            return total // 7
    return 3072  # Default


def map_to_diffusers(
    state_dict: dict[str, torch.Tensor],
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Convert a Flux transformer state_dict from original format to diffusers format."""
    inner_dim = _get_inner_dim(state_dict)
    return WeightMappingHandler.apply_transforms(
        state_dict, FLUX_TRANSFORMS, FLUX_RENAME_PATTERNS, inner_dim=inner_dim
    )


# Build reverse patterns for map_from_diffusers
FLUX_RENAME_PATTERNS_REVERSE: dict[str, str] = {v: k for k, v in FLUX_RENAME_PATTERNS.items()}
FLUX_SPECIAL_KEYS_REVERSE: dict[str, dict] = {
    v["target"]: {"target": k, "transform": v["transform"]} for k, v in FLUX_SPECIAL_KEYS.items()
}
FLUX_QKV_SPLIT_PATTERNS_REVERSE: dict[str, str] = {
    target: pattern for pattern, targets in FLUX_QKV_SPLIT_PATTERNS.items() for target in targets
}


def map_from_diffusers(
    state_dict: dict[str, torch.Tensor],
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Convert a Flux transformer state_dict from diffusers format to original format.

    Args:
        state_dict: State dict in diffusers format

    Returns:
        State dict in original Flux format
    """
    converted_state_dict = {}
    keys = list(state_dict.keys())

    # Group keys for QKV concatenation
    qkv_groups: dict[str, list[tuple[str, torch.Tensor]]] = {}
    qkvmlp_groups: dict[str, list[tuple[str, torch.Tensor]]] = {}

    for key in keys:
        value = state_dict[key]

        # Handle special keys with transforms
        if key in FLUX_SPECIAL_KEYS_REVERSE:
            spec = FLUX_SPECIAL_KEYS_REVERSE[key]
            converted_state_dict[spec["target"]] = spec["transform"](value)
            continue

        # Check if this is part of a QKV group (double blocks)
        qkv_pattern = None
        for target, pattern in FLUX_QKV_SPLIT_PATTERNS_REVERSE.items():
            if target in key:
                qkv_pattern = pattern
                break

        if qkv_pattern and "transformer_blocks." in key:
            # Build the original key by replacing target with pattern
            base_key = key
            for target in FLUX_QKV_SPLIT_PATTERNS_REVERSE:
                if target in base_key:
                    base_key = base_key.replace(target, qkv_pattern)
                    break
            orig_key = WeightMappingHandler.rename_key(base_key, FLUX_RENAME_PATTERNS_REVERSE)

            if orig_key not in qkv_groups:
                qkv_groups[orig_key] = []
            qkv_groups[orig_key].append((key, value))
            continue

        # Check if this is part of a QKV+MLP group (single blocks)
        is_qkvmlp = False
        for target in FLUX_QKVMLP_TARGETS:
            if target in key and "single_transformer_blocks." in key:
                base_key = key.replace(target, FLUX_QKVMLP_SPLIT_PATTERN)
                orig_key = WeightMappingHandler.rename_key(base_key, FLUX_RENAME_PATTERNS_REVERSE)

                if orig_key not in qkvmlp_groups:
                    qkvmlp_groups[orig_key] = []
                qkvmlp_groups[orig_key].append((key, value))
                is_qkvmlp = True
                break

        if is_qkvmlp:
            continue

        # Standard rename
        new_key = WeightMappingHandler.rename_key(key, FLUX_RENAME_PATTERNS_REVERSE)
        converted_state_dict[new_key] = value

    # Concatenate QKV groups
    for orig_key, items in qkv_groups.items():
        if len(items) == 3:
            # Sort by the target pattern order
            items.sort(
                key=lambda x: next(
                    i
                    for i, t in enumerate(
                        FLUX_QKV_SPLIT_PATTERNS[".img_attn.qkv."]
                        if ".img_attn." in orig_key
                        else FLUX_QKV_SPLIT_PATTERNS[".txt_attn.qkv."]
                    )
                    if t in x[0]
                )
            )
            converted_state_dict[orig_key] = torch.cat([v for _, v in items], dim=0)

    # Concatenate QKV+MLP groups
    for orig_key, items in qkvmlp_groups.items():
        if len(items) == 4:
            items.sort(key=lambda x: next(i for i, t in enumerate(FLUX_QKVMLP_TARGETS) if t in x[0]))
            converted_state_dict[orig_key] = torch.cat([v for _, v in items], dim=0)

    return converted_state_dict


_FLUX_CHECKPOINT_KEY_PREFIXES: list[str] = ["model.diffusion_model."]

# Distinctive keys for original format detection (only keys that use simple renaming, not splits)
_FLUX_CHECKPOINT_KEYS: set[str] = {
    "time_in.in_layer.weight",
    "double_blocks.0.img_mod.lin.weight",
}
_FLUX_AVAILABLE_CONFIGS: dict[str, str] = {
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux-fill": "black-forest-labs/FLUX.1-Fill-dev",
    "flux-depth": "black-forest-labs/FLUX.1-Depth-dev",
}


def detect_config(weight_mapping, state_dict: dict[str, Any]) -> str | None:
    """Detect which Flux config name matches this state_dict.

    Receives the :class:`WeightMappingHandler` (not the model class) so it can call ``is_original_format`` and
    ``rename_key`` directly on the subsystem that owns them.
    """
    guidance_key = "guidance_in.in_layer.bias"
    x_embedder_key = "img_in.weight"

    if not weight_mapping.is_original_format(state_dict):
        guidance_key = weight_mapping.rename_key(guidance_key, FLUX_RENAME_PATTERNS)
        x_embedder_key = weight_mapping.rename_key(x_embedder_key, FLUX_RENAME_PATTERNS)

    if x_embedder_key not in state_dict:
        return None

    if guidance_key not in state_dict:
        return "flux-schnell"

    in_channels = state_dict[x_embedder_key].shape[1]
    if in_channels == 384:
        return "flux-fill"
    elif in_channels == 128:
        return "flux-depth"

    return "flux-dev"


# Handler assembled into ``ModelMetadata`` by ``flux/model.py``.
FLUX_WEIGHT_MAPPING = WeightMappingHandler(
    checkpoint_keys=_FLUX_CHECKPOINT_KEYS,
    checkpoint_key_prefixes=_FLUX_CHECKPOINT_KEY_PREFIXES,
    rename_patterns=FLUX_RENAME_PATTERNS,
    available_configs=_FLUX_AVAILABLE_CONFIGS,
    map_to_diffusers_fn=map_to_diffusers,
    map_from_diffusers_fn=map_from_diffusers,
    detect_config_fn=detect_config,
    # Kicks in only when ``detect_config`` returns ``None`` (e.g. the ``img_in`` / ``x_embedder`` key is
    # absent so we can't read in_channels). Most Flux checkpoints in the wild are dev-derived, so it's
    # the safest fallback config to load.
    default_config="flux-dev",
    default_subfolder="transformer",
)
