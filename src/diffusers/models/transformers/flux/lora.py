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

"""Flux LoRA conversion.

Each supported foreign format has a top-level ``map_<format>_to_diffusers`` entry point:

    - :func:`map_bfl_to_diffusers` — original BFL repo layout
    - :func:`map_kontext_to_diffusers` — fal Kontext checkpoints (BFL + ``base_model.model.`` prefix)
    - :func:`map_xlabs_to_diffusers` — XLabs ``.processor.qkv_lora`` / ``.processor.proj_lora`` shape
    - :func:`map_kohya_to_diffusers` — kohya sd-scripts (and "mixture" / ``lora_transformer_*`` variants)

Each entry point produces a state dict with diffusers naming. Internally they all funnel through
:func:`_map_to_diffusers`, which converts a BFL-style state dict (original Flux module names + ``.lora_A``/``.lora_B``
suffixes) to diffusers names by reusing the rename / QKV-split / special-key tables in ``weight_mapping.py`` and
applying LoRA-specific QKV semantics (``lora_A.weight`` replicates across heads; everything else chunks).

A format-specific converter may also emit pre-converted diffusers keys directly when a key shape doesn't fit the
canonical intermediate (e.g., XLabs single-block QKV without a paired MLP LoRA).
"""

import re

import torch

from ....loaders.lora import LoRAHandler
from ....utils import logging, state_dict_all_zero
from .weight_mapping import (
    FLUX_QKV_SPLIT_PATTERNS,
    FLUX_QKVMLP_SPLIT_PATTERN,
    FLUX_QKVMLP_TARGETS,
    FLUX_RENAME_PATTERNS,
    FLUX_SPECIAL_KEYS,
)


logger = logging.get_logger(__name__)


# ============================================================================
# Shared canonical -> diffusers converter
# ============================================================================
# Canonical keys are BFL-style: original Flux module names + .lora_A/.lora_B
# suffixes. The shared converter handles three cases — pure renames, QKV splits,
# special transforms — by reusing the tables from weight_mapping.

_LORA_SUFFIXES = (".lora_A.weight", ".lora_A.bias", ".lora_B.weight", ".lora_B.bias")

# Module-path versions (boundary dots stripped) of the weight-mapping tables.
_QKV_PATTERNS = {p.strip("."): [t.strip(".") for t in ts] for p, ts in FLUX_QKV_SPLIT_PATTERNS.items()}
_QKVMLP_PATTERN = FLUX_QKVMLP_SPLIT_PATTERN.strip(".")
_QKVMLP_TARGETS = [t.strip(".") for t in FLUX_QKVMLP_TARGETS]
_SPECIAL_MODULES = {}
for _full_src, _spec in FLUX_SPECIAL_KEYS.items():
    for _tail in (".weight", ".bias"):
        if _full_src.endswith(_tail) and _spec["target"].endswith(_tail):
            _SPECIAL_MODULES.setdefault(_full_src[: -len(_tail)], (_spec["target"][: -len(_tail)], _spec["transform"]))
            break


def _apply_renames(s, patterns):
    for old, new in patterns.items():
        s = s.replace(old, new)
    return s


def _map_to_diffusers(state_dict, inner_dim=3072, mlp_ratio=4.0):
    """Convert a BFL-style canonical LoRA state dict to diffusers naming."""
    out = {}
    qkvmlp_dims = (inner_dim, inner_dim, inner_dim, int(inner_dim * mlp_ratio))

    for key, value in state_dict.items():
        # Split off the .lora_A/.lora_B suffix; non-LoRA keys pass through with renames.
        suffix = next((s for s in _LORA_SUFFIXES if key.endswith(s)), "")
        if not suffix:
            out[f"transformer.{_apply_renames(key, FLUX_RENAME_PATTERNS)}"] = value
            continue
        module_path = key[: -len(suffix)]

        # FLUX_RENAME_PATTERNS keys often end with "."; pad-and-strip so bare module paths
        # like "final_layer.linear" still match patterns like "final_layer.linear.".
        def _rename(path):
            renamed = _apply_renames(path + ".", FLUX_RENAME_PATTERNS)
            return renamed[:-1] if renamed.endswith(".") else renamed

        qkv = next(((p, ts) for p, ts in _QKV_PATTERNS.items() if p in module_path), None)
        if qkv is not None:
            pattern, targets = qkv
            chunks = (
                [value] * len(targets) if suffix == ".lora_A.weight" else list(torch.chunk(value, len(targets), dim=0))
            )
            for target, chunk in zip(targets, chunks):
                out[f"transformer.{_rename(module_path.replace(pattern, target))}{suffix}"] = chunk
            continue

        if _QKVMLP_PATTERN in module_path and "single_blocks." in module_path:
            chunks = (
                [value] * len(_QKVMLP_TARGETS)
                if suffix == ".lora_A.weight"
                else list(torch.split(value, qkvmlp_dims, dim=0))
            )
            for target, chunk in zip(_QKVMLP_TARGETS, chunks):
                out[f"transformer.{_rename(module_path.replace(_QKVMLP_PATTERN, target))}{suffix}"] = chunk
            continue

        if module_path in _SPECIAL_MODULES:
            target_module, transform = _SPECIAL_MODULES[module_path]
            out[f"transformer.{target_module}{suffix}"] = transform(value)
            continue

        out[f"transformer.{_rename(module_path)}{suffix}"] = value

    return out


# ============================================================================
# BFL — identity (canonical form is BFL-style)
# ============================================================================


def map_bfl_to_diffusers(state_dict):
    """Convert a Flux LoRA state dict from BFL format to diffusers naming."""
    return _map_to_diffusers(dict(state_dict))


# ============================================================================
# fal Kontext — BFL with ``base_model.model.`` prefix
# ============================================================================


def map_kontext_to_diffusers(state_dict):
    """Convert a Flux LoRA state dict from fal Kontext format to diffusers naming."""
    prefix = "base_model.model."
    canonical = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
    return _map_to_diffusers(canonical)


# ============================================================================
# XLabs
# ============================================================================
# XLabs key shape: [diffusion_model.]{double|single}_blocks.{i}.processor.{X}.{down|up}.weight
# Double-block X ∈ {qkv_lora1, qkv_lora2, proj_lora1, proj_lora2} — renameable to canonical
# BFL form. Single-block X ∈ {qkv_lora, proj_lora} — single blocks lack an MLP LoRA, so
# qkv keys can't be expressed as canonical "linear1" (QKV+MLP fused); we emit pre-converted
# diffusers keys for single-block extras and route only double-block keys through canonical.

_XLABS_DOUBLE_RENAMES = {
    ".processor.proj_lora1.": ".img_attn.proj.",
    ".processor.proj_lora2.": ".txt_attn.proj.",
    ".processor.qkv_lora1.": ".img_attn.qkv.",
    ".processor.qkv_lora2.": ".txt_attn.qkv.",
}
_XLABS_SINGLE_QKV_TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v"]


def map_xlabs_to_diffusers(state_dict):
    """Convert a Flux LoRA state dict from XLabs format to diffusers naming."""
    canonical = {}
    extras = {}
    for key, value in state_dict.items():
        k = key.removeprefix("diffusion_model.")

        if "single_blocks." in k:
            block = re.search(r"single_blocks\.(\d+)", k).group(1)
            base = f"transformer.single_transformer_blocks.{block}"
            suffix = ".lora_A.weight" if k.endswith(".lora_A.weight") else ".lora_B.weight"
            if "proj_lora" in k:
                extras[f"{base}.proj_out{suffix}"] = value
            elif "qkv_lora" in k:
                chunks = (
                    [value] * len(_XLABS_SINGLE_QKV_TARGETS)
                    if suffix == ".lora_A.weight"
                    else list(torch.chunk(value, 3, dim=0))
                )
                for t, chunk in zip(_XLABS_SINGLE_QKV_TARGETS, chunks):
                    extras[f"{base}.{t}{suffix}"] = chunk
            continue

        # Double block: rename to canonical BFL-style; shared converter handles the QKV split.
        for old, new in _XLABS_DOUBLE_RENAMES.items():
            k = k.replace(old, new)
        canonical[k] = value

    converted = _map_to_diffusers(canonical) if canonical else {}
    return {**converted, **extras}


# ============================================================================
# Kohya (sd-scripts + "mixture" variant)
# ============================================================================
# Kohya keys collapse dots into underscores in the module path, then append
# .lora_down/.lora_up/.alpha. We invert this with a single explicit suffix table
# (the original-name underscore <-> dot mapping isn't recoverable by rule), then
# apply alpha-driven scaling so canonical tensors are pre-scaled.

# Kohya stub-suffix → BFL form. Block stubs (``double_blocks_{i}_<suffix>`` and
# ``single_blocks_{i}_<suffix>``) look up just the trailing <suffix> here; everything
# else is a global stub that maps directly. No overlap between contexts.
_KOHYA_TO_BFL = {
    # double_blocks_{i}_<suffix>
    "img_attn_proj": "img_attn.proj",
    "img_attn_qkv": "img_attn.qkv",
    "img_mlp_0": "img_mlp.0",
    "img_mlp_2": "img_mlp.2",
    "img_mod_lin": "img_mod.lin",
    "txt_attn_proj": "txt_attn.proj",
    "txt_attn_qkv": "txt_attn.qkv",
    "txt_mlp_0": "txt_mlp.0",
    "txt_mlp_2": "txt_mlp.2",
    "txt_mod_lin": "txt_mod.lin",
    # single_blocks_{i}_<suffix>
    "linear1": "linear1",
    "linear2": "linear2",
    "modulation_lin": "modulation.lin",
    # Global stubs (used directly as canonical path)
    "guidance_in_in_layer": "guidance_in.in_layer",
    "guidance_in_out_layer": "guidance_in.out_layer",
    "img_in": "img_in",
    "txt_in": "txt_in",
    "time_in_in_layer": "time_in.in_layer",
    "time_in_out_layer": "time_in.out_layer",
    "vector_in_in_layer": "vector_in.in_layer",
    "vector_in_out_layer": "vector_in.out_layer",
    "final_layer_linear": "final_layer.linear",
    "final_layer_adaLN_modulation_1": "final_layer.adaLN_modulation.1",
}


def _kohya_scale(alpha, rank):
    """Split alpha/rank into (down, up) factors so down*up == alpha/rank but stays bounded."""
    scale = alpha / rank
    down, up = scale, 1.0
    while down * 2 < up:
        down *= 2
        up /= 2
    return down, up


def _kohya_mixture_to_diffusers(state_dict):
    """Convert Kohya mixture-format (``lora_transformer_*`` keys) directly to diffusers naming."""
    out = {}
    unique = {
        k.replace(".lora_A.weight", "").replace(".lora_B.weight", "").replace(".alpha", "")
        for k in state_dict
        if k.startswith("lora_transformer_")
    }

    for k in unique:
        if k.startswith("lora_transformer_single_transformer_blocks_"):
            i = int(k.split("lora_transformer_single_transformer_blocks_")[-1].split("_")[0])
            diffusers_key = f"single_transformer_blocks.{i}"
        elif k.startswith("lora_transformer_transformer_blocks_"):
            i = int(k.split("lora_transformer_transformer_blocks_")[-1].split("_")[0])
            diffusers_key = f"transformer_blocks.{i}"
        elif k.startswith("lora_transformer_context_embedder"):
            diffusers_key = "context_embedder"
        elif k.startswith("lora_transformer_norm_out_linear"):
            diffusers_key = "norm_out.linear"
        elif k.startswith("lora_transformer_proj_out"):
            diffusers_key = "proj_out"
        elif k.startswith("lora_transformer_x_embedder"):
            diffusers_key = "x_embedder"
        elif k.startswith("lora_transformer_time_text_embed_guidance_embedder_linear_"):
            i = int(k.split("lora_transformer_time_text_embed_guidance_embedder_linear_")[-1])
            diffusers_key = f"time_text_embed.guidance_embedder.linear_{i}"
        elif k.startswith("lora_transformer_time_text_embed_text_embedder_linear_"):
            i = int(k.split("lora_transformer_time_text_embed_text_embedder_linear_")[-1])
            diffusers_key = f"time_text_embed.text_embedder.linear_{i}"
        elif k.startswith("lora_transformer_time_text_embed_timestep_embedder_linear_"):
            i = int(k.split("lora_transformer_time_text_embed_timestep_embedder_linear_")[-1])
            diffusers_key = f"time_text_embed.timestep_embedder.linear_{i}"
        else:
            raise NotImplementedError(f"Handling for key ({k}) is not implemented.")

        if "attn_" in k:
            tail = k.split("attn_")[-1]
            if "_to_out_0" in k:
                diffusers_key += ".attn.to_out.0"
            elif "_to_add_out" in k:
                diffusers_key += ".attn.to_add_out"
            elif any(qkv in k for qkv in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")):
                diffusers_key += f".attn.{tail}"

        down = state_dict.pop(f"{k}.lora_A.weight")
        up = state_dict.pop(f"{k}.lora_B.weight")
        alpha = state_dict.pop(f"{k}.alpha")
        d_scale, u_scale = _kohya_scale(alpha, down.shape[0])
        out[f"transformer.{diffusers_key}.lora_A.weight"] = down * d_scale
        out[f"transformer.{diffusers_key}.lora_B.weight"] = up * u_scale

    leftover = [k for k in state_dict if not k.startswith("lora_unet_")]
    if leftover:
        logger.warning(f"Unsupported mixture keys ignored: {leftover}")
    return out


def map_kohya_to_diffusers(state_dict):
    """Convert a Flux LoRA state dict from Kohya format (sd-scripts or mixture) to diffusers naming."""
    # ---- Pre-filter: rename prefix, drop unsupported keys, collapse leading dots. ----
    state_dict = {k.replace("diffusion_model.", "lora_unet_"): v for k, v in state_dict.items()}

    drop_specs = [
        (lambda k: "position_embedding" in k, "position_embedding", "position_embedding"),
        (lambda k: ".diff_b" in k and k.startswith("lora_unet_"), ".diff_b", "diff_b"),
        (lambda k: ".norm" in k and ".diff" in k, ".diff", "diff"),
    ]
    for predicate, marker, label in drop_specs:
        if not any(predicate(k) for k in state_dict):
            continue
        msg = (
            f"The `{label}` LoRA params are all zeros which make them ineffective. So, we will purge them out of "
            "the current state dict to make loading possible."
            if state_dict_all_zero(state_dict, marker)
            else f"`{label}` keys found in the state dict are currently unsupported and will be filtered out. "
            "Open an issue if this is a problem - https://github.com/huggingface/diffusers/issues/new."
        )
        logger.info(msg)
        state_dict = {k: v for k, v in state_dict.items() if not predicate(k)}

    # Some keys come with dots in the prefix; collapse them up to lora_A/lora_B/alpha.
    limit = ["lora_A", "lora_B"] + (["alpha"] if any("alpha" in k for k in state_dict) else [])
    boundary_re = re.compile("(" + "|".join(re.escape(s) for s in limit) + ")")

    def _collapse_prefix(key):
        match = boundary_re.search(key)
        if not match:
            return key.replace(".", "_")
        i = match.start()
        boundary = i - 1 if i > 0 and key[i - 1] == "." else i
        return key[:boundary].replace(".", "_") + key[boundary:]

    state_dict = {_collapse_prefix(k): v for k, v in state_dict.items() if k.startswith("lora_unet_")}

    # ---- Mixture variant has its own prefix; route to its direct converter. ----
    if any(
        k.startswith("lora_transformer_") and ("lora_down" in k or "lora_up" in k or "alpha" in k) for k in state_dict
    ):
        return _kohya_mixture_to_diffusers(state_dict)

    # ---- sd-scripts variant: group by stub, apply alpha scaling, rewrite to canonical. ----
    groups = {}  # stub -> {"lora_A": full_key, "lora_B": full_key, "alpha": full_key}
    for key in list(state_dict):
        if not key.startswith("lora_unet_"):
            continue
        for kind in ("lora_A.weight", "lora_B.weight", "alpha"):
            tail = "." + kind
            if key.endswith(tail):
                stub = key[len("lora_unet_") : -len(tail)]
                groups.setdefault(stub, {})[kind.split(".")[0]] = key
                break

    canonical = {}
    for stub, group in groups.items():
        down_key, up_key = group.get("lora_A"), group.get("lora_B")
        if down_key is None or up_key is None:
            continue
        rank = state_dict[down_key].shape[0]
        alpha = state_dict.pop(group["alpha"]).item() if "alpha" in group else float(rank)
        d_scale, u_scale = _kohya_scale(alpha, rank)
        down = state_dict.pop(down_key) * d_scale
        up = state_dict.pop(up_key) * u_scale

        # Map kohya stub → BFL canonical path. Block stubs strip their "{kind}_blocks_{i}_"
        # prefix and look up the trailing suffix; global stubs map directly.
        bfl = None
        for kind in ("double_blocks", "single_blocks"):
            m = re.match(rf"{kind}_(\d+)_(.+)$", stub)
            if m:
                suffix = _KOHYA_TO_BFL.get(m.group(2))
                bfl = f"{kind}.{m.group(1)}.{suffix}" if suffix else None
                break
        else:
            bfl = _KOHYA_TO_BFL.get(stub)

        if bfl is None:
            logger.warning(f"Unsupported Kohya key: lora_unet_{stub}")
            continue
        canonical[f"{bfl}.lora_A.weight"] = down
        canonical[f"{bfl}.lora_B.weight"] = up

    if state_dict:
        logger.warning(f"Unsupported keys after Kohya normalization: {list(state_dict.keys())}")

    return _map_to_diffusers(canonical)


# ============================================================================
# Top-level dispatch
# ============================================================================
# Per-format identifying key substrings. Single source of truth — also exported
# via ``FLUX_LORA_METADATA`` so ``LoRAModelMixin._detect_lora_format`` finds it.

_FLUX_LORA_FORMAT_KEYS: dict[str, set[str]] = {
    "kohya": {"lora_unet_double_blocks_", "lora_unet_single_blocks_"},
    "xlabs": {".processor.qkv_lora", ".processor.proj_lora"},
    "bfl": {"time_in.in_layer.lora_A", "double_blocks.0.img_mod.lin.lora_A"},
    "kontext": {"base_model.model.double_blocks"},
}

_FORMAT_DISPATCH = {
    "bfl": map_bfl_to_diffusers,
    "kontext": map_kontext_to_diffusers,
    "xlabs": map_xlabs_to_diffusers,
    "kohya": map_kohya_to_diffusers,
}


def map_lora_to_diffusers(state_dict, **kwargs):
    """Detect a Flux LoRA's source format and dispatch to its per-format converter.

    Already-converted (peft) state dicts pass through after filtering to ``transformer.*`` keys. Unknown formats (incl.
    diffusers-native LoRAs with raw ``.alpha`` keys) pass through unchanged so the pipeline's diffusers-native fallback
    can run.
    """
    if any(k.startswith("transformer.") for k in state_dict):
        return {k: v for k, v in state_dict.items() if k.startswith("transformer.")}

    keys = set(state_dict)
    for fmt, fmt_keys in _FLUX_LORA_FORMAT_KEYS.items():
        if any(any(fk in k for k in keys) for fk in fmt_keys):
            return _FORMAT_DISPATCH[fmt](state_dict)
    return state_dict


# Handler assembled into ``ModelMetadata`` by ``flux/model.py``.
FLUX_LORA = LoRAHandler(
    format_keys=_FLUX_LORA_FORMAT_KEYS,
    map_lora_to_diffusers_fn=map_lora_to_diffusers,
)
