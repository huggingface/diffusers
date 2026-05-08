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

Pipeline:
    1. Detect the source format (kohya / xlabs / bfl / kontext) via
       ``FluxLoRAMappingMixin._detect_lora_format``.
    2. Run the format-specific *normalizer* that rewrites keys into the
       canonical "BFL-style" form: original Flux module names with
       ``.lora_A`` / ``.lora_B`` suffixes. Tensor-level transforms unique
       to a format (e.g., Kohya alpha scaling) happen here.
    3. Run the shared *converter* that maps canonical keys to diffusers
       names by reusing the rename / QKV-split / special-key tables in
       ``weight_mapping.py``, applying the LoRA-specific QKV semantics
       (lora_A weight replicates, everything else chunks).

A normalizer may also emit keys that bypass step 3 (returned as a second
"extras" dict) — used for keys that don't fit the canonical intermediate
(e.g., text-encoder LoRA keys, XLabs single-block QKV without MLP).
"""

import re

import torch

from ....loaders.lora_base import LoRAMappingMixin
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
# Stage 3: shared canonical -> diffusers converter
# ============================================================================
# Canonical keys are BFL-style: original Flux module names + .lora_A/.lora_B
# suffixes. The shared converter handles three cases — pure renames, QKV
# splits, special transforms — by reusing the tables from weight_mapping.

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


def _split_lora_suffix(key):
    for suffix in _LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return key, ""


def _apply_renames(s, patterns):
    for old, new in patterns.items():
        s = s.replace(old, new)
    return s


def _rename_module(module_path):
    # FLUX_RENAME_PATTERNS keys often have a trailing "."; pad-and-strip so
    # bare module paths like "final_layer.linear" still match patterns like
    # "final_layer.linear.".
    out = _apply_renames(module_path + ".", FLUX_RENAME_PATTERNS)
    return out[:-1] if out.endswith(".") else out


def _map_lora_to_diffusers(state_dict, inner_dim=3072, mlp_ratio=4.0):
    out = {}
    qkvmlp_dims = (inner_dim, inner_dim, inner_dim, int(inner_dim * mlp_ratio))

    for key, value in state_dict.items():
        module_path, suffix = _split_lora_suffix(key)

        if not suffix:
            out[f"transformer.{_apply_renames(key, FLUX_RENAME_PATTERNS)}"] = value
            continue

        qkv = next(((p, ts) for p, ts in _QKV_PATTERNS.items() if p in module_path), None)
        if qkv is not None:
            pattern, targets = qkv
            chunks = (
                [value] * len(targets) if suffix == ".lora_A.weight" else list(torch.chunk(value, len(targets), dim=0))
            )
            for target, chunk in zip(targets, chunks):
                new_module = _rename_module(module_path.replace(pattern, target))
                out[f"transformer.{new_module}{suffix}"] = chunk
            continue

        if _QKVMLP_PATTERN in module_path and "single_blocks." in module_path:
            chunks = (
                [value] * len(_QKVMLP_TARGETS)
                if suffix == ".lora_A.weight"
                else list(torch.split(value, qkvmlp_dims, dim=0))
            )
            for target, chunk in zip(_QKVMLP_TARGETS, chunks):
                new_module = _rename_module(module_path.replace(_QKVMLP_PATTERN, target))
                out[f"transformer.{new_module}{suffix}"] = chunk
            continue

        if module_path in _SPECIAL_MODULES:
            target_module, transform = _SPECIAL_MODULES[module_path]
            out[f"transformer.{target_module}{suffix}"] = transform(value)
            continue

        out[f"transformer.{_rename_module(module_path)}{suffix}"] = value

    return out


# ============================================================================
# Stage 2a: BFL normalizer (identity)
# ============================================================================


def _normalize_bfl(state_dict):
    return dict(state_dict), {}


# ============================================================================
# Stage 2b: fal Kontext normalizer (strip "base_model.model." prefix)
# ============================================================================


def _normalize_kontext(state_dict):
    prefix = "base_model.model."
    canonical = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
    return canonical, {}


# ============================================================================
# Stage 2c: XLabs normalizer
# ============================================================================
# XLabs key shape: [diffusion_model.]{double|single}_blocks.{i}.processor.{X}.{down|up}.weight
# Double-block X ∈ {qkv_lora1, qkv_lora2, proj_lora1, proj_lora2}.
# Single-block X ∈ {qkv_lora, proj_lora}. Single-block lacks an MLP LoRA, so
# its qkv keys can't be expressed as canonical "linear1" (which is QKV+MLP);
# we emit pre-converted diffusers keys for single blocks instead.

_XLABS_DOUBLE_RENAMES = {
    ".processor.proj_lora1.": ".img_attn.proj.",
    ".processor.proj_lora2.": ".txt_attn.proj.",
    ".processor.qkv_lora1.": ".img_attn.qkv.",
    ".processor.qkv_lora2.": ".txt_attn.qkv.",
}
_XLABS_SINGLE_QKV_TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v"]


def _normalize_xlabs(state_dict):
    canonical = {}
    extras = {}
    for key, value in state_dict.items():
        k = key
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model.") :]

        if "single_blocks." in k:
            block = re.search(r"single_blocks\.(\d+)", k).group(1)
            base = f"transformer.single_transformer_blocks.{block}"
            suffix = ".lora_A.weight" if k.endswith(".lora_A.weight") else ".lora_B.weight"
            if "proj_lora" in k:
                extras[f"{base}.proj_out{suffix}"] = value
            elif "qkv_lora" in k:
                if suffix == ".lora_A.weight":
                    for t in _XLABS_SINGLE_QKV_TARGETS:
                        extras[f"{base}.{t}{suffix}"] = value
                else:
                    for t, chunk in zip(_XLABS_SINGLE_QKV_TARGETS, torch.chunk(value, 3, dim=0)):
                        extras[f"{base}.{t}{suffix}"] = chunk
            continue

        # Double block: rename to canonical BFL-style; shared converter handles the QKV split.
        for old, new in _XLABS_DOUBLE_RENAMES.items():
            k = k.replace(old, new)
        canonical[k] = value

    return canonical, extras


# ============================================================================
# Stage 2d: Kohya normalizer (sd-scripts and mixture variants)
# ============================================================================
# Kohya keys collapse all dots into underscores in the module path, then append
# .lora_down/.lora_up/.alpha. We invert this with explicit per-suffix tables
# (the original-name underscore <-> dot mapping isn't recoverable by rule), then
# apply alpha-driven scaling so canonical tensors are pre-scaled.

_KOHYA_DOUBLE_SUFFIXES = {
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
}
_KOHYA_SINGLE_SUFFIXES = {
    "linear1": "linear1",
    "linear2": "linear2",
    "modulation_lin": "modulation.lin",
}
_KOHYA_GLOBAL_SUFFIXES = {
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


def _custom_replace(key, substrings):
    """Replace dots with underscores in `key` up to the first occurrence of any substring."""
    pattern = "(" + "|".join(re.escape(s) for s in substrings) + ")"
    match = re.search(pattern, key)
    if not match:
        return key.replace(".", "_")
    boundary = match.start() - 1 if match.start() > 0 and key[match.start() - 1] == "." else match.start()
    return key[:boundary].replace(".", "_") + key[boundary:]


def _kohya_pre_filter(state_dict):
    """Drop Kohya keys we don't support (with logging), then normalize key prefixes."""
    state_dict = {k.replace("diffusion_model.", "lora_unet_"): v for k, v in state_dict.items()}

    drop_specs = [
        (lambda k: "position_embedding" in k, "position_embedding", "position_embedding"),
        (lambda k: ".diff_b" in k and k.startswith("lora_unet_"), ".diff_b", "diff_b"),
        (lambda k: ".norm" in k and ".diff" in k, ".diff", "diff"),
    ]
    for predicate, marker, label in drop_specs:
        if not any(predicate(k) for k in state_dict):
            continue
        if state_dict_all_zero(state_dict, marker):
            logger.info(
                f"The `{label}` LoRA params are all zeros which make them ineffective. "
                "So, we will purge them out of the current state dict to make loading possible."
            )
        else:
            logger.info(
                f"`{label}` keys found in the state dict are currently unsupported and will be filtered out. "
                "Open an issue if this is a problem - https://github.com/huggingface/diffusers/issues/new."
            )
        state_dict = {k: v for k, v in state_dict.items() if not predicate(k)}

    # Some keys come with dots in the prefix; collapse them up to lora_A/lora_B/alpha.
    limit = ["lora_A", "lora_B"]
    if any("alpha" in k for k in state_dict):
        limit.append("alpha")
    state_dict = {_custom_replace(k, limit): v for k, v in state_dict.items() if k.startswith("lora_unet_")}

    return state_dict


def _kohya_canonical_path(stub):
    """Map a Kohya stub like 'double_blocks_0_img_attn_qkv' to BFL-style 'double_blocks.0.img_attn.qkv'."""
    m = re.match(r"double_blocks_(\d+)_(.+)$", stub)
    if m:
        i, suffix = m.group(1), m.group(2)
        bfl = _KOHYA_DOUBLE_SUFFIXES.get(suffix)
        return f"double_blocks.{i}.{bfl}" if bfl else None

    m = re.match(r"single_blocks_(\d+)_(.+)$", stub)
    if m:
        i, suffix = m.group(1), m.group(2)
        bfl = _KOHYA_SINGLE_SUFFIXES.get(suffix)
        return f"single_blocks.{i}.{bfl}" if bfl else None

    return _KOHYA_GLOBAL_SUFFIXES.get(stub)


def _normalize_kohya(state_dict):
    state_dict = _kohya_pre_filter(state_dict)

    # Mixture variant has its own prefix (lora_transformer_*); dispatch separately.
    has_mixture = any(
        k.startswith("lora_transformer_") and ("lora_down" in k or "lora_up" in k or "alpha" in k) for k in state_dict
    )
    if has_mixture:
        return {}, _convert_mixture(state_dict)

    # Group keys per Kohya module (lora_unet_<stub>) so we can apply alpha
    # scaling, then rewrite to canonical names.
    groups = {}  # stub -> {"lora_A": full_key, "lora_B": ..., "alpha": ...}
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

        bfl = _kohya_canonical_path(stub)
        if bfl is None:
            logger.warning(f"Unsupported Kohya key: lora_unet_{stub}")
            continue
        canonical[f"{bfl}.lora_A.weight"] = down
        canonical[f"{bfl}.lora_B.weight"] = up

    if state_dict:
        logger.warning(f"Unsupported keys after Kohya normalization: {list(state_dict.keys())}")

    return canonical, {}


# ----------------------------------------------------------------------------
# Mixture variant (Kohya-trained but using lora_transformer_* keys)
# ----------------------------------------------------------------------------


def _convert_mixture(state_dict):
    """Convert Kohya mixture-format LoRA directly to diffusers keys."""
    new_state_dict = {}

    def emit(orig, diffusers_key):
        down = state_dict.pop(f"{orig}.lora_A.weight")
        up = state_dict.pop(f"{orig}.lora_B.weight")
        alpha = state_dict.pop(f"{orig}.alpha")
        rank = down.shape[0]
        d_scale, u_scale = _kohya_scale(alpha, rank)
        new_state_dict[f"{diffusers_key}.lora_A.weight"] = down * d_scale
        new_state_dict[f"{diffusers_key}.lora_B.weight"] = up * u_scale

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

        emit(k, diffusers_key)

    leftover = [k for k in state_dict if not k.startswith("lora_unet_")]
    if leftover:
        logger.warning(f"Unsupported mixture keys ignored: {leftover}")

    return {f"transformer.{k}": v for k, v in new_state_dict.items()}


# ============================================================================
# Top-level dispatch
# ============================================================================


_NORMALIZERS = {
    "bfl": _normalize_bfl,
    "kontext": _normalize_kontext,
    "xlabs": _normalize_xlabs,
    "kohya": _normalize_kohya,
}


def map_lora_to_diffusers(state_dict, **kwargs):
    """Convert a Flux LoRA state_dict from any supported format to diffusers naming.

    Suffix normalization (lora_down/up -> lora_A/B) is run by
    ``LoRAMappingMixin.map_lora_to_diffusers`` before this is dispatched.
    """
    # Already-converted (peft) state dicts: keep only the transformer.* keys.
    if any(k.startswith("transformer.") for k in state_dict):
        return {k: v for k, v in state_dict.items() if k.startswith("transformer.")}

    fmt = FluxTransformerLoRAMixin._detect_lora_format(state_dict)
    if fmt is None or fmt not in _NORMALIZERS:
        raise ValueError(
            f"Unable to determine format of LoRA weights. Supported formats are: {FluxTransformerLoRAMixin._lora_format_keys.keys()}"
        )

    canonical, extras = _NORMALIZERS[fmt](state_dict)
    converted = _map_lora_to_diffusers(canonical) if canonical else {}
    return {**converted, **extras}


class FluxTransformerLoRAMixin(LoRAMappingMixin):
    """Mixin providing Flux-specific LoRA format detection and conversion."""

    _lora_format_keys: dict[str, set[str]] = {
        "kohya": {"lora_unet_double_blocks_", "lora_unet_single_blocks_"},
        "xlabs": {".processor.qkv_lora", ".processor.proj_lora"},
        "bfl": {"time_in.in_layer.lora_A", "double_blocks.0.img_mod.lin.lora_A"},
        "kontext": {"base_model.model.double_blocks"},
    }

    _map_lora_to_diffusers = staticmethod(map_lora_to_diffusers)
