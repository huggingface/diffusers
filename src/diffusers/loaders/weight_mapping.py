# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Reusable infrastructure for converting model checkpoints between original and diffusers naming conventions.

A model declares its mapping in a ``WeightMappingMetadata`` instance (typically in its ``weight_mapping.py`` module)
and attaches it via ``@register_model_metadata(weight_mapping=...)``. This mixin supplies the generic dispatch methods
that read from that metadata.

The :meth:`apply_transforms` helper drives the forward direction from a single declarative table — see
``models/transformers/flux/weight_mapping.py`` for an example.
"""

from typing import Optional


class WeightMappingMixin:
    """
    Base mixin providing utilities for checkpoint weight mapping and conversion.

    Per-model configuration (rename patterns, format-identifying keys, conversion callables, etc.) lives in the model's
    registered ``WeightMappingMetadata`` — declared in the model's ``weight_mapping.py`` and attached via
    ``@register_model_metadata``. This mixin just supplies the dispatch methods.
    """

    # Default class-attribute values; populated per-model by ``register_model_metadata``.
    _checkpoint_key_prefixes: list = []
    _checkpoint_keys: set = set()
    _rename_patterns: dict = {}
    _model_variants: dict = {}
    _map_to_diffusers = None
    _map_from_diffusers = None
    _detect_model_variant_fn = None
    _default_subfolder: str = "transformer"

    @staticmethod
    def _rename_key(key: str, patterns: dict) -> str:
        """Apply rename patterns to a key."""
        for old, new in patterns.items():
            key = key.replace(old, new)
        return key

    @classmethod
    def _normalize_checkpoint_keys(cls, state_dict: dict) -> dict:
        """Strip known prefixes from state_dict keys."""
        if not cls._checkpoint_key_prefixes:
            return state_dict
        result = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in cls._checkpoint_key_prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    break
            result[new_key] = value
        return result

    @classmethod
    def _is_original_format(cls, state_dict: dict) -> bool:
        """Check if state_dict is in original (non-diffusers) format."""
        if not cls._checkpoint_keys:
            return False
        return bool(cls._checkpoint_keys & set(state_dict.keys()))

    @classmethod
    def _detect_model_variant(cls, state_dict: dict) -> Optional[str]:
        """Detect which model variant a state_dict belongs to.

        Dispatches to ``cls._detect_model_variant_fn`` (mirrored from the model's metadata); raises if no detector is
        registered.
        """
        if cls._detect_model_variant_fn is None:
            raise NotImplementedError(
                f"{cls.__name__} did not register a `_detect_model_variant_fn` in its WeightMappingMetadata."
            )
        return cls._detect_model_variant_fn(cls, state_dict)

    @classmethod
    def _get_model_config(cls, state_dict: dict) -> str:
        """Get the default config repo for the detected variant."""
        variant = cls._detect_model_variant(state_dict)
        if variant is None:
            raise ValueError(f"Could not detect model variant from state_dict. Expected keys: {cls._checkpoint_keys}")
        return cls._model_variants[variant]

    @staticmethod
    def apply_transforms(state_dict, transforms, rename_patterns, **ctx):
        """Drive a forward state-dict conversion from a list of (source, targets, fn) entries.

        Each entry is a tuple ``(source, targets, forward_fn, reverse_fn)``:
          - ``source``: substring matched against each key (with surrounding dots, e.g. ``".img_attn.qkv."``); the
            first matching entry wins.
          - ``targets``: list of substrings substituted for ``source`` to build the output keys. ``len(targets)`` is
            the fan-out (1 for a unary transform, >1 for a split).
          - ``forward_fn(value, **ctx) -> list[tensor]`` returns one tensor per target. (``reverse_fn`` is reserved for
            a future ``apply_reverse_transforms`` driver.)

        Keys that match no transform get their dots renamed via ``rename_patterns``.
        """
        out = {}
        for key, value in state_dict.items():
            for source, targets, forward_fn, _ in transforms:
                if source in key:
                    tensors = forward_fn(value, **ctx)
                    for target, tensor in zip(targets, tensors):
                        new_key = WeightMappingMixin._rename_key(key.replace(source, target), rename_patterns)
                        out[new_key] = tensor
                    break
            else:
                out[WeightMappingMixin._rename_key(key, rename_patterns)] = value
        return out

    @classmethod
    def map_to_diffusers(cls, state_dict: dict, **kwargs) -> dict:
        """Convert state_dict from original format to diffusers format."""
        if cls._map_to_diffusers is None:
            raise NotImplementedError(
                f"{cls.__name__} did not register a `_map_to_diffusers` in its WeightMappingMetadata."
            )
        return cls._map_to_diffusers(state_dict, **kwargs)

    @classmethod
    def map_from_diffusers(cls, state_dict: dict, **kwargs) -> dict:
        """Convert state_dict from diffusers format to original format."""
        if cls._map_from_diffusers is None:
            raise NotImplementedError(
                f"{cls.__name__} did not register a `_map_from_diffusers` in its WeightMappingMetadata."
            )
        return cls._map_from_diffusers(state_dict, **kwargs)
