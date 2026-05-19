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
    _available_configs: dict = {}
    _map_to_diffusers = None
    _map_from_diffusers = None
    _detect_config_fn = None
    _default_config: Optional[str] = None
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
    def _detect_config(cls, state_dict: dict) -> Optional[str]:
        """Detect which config name from ``_available_configs`` matches this state_dict.

        Dispatches to ``cls._detect_config_fn`` (mirrored from the model's metadata). If no detector is registered,
        returns ``None`` so the caller can fall back to ``_default_config``.
        """
        if cls._detect_config_fn is None:
            return None
        return cls._detect_config_fn(cls, state_dict)

    @classmethod
    def _get_model_config(cls, state_dict: dict) -> str:
        """Resolve the hub repo id whose config best matches this checkpoint.

        Resolution order:
            1. Run ``_detect_config_fn`` (if registered) against the state_dict; it should return a config name from
               ``_available_configs`` or ``None``.
            2. If detection returns ``None`` (or no detector is registered), fall back to ``_default_config``.
            3. Look up the chosen name in ``_available_configs`` to get the hub repo id.
        """
        config_name = cls._detect_config(state_dict) or cls._default_config
        if config_name is None:
            available = sorted(cls._available_configs) or "<none registered>"
            has_detector = cls._detect_config_fn is not None
            raise ValueError(
                f"`{cls.__name__}.from_single_file` could not determine which config to load for this checkpoint.\n"
                f"\n"
                f"  Detection: {'registered, but returned None for this state_dict' if has_detector else 'no `_detect_config_fn` registered'}\n"
                f"  Default config: not set\n"
                f"  Available configs: {available}\n"
                f"\n"
                f"To fix this, either:\n"
                f'  - pass `config="<hub-repo-id>"` to `from_single_file(...)` to skip auto-detection, OR\n'
                f"  - update `{cls.__name__}`'s `WeightMappingMetadata` to register a `_detect_config_fn` that "
                f"returns a name from `_available_configs`, and/or set `_default_config` to a name in "
                f"`_available_configs`."
            )
        if config_name not in cls._available_configs:
            raise ValueError(
                f"{cls.__name__}: resolved config name '{config_name}' is not a key of `_available_configs` "
                f"(available: {sorted(cls._available_configs)})."
            )
        return cls._available_configs[config_name]

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
