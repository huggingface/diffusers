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

"""Reusable infrastructure for converting model checkpoints between original
and diffusers naming conventions.

A model defines its mapping by subclassing :class:`WeightMappingMixin` and
populating the class attributes (`_rename_patterns`, `_checkpoint_keys`, etc.)
plus assigning ``_map_to_diffusers`` / ``_map_from_diffusers`` callables.

The :meth:`apply_transforms` helper drives the forward direction from a single
declarative table — see ``models/transformers/flux/weight_mapping.py`` for an
example.
"""


class WeightMappingMixin:
    """
    Base mixin providing utilities for checkpoint weight mapping and conversion.

    Subclasses should define:
    - _checkpoint_key_prefixes: List of key prefixes to strip (e.g., ["model.diffusion_model."])
    - _checkpoint_keys: Set of keys to identify compatible checkpoints
    - _rename_patterns: Dict of substring replacements for key renaming
    - _model_variants: Dict mapping variant names to config repos
    - _map_to_diffusers: Function to convert original format to diffusers format
    - _map_from_diffusers: Function to convert diffusers format to original format
    """

    _checkpoint_key_prefixes: list[str] = []
    _checkpoint_keys: set[str] = set()
    _rename_patterns: dict[str, str] = {}
    _model_variants: dict[str, str] = {}
    _map_to_diffusers = None
    _map_from_diffusers = None

    @staticmethod
    def _rename_key(key: str, patterns: dict[str, str]) -> str:
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
        keys = set(state_dict.keys())
        return bool(cls._checkpoint_keys & keys)

    @classmethod
    def _detect_model_variant(cls, state_dict: dict) -> str | None:
        """Detect which model variant a state_dict belongs to. Subclasses should override."""
        raise NotImplementedError(f"{cls.__name__} does not implement _detect_model_variant")

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
          - ``source``: substring matched against each key (with surrounding dots,
            e.g. ``".img_attn.qkv."``); the first matching entry wins.
          - ``targets``: list of substrings substituted for ``source`` to build the
            output keys. ``len(targets)`` is the fan-out (1 for a unary transform,
            >1 for a split).
          - ``forward_fn(value, **ctx) -> list[tensor]`` returns one tensor per
            target. (``reverse_fn`` is reserved for a future
            ``apply_reverse_transforms`` driver.)

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
            raise NotImplementedError(f"{cls.__name__} does not define _map_to_diffusers")
        return cls._map_to_diffusers(state_dict, **kwargs)

    @classmethod
    def map_from_diffusers(cls, state_dict: dict, **kwargs) -> dict:
        """Convert state_dict from diffusers format to original format."""
        if cls._map_from_diffusers is None:
            raise NotImplementedError(f"{cls.__name__} does not define _map_from_diffusers")
        return cls._map_from_diffusers(state_dict, **kwargs)
