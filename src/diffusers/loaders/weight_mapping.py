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

A model declares its mapping in a :class:`WeightMappingMetadata` instance (typically in its ``weight_mapping.py``
module). The ``@register_metadata`` decorator instantiates a :class:`WeightMappingHandler` from that metadata and
attaches it to the model class as ``cls._weight_mapping``. Internal call sites then go through
``self._weight_mapping.X`` (e.g. ``self._weight_mapping.normalize_checkpoint_keys(state_dict)``) instead of flattening
the methods onto the model class itself.

The :meth:`WeightMappingHandler.apply_transforms` helper drives the forward direction from a single declarative table —
see ``models/transformers/flux/weight_mapping.py`` for an example.
"""

from typing import Callable, Optional


class WeightMappingHandler:
    """Composition-style holder for a model class's weight-mapping configuration and helpers.

    Instances are attached to model classes as ``cls._weight_mapping`` by ``WeightMappingMetadata._register``. Owns all
    the data (available configs, prefixes, rename patterns, converter callables) and all the methods (rename, detect,
    normalize) that the legacy ``WeightMappingMixin`` flattened onto the model class. The model class itself no longer
    carries those attributes; access is always via ``cls._weight_mapping.X`` / ``self._weight_mapping.X``.
    """

    def __init__(
        self,
        *,
        checkpoint_keys: Optional[set] = None,
        checkpoint_key_prefixes: Optional[list] = None,
        rename_patterns: Optional[dict] = None,
        available_configs: Optional[dict] = None,
        default_config: Optional[str] = None,
        default_subfolder: str = "transformer",
        map_to_diffusers: Optional[Callable] = None,
        map_from_diffusers: Optional[Callable] = None,
        detect_config_fn: Optional[Callable] = None,
    ):
        self.checkpoint_keys = checkpoint_keys or set()
        self.checkpoint_key_prefixes = checkpoint_key_prefixes or []
        self.rename_patterns = rename_patterns or {}
        self.available_configs = available_configs or {}
        self.default_config = default_config
        self.default_subfolder = default_subfolder
        self._map_to_diffusers_fn = map_to_diffusers
        self._map_from_diffusers_fn = map_from_diffusers
        self._detect_config_fn = detect_config_fn

    # ---- single-file capability ----

    @property
    def supports_single_file(self) -> bool:
        """Whether the model has enough metadata to load from a single-file checkpoint.

        Requires ``available_configs`` (so a config repo can be resolved) plus either a converter callable
        (``_map_to_diffusers_fn``) or a non-empty ``checkpoint_key_prefixes`` (declarative prefix-only path).
        """
        has_normalizer = self._map_to_diffusers_fn is not None or bool(self.checkpoint_key_prefixes)
        return bool(self.available_configs) and has_normalizer

    # ---- key utilities ----

    @staticmethod
    def rename_key(key: str, patterns: dict) -> str:
        """Apply rename patterns to a key (first match wins per substring)."""
        for old, new in patterns.items():
            key = key.replace(old, new)
        return key

    def is_original_format(self, state_dict: dict) -> bool:
        """Check if state_dict is in original (non-diffusers) format by presence of a known foreign key."""
        if not self.checkpoint_keys:
            return False
        return bool(self.checkpoint_keys & set(state_dict.keys()))

    def normalize_checkpoint_keys(self, state_dict: dict) -> dict:
        """Strip known foreign prefixes (e.g. ``model.diffusion_model.``) from state_dict keys."""
        if not self.checkpoint_key_prefixes:
            return state_dict
        result = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in self.checkpoint_key_prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    break
            result[new_key] = value
        return result

    # ---- config resolution ----

    def detect_config(self, state_dict: dict) -> Optional[str]:
        """Detect which config name from ``available_configs`` matches this state_dict.

        Dispatches to ``self._detect_config_fn(self, state_dict)``. If unregistered, returns ``None`` so the caller can
        fall back to ``self.default_config``.
        """
        if self._detect_config_fn is None:
            return None
        return self._detect_config_fn(self, state_dict)

    def get_model_config(self, state_dict: dict) -> str:
        """Resolve the hub repo id whose config best matches this checkpoint.

        Resolution order:
            1. Run ``detect_config(state_dict)`` (if a detector is registered).
            2. If detection returns ``None``, fall back to ``default_config``.
            3. Look up the chosen name in ``available_configs`` to get the hub repo id.
        """
        config_name = self.detect_config(state_dict) or self.default_config
        if config_name is None:
            available = sorted(self.available_configs) or "<none registered>"
            has_detector = self._detect_config_fn is not None
            raise ValueError(
                "Could not determine which config to load for this checkpoint.\n"
                "\n"
                f"  Detection: {'registered, but returned None for this state_dict' if has_detector else 'no detect_config_fn registered'}\n"
                "  Default config: not set\n"
                f"  Available configs: {available}\n"
                "\n"
                "To fix this, either:\n"
                '  - pass `config="<hub-repo-id>"` to `from_single_file(...)` to skip auto-detection, OR\n'
                "  - update the model's `WeightMappingMetadata` to register a `_detect_config_fn` that returns a "
                "name from `_available_configs`, and/or set `_default_config` to a name in `_available_configs`."
            )
        if config_name not in self.available_configs:
            raise ValueError(
                f"Resolved config name '{config_name}' is not a key of `available_configs` "
                f"(available: {sorted(self.available_configs)})."
            )
        return self.available_configs[config_name]

    # ---- conversion ----

    def map_to_diffusers(self, state_dict: dict, **kwargs) -> dict:
        """Convert state_dict from original format to diffusers format.

        No-op (returns ``state_dict`` unchanged) if no converter callable is registered; callers are expected to use
        the prefix-only path (via :meth:`normalize_checkpoint_keys`) in that case.
        """
        if self._map_to_diffusers_fn is None:
            return state_dict
        return self._map_to_diffusers_fn(state_dict, **kwargs)

    def maybe_convert_state_dict(self, model, state_dict: dict) -> dict:
        """Bring ``state_dict`` to diffusers naming if it isn't already. Two phases:

        1. :meth:`normalize_checkpoint_keys` — strip known prefixes (idempotent; no-op if none registered).
        2. :meth:`map_to_diffusers` — full key conversion, only invoked if step 1 alone didn't make the keys match the
           model's. Skipped (no-op) if no converter callable was registered.

        Idempotent overall: calling twice produces the same result as calling once.
        """
        state_dict = self.normalize_checkpoint_keys(state_dict)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        # If the model's keys are a (strict) subset of the checkpoint's, the rest is extras we'll surface later
        # via the missing/unexpected keys report — but no key-renaming pass is needed.
        if model_keys.issubset(ckpt_keys):
            return state_dict
        return self.map_to_diffusers(state_dict)

    def map_from_diffusers(self, state_dict: dict, **kwargs) -> dict:
        """Convert state_dict from diffusers format to original format."""
        if self._map_from_diffusers_fn is None:
            raise NotImplementedError("No `_map_from_diffusers` callable registered for this model.")
        return self._map_from_diffusers_fn(state_dict, **kwargs)

    # ---- driver for declarative transforms ----

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
                        new_key = WeightMappingHandler.rename_key(key.replace(source, target), rename_patterns)
                        out[new_key] = tensor
                    break
            else:
                out[WeightMappingHandler.rename_key(key, rename_patterns)] = value
        return out
