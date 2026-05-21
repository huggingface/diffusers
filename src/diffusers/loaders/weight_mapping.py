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

A model declares its mapping in a :class:`WeightMappingHandler` instance (typically in its ``weight_mapping.py``
module). The ``@register_metadata`` decorator bundles it into the model's ``ModelMetadata``, reachable as
``cls._metadata._weight_mapping``. Internal call sites go through ``cls._metadata._weight_mapping.X`` (e.g.
``cls._metadata._weight_mapping.normalize_checkpoint_keys(state_dict)``) instead of flattening the methods onto the
model class itself.

The :meth:`WeightMappingHandler.apply_transforms` helper drives the forward direction from a single declarative table —
see ``models/transformers/flux/weight_mapping.py`` for an example.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..utils import logging


logger = logging.get_logger(__name__)


@dataclass
class WeightMappingHandler:
    """Composition-style holder for a model class's weight-mapping configuration and helpers.

    Attached to ``cls._metadata._weight_mapping`` by :meth:`ModelMetadata._register`. Owns all the data (available
    configs, prefixes, rename patterns, converter callables) and all the methods (rename, detect, normalize) for
    single-file checkpoint loading. Internal callers reach it via ``cls._metadata._weight_mapping.X``.

    Attributes:
        checkpoint_keys: Distinctive keys whose presence indicates the checkpoint is in the original
            (pre-diffusers) format.
        checkpoint_key_prefixes: Foreign prefixes (e.g. ``["model.diffusion_model."]``) the handler will strip via
            :meth:`normalize_checkpoint_keys`. Set this on prefix-only models to skip registering a
            ``map_to_diffusers_fn`` callable.
        rename_patterns: Default rename patterns shared between forward and reverse conversions (consumed by
            :meth:`apply_transforms`).
        available_configs:
            Map of short config name to hub repo id (e.g. ``{"flux-dev": "black-forest-labs/FLUX.1-dev"}``).
        default_config: Config name (key into ``available_configs``) used when ``detect_config_fn`` is
            unregistered or returns ``None``.
        default_subfolder: Default ``subfolder`` to use when fetching configs (e.g. ``"transformer"``).
        map_to_diffusers_fn: Callable ``(state_dict, **kwargs) -> state_dict`` performing full key conversion.
            ``None`` for prefix-only models.
        map_from_diffusers_fn: Reverse callable (diffusers → original format).
        detect_config_fn: ``(handler, state_dict) -> Optional[str]`` returning a config name from
            ``available_configs``, or ``None`` to fall back to ``default_config``.
    """

    checkpoint_keys: set = field(default_factory=set)
    checkpoint_key_prefixes: list = field(default_factory=list)
    rename_patterns: dict = field(default_factory=dict)
    available_configs: dict = field(default_factory=dict)
    default_config: Optional[str] = None
    default_subfolder: str = "transformer"
    map_to_diffusers_fn: Optional[Callable] = None
    map_from_diffusers_fn: Optional[Callable] = None
    detect_config_fn: Optional[Callable] = None

    # ---- single-file capability ----

    @property
    def supports_single_file(self) -> bool:
        """Whether ``from_single_file(path)`` works for this model with no extra arguments.

        Requires ``default_config`` to be set so config resolution always succeeds (with or without a successful
        ``detect_config_fn`` call). Models that declare only ``available_configs`` still load via
        ``from_single_file(path, config=...)``, but they don't auto-resolve and so don't count as supporting. Key
        normalization is all no-op-safe; the architecture-resolution step is the only hard requirement.
        """
        return self.default_config is not None

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

        Dispatches to ``self.detect_config_fn(self, state_dict)``. If unregistered, returns ``None`` so the caller can
        fall back to ``self.default_config``.
        """
        if self.detect_config_fn is None:
            return None
        return self.detect_config_fn(self, state_dict)

    def get_model_config(self, state_dict: dict) -> str:
        """Resolve the hub repo id whose config best matches this checkpoint.

        Resolution order:
            1. Run ``detect_config(state_dict)`` (if a detector is registered).
            2. If detection returns ``None``, fall back to ``default_config`` and warn (since the user is now getting a
               config that may not match the checkpoint shape).
            3. Look up the chosen name in ``available_configs`` to get the hub repo id.
        """
        detected = self.detect_config(state_dict)
        if detected is None and self.default_config is not None and self.detect_config_fn is not None:
            logger.warning(
                f"Could not auto-detect a config for this checkpoint; falling back to default_config="
                f"'{self.default_config}' ({self.available_configs.get(self.default_config)}). "
                f"If this is the wrong architecture, pass `config=<hub-repo-id>` to `from_single_file(...)` "
                f"explicitly. Known configs: {sorted(self.available_configs)}."
            )
        config_name = detected or self.default_config
        if config_name is None:
            available = sorted(self.available_configs) or "<none registered>"
            has_detector = self.detect_config_fn is not None
            raise ValueError(
                "Could not determine which config to load for this checkpoint.\n"
                "\n"
                f"  Detection: {'registered, but returned None for this state_dict' if has_detector else 'no detect_config_fn registered'}\n"
                "  Default config: not set\n"
                f"  Available configs: {available}\n"
                "\n"
                "To fix this, either:\n"
                '  - pass `config="<hub-repo-id>"` to `from_single_file(...)` to skip auto-detection, OR\n'
                "  - update the model's `WeightMappingHandler` to set `detect_config_fn` (returns a name from "
                "`available_configs`), and/or set `default_config` to a name in `available_configs`."
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
        if self.map_to_diffusers_fn is None:
            return state_dict
        return self.map_to_diffusers_fn(state_dict, **kwargs)

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
        if self.map_from_diffusers_fn is None:
            raise NotImplementedError("No `map_from_diffusers_fn` callable registered for this model.")
        return self.map_from_diffusers_fn(state_dict, **kwargs)

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
