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
module) and assigns it to the class as ``_weight_mapping = FLUX_WEIGHT_MAPPING``. Internal call sites go through
``cls._weight_mapping.X`` (e.g. ``cls._weight_mapping.normalize_state_dict_keys(state_dict)``) instead of flattening
the methods onto the model class itself.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..utils import logging


logger = logging.get_logger(__name__)


# Foreign key prefixes seen across multiple model families' single-file checkpoints. Stripping these is
# universally safe (no model uses them as native diffusers keys), so the handler defaults to removing them on
# every load. Models with additional, family-specific prefixes can extend or override
# ``prefixes_to_remove`` on their handler.
PREFIXES_TO_REMOVE: list[str] = [
    "model.diffusion_model.",
]


@dataclass
class WeightMappingHandler:
    """Composition-style holder for a model class's weight-mapping configuration and helpers.

    Attached as the ``_weight_mapping`` class attribute on :class:`ModelMixin` (overridden per-model). Owns all
    the data (available configs, prefixes, rename patterns, converter callables) and all the methods (rename,
    detect, normalize) for single-file checkpoint loading. Internal callers reach it via ``cls._weight_mapping.X``.

    Attributes:
        original_format_keys: Distinctive keys whose presence indicates the state_dict is in the original
            (pre-diffusers) format. Used by :meth:`is_original_format` to decide whether key conversion is
            needed.
        prefixes_to_remove: Foreign prefixes (e.g. ``["model.diffusion_model."]``) the handler will strip via
            :meth:`normalize_state_dict_keys`. Defaults to the shared :data:`PREFIXES_TO_REMOVE` list — most models
            only need that. Extend it for family-specific wrappers; prefix-only models can rely on the default and skip
            registering a ``map_to_diffusers_fn`` callable.
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

    original_format_keys: set = field(default_factory=set)
    prefixes_to_remove: list = field(default_factory=lambda: list(PREFIXES_TO_REMOVE))
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
        """Check if state_dict is in the original (pre-diffusers) format by presence of a known marker key.

        Returns ``True`` only when a registered ``original_format_keys`` entry is observed in the state_dict.
        Returning ``False`` means "no positive evidence of original format" — empty / unrelated / unknown
        state_dicts all fall here. Callers treat ``False`` as "proceed with diffusers-native keys."
        """
        if not self.original_format_keys:
            return False
        return bool(self.original_format_keys & set(state_dict.keys()))

    def normalize_state_dict_keys(self, state_dict: dict) -> dict:
        """Strip known foreign prefixes (e.g. ``model.diffusion_model.``) from state_dict keys."""
        if not self.prefixes_to_remove:
            return state_dict
        result = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in self.prefixes_to_remove:
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
        """Resolve the hub repo id whose config best matches this state_dict.

        Resolution order:
            1. Run ``detect_config(state_dict)`` (if a detector is registered).
            2. If detection returns ``None``, fall back to ``default_config`` and warn (since the user is now getting a
               config that may not match the state_dict shape).
            3. Look up the chosen name in ``available_configs`` to get the hub repo id.
        """
        detected = self.detect_config(state_dict)
        if detected is None and self.default_config is not None and self.detect_config_fn is not None:
            logger.warning(
                f"Could not auto-detect a config for this state_dict; falling back to default_config="
                f"'{self.default_config}' ({self.available_configs.get(self.default_config)}). "
                f"If this is the wrong architecture, pass `config=<hub-repo-id>` to `from_single_file(...)` "
                f"explicitly. Known configs: {sorted(self.available_configs)}."
            )
        config_name = detected or self.default_config
        if config_name is None:
            available = sorted(self.available_configs) or "<none registered>"
            has_detector = self.detect_config_fn is not None
            raise ValueError(
                "Could not determine which config to load for this state_dict.\n"
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
        the prefix-only path (via :meth:`normalize_state_dict_keys`) in that case.
        """
        if self.map_to_diffusers_fn is None:
            return state_dict
        return self.map_to_diffusers_fn(state_dict, **kwargs)

    def maybe_convert_state_dict(self, model, state_dict: dict) -> dict:
        """Bring ``state_dict`` to diffusers naming if it isn't already. Two phases:

        1. :meth:`normalize_state_dict_keys` — strip known prefixes (idempotent; no-op if none registered).
        2. :meth:`map_to_diffusers` — full key conversion, only invoked if step 1 alone didn't make the keys match the
           model's. Skipped (no-op) if no converter callable was registered.

        Idempotent overall: calling twice produces the same result as calling once.
        """
        state_dict = self.normalize_state_dict_keys(state_dict)
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        # If the model's keys are a (strict) subset of the state_dict's, the rest is extras we'll surface later
        # via the missing/unexpected keys report — but no key-renaming pass is needed.
        if model_keys.issubset(state_dict_keys):
            return state_dict
        return self.map_to_diffusers(state_dict)

    def map_from_diffusers(self, state_dict: dict, **kwargs) -> dict:
        """Convert state_dict from diffusers format to original format."""
        if self.map_from_diffusers_fn is None:
            raise NotImplementedError("No `map_from_diffusers_fn` callable registered for this model.")
        return self.map_from_diffusers_fn(state_dict, **kwargs)
