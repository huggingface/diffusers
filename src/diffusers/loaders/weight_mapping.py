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

from contextlib import nullcontext
from typing import Any, Optional

import torch
from huggingface_hub.utils import validate_hf_hub_args
from typing_extensions import Self

from .. import __version__
from ..utils import HUB_KWARGS, deprecate, is_accelerate_available, logging


logger = logging.get_logger(__name__)


# Foreign key prefixes seen across multiple model families' single-file checkpoints. Stripping these is
# universally safe (no model uses them as native diffusers keys), so the handler strips them on every load.
# Models with additional, family-specific prefixes can override the internal ``_prefixes_to_remove`` class
# attribute on their handler subclass.
PREFIXES_TO_REMOVE: list[str] = [
    "model.diffusion_model.",
]


class WeightMappingHandler:
    """Composition-style holder for a model class's weight-mapping configuration and helpers.

    Attached as the ``_weight_mapping`` class attribute on models mixing in :class:`WeightMappingMixin`. Owns the data
    (available configs, prefixes, distinctive keys) and the methods (rename, detect, convert, normalize) for
    single-file checkpoint loading. Internal callers reach it via ``cls._weight_mapping.X``.

    Configured entirely by subclassing — a model declares a ``FooWeightMappingHandler(WeightMappingHandler)`` that sets
    the internal ``_``-prefixed class attributes below and overrides :meth:`map_to_diffusers`,
    :meth:`map_from_diffusers`, and/or :meth:`detect_config` as needed. Prefix-only models (no key renaming, no
    per-variant detection) only need to set ``_available_configs`` / ``_default_config``.

    Class attributes (override on a subclass; none are constructor arguments):
        _source_key_names: Distinctive marker keys whose presence indicates the state_dict is in the source
            (upstream, pre-diffusers) format. Used by :meth:`is_source_format` to decide whether key conversion is
            needed.
        _available_configs:
            Map of short config name to hub repo id (e.g. ``{"flux-dev": "black-forest-labs/FLUX.1-dev"}``).
        _default_config: Config name (key into ``_available_configs``) used when :meth:`detect_config` returns
        ``None``. _default_subfolder: Default ``subfolder`` to use when fetching configs (e.g. ``"transformer"``).
        _prefixes_to_remove: Foreign prefixes (default ``["model.diffusion_model."]``) stripped by
            :meth:`normalize_state_dict_keys`. Extend for family-specific wrappers.
    """

    _source_key_names: set = set()
    _available_configs: dict = {}
    _default_config: Optional[str] = None
    _default_subfolder: str = "transformer"
    _prefixes_to_remove: list = list(PREFIXES_TO_REMOVE)

    # ---- single-file capability ----

    @property
    def supports_single_file(self) -> bool:
        """Whether ``from_single_file(path)`` works for this model with no extra arguments.

        Requires ``_default_config`` to be set so config resolution always succeeds (with or without a successful
        :meth:`detect_config` call). Models that declare only ``_available_configs`` still load via
        ``from_single_file(path, config=...)``, but they don't auto-resolve and so don't count as supporting. Key
        normalization is all no-op-safe; the architecture-resolution step is the only hard requirement.
        """
        return self._default_config is not None

    # ---- key utilities ----

    @staticmethod
    def rename_key(key: str, patterns: dict) -> str:
        """Apply rename patterns to a key (first match wins per substring)."""
        for old, new in patterns.items():
            key = key.replace(old, new)
        return key

    def is_source_format(self, state_dict: dict) -> bool:
        """Check if state_dict is in the source (upstream, pre-diffusers) format by presence of a known marker key.

        Returns ``True`` only when a registered ``_source_key_names`` entry is observed in the state_dict. Returning
        ``False`` means "no positive evidence of source format" — empty / unrelated / unknown state_dicts all fall
        here. Callers treat ``False`` as "proceed with diffusers-native keys."
        """
        if not self._source_key_names:
            return False
        return bool(self._source_key_names & set(state_dict.keys()))

    def normalize_state_dict_keys(self, state_dict: dict) -> dict:
        """Strip known foreign prefixes (e.g. ``model.diffusion_model.``) from state_dict keys."""
        if not self._prefixes_to_remove:
            return state_dict
        result = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in self._prefixes_to_remove:
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    break
            result[new_key] = value
        return result

    # ---- config resolution ----

    def detect_config(self, state_dict: dict) -> Optional[str]:
        """Detect which config name from ``_available_configs`` matches this state_dict.

        Default implementation returns ``None`` so the caller falls back to ``_default_config``. Subclasses override to
        inspect the state_dict for architecture-distinguishing keys or shapes.
        """
        return None

    def get_model_config(self, state_dict: dict) -> str:
        """Resolve the hub repo id whose config best matches this state_dict.

        Resolution order:
            1. Run :meth:`detect_config` (subclasses override; base returns ``None``).
            2. If detection returns ``None`` but there was a genuine choice to make (more than one available config)
               and the state_dict is in source format, warn and fall back to ``_default_config`` (the user is getting a
               config that may not match the state_dict shape). A single-config handler falls back silently
               (unambiguous), and diffusers-format inputs skip the warning entirely (the fallback is expected there).
            3. Look up the chosen name in ``_available_configs`` to get the hub repo id.
        """
        detected = self.detect_config(state_dict)
        if (
            detected is None
            and self._default_config is not None
            and len(self._available_configs) > 1
            and self.is_source_format(state_dict)
        ):
            logger.warning(
                f"Could not auto-detect a config for this state_dict; falling back to default_config="
                f"'{self._default_config}' ({self._available_configs.get(self._default_config)}). "
                f"If this is the wrong architecture, pass `config=<hub-repo-id>` to `from_single_file(...)` "
                f"explicitly. Known configs: {sorted(self._available_configs)}."
            )
        config_name = detected or self._default_config
        if config_name is None:
            available = sorted(self._available_configs) or "<none registered>"
            raise ValueError(
                "Could not determine which config to load for this state_dict.\n"
                "\n"
                "  Default config: not set\n"
                f"  Available configs: {available}\n"
                "\n"
                "To fix this, either:\n"
                '  - pass `config="<hub-repo-id>"` to `from_single_file(...)` to skip auto-detection, OR\n'
                "  - subclass `WeightMappingHandler` and override `detect_config` (returns a name from "
                "`_available_configs`), and/or set `_default_config` to a name in `_available_configs`."
            )
        if config_name not in self._available_configs:
            raise ValueError(
                f"Resolved config name '{config_name}' is not a key of `_available_configs` "
                f"(available: {sorted(self._available_configs)})."
            )
        return self._available_configs[config_name]

    # ---- conversion ----

    def map_to_diffusers(self, state_dict: dict, **kwargs) -> dict:
        """Convert state_dict from original format to diffusers format.

        Default implementation is a no-op (returns ``state_dict`` unchanged) — prefix-only models rely on
        :meth:`normalize_state_dict_keys` and don't need to override. Models with real key renaming subclass and
        replace this method.
        """
        return state_dict

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
        """Convert state_dict from diffusers format to original format. Subclasses override."""
        raise NotImplementedError(
            f"`{type(self).__name__}` does not implement `map_from_diffusers`. Subclass `WeightMappingHandler` "
            "and override this method to enable diffusers → original conversion."
        )


class WeightMappingMixin:
    """Opt-in mixin that adds ``from_single_file`` and the matching metadata row to a model class.

    Inherit from this mixin **and** assign a :class:`WeightMappingHandler` instance to ``_weight_mapping`` on the model
    class to enable original-format checkpoint loading without a local ``config.json``::

        class FluxTransformer2DModel(ModelMixin, WeightMappingMixin, ...):
            _weight_mapping = FLUX_WEIGHT_MAPPING

    Models that haven't been ported keep using ``FromOriginalModelMixin`` from main.
    """

    _weight_mapping: WeightMappingHandler = WeightMappingHandler()

    @classmethod
    def _metadata(cls) -> dict[str, tuple[Any, str, str, str]]:
        from ..models.modeling_utils import DOCS_BASE

        rows: dict[str, tuple[Any, str, str, str]] = {}
        if cls._weight_mapping.supports_single_file:
            configs = sorted(cls._weight_mapping._available_configs)
            rows["_weight_mapping"] = (
                configs,
                ", ".join(configs),
                "Auto-resolvable configs for `from_single_file(path)` (no `config=` argument required).",
                f"{DOCS_BASE}/api/loaders/single_file",
            )
        return rows

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path_or_dict: str | None = None, **kwargs) -> Self:
        r"""
        Instantiate a model from pretrained weights saved in the original `.ckpt` or `.safetensors` format. The model
        is set in evaluation mode (`model.eval()`) by default. Available on classes that mix in
        :class:`WeightMappingMixin` and declare a ``_weight_mapping`` with ``default_config`` set.

        Parameters:
            pretrained_model_link_or_path_or_dict (`str`, *optional*):
                Can be either:
                    - A link to the `.safetensors` or `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A path to a local *file* containing the weights of the component model.
                    - A state dict containing the component model weights.
            config (`str`, *optional*):
                Repo id or local directory pointing at a diffusers-format config. If omitted, resolved automatically
                via ``cls._weight_mapping``.
            subfolder (`str`, *optional*):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype.
            force_download (`bool`, *optional*, defaults to `False`):
                Force re-downloading the weights and config.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Cache directory override.
            proxies (`Dict[str, str]`, *optional*):
                Proxy servers keyed by protocol/endpoint.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Only use locally cached files.
            token (`str` or *bool*, *optional*):
                HF auth token.
            revision (`str`, *optional*, defaults to `"main"`):
                Revision of the model on the Hub.
            low_cpu_mem_usage (`bool`, *optional*):
                Skip weight initialization for faster loading.
            disable_mmap (`bool`, *optional*, defaults to `False`):
                Disable mmap for safetensors loading.

        Example:
            ```python
            >>> from diffusers import FluxTransformer2DModel

            >>> ckpt_path = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
            >>> model = FluxTransformer2DModel.from_single_file(ckpt_path)
            ```
        """
        from ..models.model_loading_utils import _determine_device_map
        from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
        from ..quantizers import DiffusersAutoQuantizer
        from .single_file_utils import SingleFileComponentError, load_single_file_checkpoint

        _weight_mapping = cls._weight_mapping
        if not _weight_mapping.supports_single_file:
            raise ValueError(
                f"`{cls.__name__}.from_single_file` is not supported. "
                "The model's `WeightMappingHandler` must declare `default_config` (a key into "
                "`available_configs`) so we can resolve which architecture to instantiate when the user "
                "doesn't pass `config=` explicitly. Use `from_pretrained` if the model is already in "
                "diffusers format."
            )
        default_subfolder = _weight_mapping._default_subfolder

        pretrained_model_link_or_path = kwargs.get("pretrained_model_link_or_path", None)
        if pretrained_model_link_or_path is not None:
            deprecation_message = (
                "Please use `pretrained_model_link_or_path_or_dict` argument instead for model classes"
            )
            deprecate("pretrained_model_link_or_path", "1.0.0", deprecation_message)
            pretrained_model_link_or_path_or_dict = pretrained_model_link_or_path

        hub_kwargs = {k: kwargs.pop(k, default) for k, default in HUB_KWARGS.items()}

        config = kwargs.pop("config", None)
        config_revision = kwargs.pop("config_revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        quantization_config = kwargs.pop("quantization_config", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        disable_mmap = kwargs.pop("disable_mmap", False)
        device_map = kwargs.pop("device_map", None)

        user_agent = {"diffusers": __version__, "file_type": "single_file", "framework": "pytorch"}
        if quantization_config is not None:
            user_agent["quant"] = quantization_config.quant_method.value

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            torch_dtype = torch.float32
            logger.warning(
                f"Passed `torch_dtype` {torch_dtype} is not a `torch.dtype`. Defaulting to `torch.float32`."
            )

        if isinstance(pretrained_model_link_or_path_or_dict, dict):
            state_dict = pretrained_model_link_or_path_or_dict
        else:
            state_dict = load_single_file_checkpoint(
                pretrained_model_link_or_path_or_dict,
                disable_mmap=disable_mmap,
                user_agent=user_agent,
                **{k: v for k, v in hub_kwargs.items() if k != "subfolder"},
            )

        state_dict = _weight_mapping.normalize_state_dict_keys(state_dict)

        if quantization_config is not None:
            hf_quantizer = DiffusersAutoQuantizer.from_config(quantization_config)
            hf_quantizer.validate_environment()
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
        else:
            hf_quantizer = None

        if config is not None:
            if isinstance(config, str):
                default_pretrained_model_config_name = config
            else:
                raise ValueError(
                    "Invalid `config` argument. Please provide a string representing a repo id "
                    "or path to a local Diffusers model repo."
                )
        else:
            default_pretrained_model_config_name = _weight_mapping.get_model_config(state_dict)
            if default_subfolder is not None:
                hub_kwargs["subfolder"] = default_subfolder

        diffusers_model_config = cls.load_config(
            pretrained_model_name_or_path=default_pretrained_model_config_name,
            **{**hub_kwargs, "revision": config_revision},
        )
        expected_kwargs, optional_kwargs = cls._get_signature_keys(cls)
        model_kwargs = {k: kwargs.get(k) for k in kwargs if k in expected_kwargs or k in optional_kwargs}
        diffusers_model_config.update(model_kwargs)

        if is_accelerate_available():
            from accelerate import init_empty_weights

            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
        else:
            ctx = nullcontext

        with ctx():
            model = cls.from_config(diffusers_model_config)

        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
        )
        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]
        else:
            keep_in_fp32_modules = []

        state_dict = _weight_mapping.maybe_convert_state_dict(model, state_dict)

        if not state_dict:
            raise SingleFileComponentError(
                f"Failed to load {cls.__name__}. Weights for this component appear to be missing in the checkpoint."
            )

        loaded_keys = list(state_dict.keys())

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
            )

        device_map = _determine_device_map(model, device_map, None, torch_dtype, keep_in_fp32_modules, hf_quantizer)
        if hf_quantizer is not None:
            hf_quantizer.validate_environment(device_map=device_map)

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            state_dict,
            None,
            None,
            loaded_keys,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            dtype=torch_dtype,
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=keep_in_fp32_modules,
        )

        if device_map is not None:
            from accelerate import dispatch_model

            device_map_kwargs = {
                "device_map": device_map,
                "offload_index": offload_index,
            }
            dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if torch_dtype is not None and hf_quantizer is None:
            model.to(torch_dtype)

        model.eval()

        return model
