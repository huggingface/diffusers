# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
import collections
import functools
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Set, Union

import safetensors
import torch
from huggingface_hub import model_info
from huggingface_hub.constants import HF_HUB_OFFLINE

from ..hooks.group_offloading import (
    _GROUP_OFFLOADING,
    _LAYER_EXECUTION_TRACKER,
    _LAZY_PREFETCH_GROUP_OFFLOADING,
    _apply_group_offloading,
    _get_top_level_group_offload_hook,
    _maybe_remove_and_reapply_group_offloading,
)
from ..hooks.hooks import HookRegistry
from ..models.model_loading_utils import load_state_dict
from ..utils import (
    HUB_KWARGS,
    USE_PEFT_BACKEND,
    _get_model_file,
    delete_adapter_layers,
    deprecate,
    get_adapter_name,
    is_accelerate_available,
    is_peft_available,
    is_peft_version,
    logging,
    recurse_remove_peft_layers,
    set_weights_and_activate_adapters,
)
from ..utils.state_dict_utils import _load_sft_state_dict_metadata
from .unet_loader_utils import _maybe_expand_lora_scales


if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, add_hook_to_module, remove_hook_from_module


if is_peft_available():
    from peft import LoraConfig, PeftConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import get_peft_model_state_dict
    from peft.utils.hotswap import (
        check_hotswap_configs_compatible,
        hotswap_adapter_from_state_dict,
        prepare_model_for_compiled_hotswap,
    )


logger = logging.get_logger(__name__)


# Minimum PEFT version this mixin relies on. Bumping this lets us delete the
# version-fallback branches scattered through the methods (DoRA, lora_bias,
# hotswap, set_adapter hasattr, etc.).
_MIN_PEFT_VERSION_FOR_LORA = "0.14.1"
_HAS_REQUIRED_PEFT = USE_PEFT_BACKEND and is_peft_version(">=", _MIN_PEFT_VERSION_FOR_LORA)

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"
LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"


def _normalize_lora_suffixes(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """Rewrite ``.lora_down/.lora_up`` (kohya-ish) suffixes to ``.lora_A/.lora_B`` (diffusers).

    Universal — every LoRA state dict goes through this regardless of model. Module-level so both :class:`LoRAHandler`
    (in its ``map_to_diffusers`` dispatcher) and :class:`LoRAModelMixin` (as a public ``normalize_lora_suffixes``
    utility) can call it without circular references.
    """
    out: Dict[str, "torch.Tensor"] = {}
    for k, v in state_dict.items():
        new_k = (
            k.replace(".lora_down.weight", ".lora_A.weight")
            .replace(".lora_up.weight", ".lora_B.weight")
            .replace(".down.weight", ".lora_A.weight")
            .replace(".up.weight", ".lora_B.weight")
        )
        out[new_k] = v
    return out


@dataclass
class LoRAHandler:
    """Composition-style holder for a model class's LoRA conversion configuration.

    Attached as the ``_lora`` class attribute on :class:`LoRAModelMixin` (overridden per-model). Holds the per-model
    foreign-format conversion data. Public conversion utilities (``normalize_lora_suffixes``, ``detect_lora_format``)
    live on :class:`LoRAModelMixin` and read from this handler.

    Attributes:
        format_keys: Map of format name (``"kohya"``, ``"xlabs"``, ...) to identifying key substrings. The first
            format whose substrings appear in the state dict wins.
        map_lora_to_diffusers_fn: Callable ``(state_dict, **kwargs) -> state_dict`` that rewrites foreign-format
            keys to diffusers naming. ``None`` for models that only ingest diffusers-native LoRAs.
    """

    format_keys: Dict[str, Set[str]] = field(default_factory=dict)
    map_lora_to_diffusers_fn: Optional[Callable[..., Dict[str, "torch.Tensor"]]] = None

    def map_to_diffusers(self, state_dict: Dict[str, "torch.Tensor"], **kwargs) -> Dict[str, "torch.Tensor"]:
        """Run the per-model converter (or pass through if none is registered).

        Callers are expected to call :meth:`LoRAModelMixin.normalize_lora_suffixes` separately before this — the
        kohya-style suffix normalization is universal and isn't this handler's responsibility.
        """
        if self.map_lora_to_diffusers_fn is None:
            return state_dict
        return self.map_lora_to_diffusers_fn(state_dict, **kwargs)


# Per-class hook for expanding adapter weights before activation. Models that need
# expansion (currently only UNet variants) register here; everything else falls
# through to the identity default so new transformers don't need an entry.
_SET_ADAPTER_SCALE_FN_MAPPING = defaultdict(
    lambda: (lambda model_cls, weights: weights),
    {
        "UNet2DConditionModel": _maybe_expand_lora_scales,
        "UNetMotionModel": _maybe_expand_lora_scales,
    },
)


def _requires_peft(method):
    """Guard a method with a uniform PEFT availability + minimum-version check."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not _HAS_REQUIRED_PEFT:
            raise ValueError(
                f"`{method.__name__}()` requires PEFT >= {_MIN_PEFT_VERSION_FOR_LORA}. "
                "Please install or upgrade PEFT: `pip install -U peft`."
            )
        return method(self, *args, **kwargs)

    return wrapper


def _fuse_lora_apply(module, lora_scale=1.0, safe_fusing=False, adapter_names=None):
    """Per-module callback for ``self.apply(...)`` in ``fuse_lora``."""
    if not isinstance(module, BaseTunerLayer):
        return
    if lora_scale != 1.0:
        module.scale_layer(lora_scale)
    module.merge(safe_merge=safe_fusing, adapter_names=adapter_names)


def _unfuse_lora_apply(module):
    if isinstance(module, BaseTunerLayer):
        module.unmerge()


def _serialize_lora_adapter_metadata(peft_config):
    """Convert a ``PeftConfig`` to a JSON string suitable for the safetensors metadata blob.

    PEFT configs may contain ``set`` values (which JSON can't serialize); coerce those to lists first.
    """
    cfg = peft_config.to_dict()
    for key, value in cfg.items():
        if isinstance(value, set):
            cfg[key] = list(value)
    return json.dumps(cfg, indent=2, sort_keys=True)


def _scope_state_dict_to_adapter(state_dict, adapter_name):
    """Rewrite ``lora_A.weight`` / ``lora_B.weight`` keys to include the adapter name
    (the format expected by ``hotswap_adapter_from_state_dict``)."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith("lora_A.weight") or k.endswith("lora_B.weight"):
            k = k[: -len(".weight")] + f".{adapter_name}.weight"
        elif k.endswith("lora_B.bias"):  # lora_bias=True option
            k = k[: -len(".bias")] + f".{adapter_name}.bias"
        out[k] = v
    return out


def _split_majority_and_outliers(value_dict):
    """Return ``(majority, outliers)`` for ``value_dict``.

    ``majority`` is the most common value (or the lone value if all are equal, or None for an empty dict). ``outliers``
    is a sub-dict of the items whose value differs from the majority — empty when every value matches.
    """
    values = list(value_dict.values())
    if not values:
        return None, {}
    if len(set(values)) == 1:
        return values[0], {}
    majority = collections.Counter(values).most_common(1)[0][0]
    return majority, {k: v for k, v in value_dict.items() if v != majority}


@contextmanager
def _offloading_disabled(model):
    """Temporarily strip accelerate and group-offload hooks from ``model``.

    PEFT injection and weight loading mutate the model graph in ways that fight with active offload hooks (sequential
    CPU offload, group offload, etc.). This context saves the hook state, removes the hooks for the duration of the
    block, and restores them on exit so existing offloading config survives a LoRA load.
    """
    saved_hf_hook = None
    is_sequential = False
    if hasattr(model, "_hf_hook"):
        hook = model._hf_hook
        if isinstance(hook, CpuOffload):
            saved_hf_hook = hook
        elif isinstance(hook, AlignDevicesHook) or (
            hasattr(hook, "hooks") and isinstance(hook.hooks[0], AlignDevicesHook)
        ):
            saved_hf_hook = hook
            is_sequential = True
    if saved_hf_hook is not None:
        remove_hook_from_module(model, recurse=is_sequential)

    saved_group_offload_config = None
    top_level_group_hook = _get_top_level_group_offload_hook(model)
    if top_level_group_hook is not None:
        saved_group_offload_config = top_level_group_hook.config
        registry = HookRegistry.check_if_exists_or_initialize(model)
        registry.remove_hook(_GROUP_OFFLOADING, recurse=True)
        registry.remove_hook(_LAYER_EXECUTION_TRACKER, recurse=True)
        registry.remove_hook(_LAZY_PREFETCH_GROUP_OFFLOADING, recurse=True)

    try:
        yield
    finally:
        if saved_hf_hook is not None:
            add_hook_to_module(model, saved_hf_hook)
        if saved_group_offload_config is not None:
            _apply_group_offloading(model, saved_group_offload_config)


def _create_lora_config(state_dict, network_alphas, rank_dict, metadata=None):
    """Build a PEFT ``LoraConfig`` from a LoRA state dict.

    ``metadata`` (when present) overrides the inferred kwargs entirely — used when a saved adapter shipped its own
    serialized ``LoraConfig`` blob. Otherwise we infer: per-module rank / alpha values that don't match the majority go
    into ``rank_pattern`` / ``alpha_pattern``; the majority becomes the global default.
    """
    if metadata is not None:
        return LoraConfig(**metadata)

    r, rank_outliers = _split_majority_and_outliers(rank_dict)
    rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_outliers.items()}

    lora_alpha = r
    alpha_pattern = {}
    if network_alphas:
        lora_alpha, alpha_outliers = _split_majority_and_outliers(network_alphas)
        if alpha_outliers:
            # PEFT-converted alpha keys (UNet / transformer LoRAs) carry ``.lora_A.``;
            # raw kohya-style alphas (legacy text-encoder LoRAs) carry ``.down.``.
            sample = next(iter(alpha_outliers))
            if ".lora_A." in sample:
                alpha_pattern = {k.split(".lora_A.")[0].replace(".alpha", ""): v for k, v in alpha_outliers.items()}
            else:
                alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_outliers.items()}

    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": list({name.split(".lora")[0] for name in state_dict}),
        "use_dora": any("lora_magnitude_vector" in k for k in state_dict),
        "lora_bias": any("lora_B" in k and k.endswith(".bias") for k in state_dict),
    }

    return LoraConfig(**lora_config_kwargs)


def _maybe_warn_for_unhandled_keys(incompatible_keys, adapter_name):
    if incompatible_keys is None:
        return
    warn_msg = ""
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    if unexpected_keys:
        lora_unexpected_keys = [k for k in unexpected_keys if "lora_" in k and adapter_name in k]
        if lora_unexpected_keys:
            warn_msg = (
                f"Loading adapter weights from state_dict led to unexpected keys found in the model: "
                f"{', '.join(lora_unexpected_keys)}. "
            )
    missing_keys = getattr(incompatible_keys, "missing_keys", None)
    if missing_keys:
        lora_missing_keys = [k for k in missing_keys if "lora_" in k and adapter_name in k]
        if lora_missing_keys:
            warn_msg += (
                f"Loading adapter weights from state_dict led to missing keys in the model: "
                f"{', '.join(lora_missing_keys)}."
            )
    if warn_msg:
        logger.warning(warn_msg)


def _fetch_state_dict(pretrained_model_name_or_path_or_dict, weight_name=None, **hub_kwargs):
    """Load a LoRA state dict from a path/repo/dict.

    Safetensors only — pickle (``.bin``) LoRAs are no longer supported. Re-save legacy checkpoints with
    ``safetensors.torch.save_file`` or load them manually with ``torch.load`` and pass the resulting dict.

    ``hub_kwargs`` are the download / file-discovery options forwarded to ``_get_model_file`` (see ``HUB_KWARGS`` for
    the canonical set).
    """
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        return pretrained_model_name_or_path_or_dict

    source = pretrained_model_name_or_path_or_dict
    local_files_only = hub_kwargs.get("local_files_only")
    name = weight_name or _best_guess_weight_name(source, ".safetensors", local_files_only)
    model_file = _get_model_file(source, weights_name=name or LORA_WEIGHT_NAME_SAFE, **hub_kwargs)
    return load_state_dict(model_file)


def _fetch_lora_metadata(pretrained_model_name_or_path_or_dict, weight_name=None, **hub_kwargs):
    """Load LoRA adapter metadata from a safetensors file's sidecar.

    Returns ``None`` for non-safetensors sources (dicts, ``.bin`` files, missing sidecar). The hub layer caches the
    file, so calling this after
    """
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        return None

    source = pretrained_model_name_or_path_or_dict
    local_files_only = hub_kwargs.get("local_files_only")
    name = weight_name or _best_guess_weight_name(source, ".safetensors", local_files_only)
    if not name or not name.endswith(".safetensors"):
        return None
    try:
        model_file = _get_model_file(source, weights_name=name, **hub_kwargs)
        return _load_sft_state_dict_metadata(model_file)
    except (IOError, safetensors.SafetensorError):
        return None


def _best_guess_weight_name(
    pretrained_model_name_or_path_or_dict, file_extension=".safetensors", local_files_only=False
):
    if local_files_only or HF_HUB_OFFLINE:
        raise ValueError("When using the offline mode, you must specify a `weight_name`.")

    if os.path.isfile(pretrained_model_name_or_path_or_dict):
        return None
    if os.path.isdir(pretrained_model_name_or_path_or_dict):
        targeted_files = [f for f in os.listdir(pretrained_model_name_or_path_or_dict) if f.endswith(file_extension)]
    else:
        files_in_repo = model_info(pretrained_model_name_or_path_or_dict).siblings
        targeted_files = [f.rfilename for f in files_in_repo if f.rfilename.endswith(file_extension)]

    # Strip non-LoRA files: scheduler/optimizer state, intermediate checkpoints.
    unallowed = {"scheduler", "optimizer", "checkpoint"}
    targeted_files = [f for f in targeted_files if not any(s in f for s in unallowed)]

    # Prefer the canonical filenames if present.
    for canonical in (LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE):
        if any(f.endswith(canonical) for f in targeted_files):
            targeted_files = [f for f in targeted_files if f.endswith(canonical)]
            break

    if not targeted_files:
        return None
    if len(targeted_files) > 1:
        logger.warning(
            f"Provided path contains more than one weights file in the {file_extension} format. "
            f"`{targeted_files[0]}` is going to be loaded; for precise control, specify a `weight_name` "
            "in `load_lora_weights`."
        )
    return targeted_files[0]


class LoRAModelMixin:
    """
    Single mixin for everything LoRA on a diffusers model: PEFT adapter lifecycle (load / fuse / unfuse / set / delete
    / hotswap) plus foreign-format conversion (kohya / xlabs / bfl / kontext / etc.) into diffusers naming.

    Per-model conversion knobs live in a :class:`LoRAHandler` declared in the model's ``lora.py`` (e.g. ``FLUX_LORA``)
    and assigned to the model class as ``_lora = FLUX_LORA``. The default no-op handler just normalizes
    ``.lora_down/.lora_up`` → ``.lora_A/.lora_B`` suffixes and returns the state dict unchanged.

    Install the latest version of PEFT, and use this mixin to:

    - Attach new adapters in the model.
    - Attach multiple adapters and iteratively activate/deactivate them.
    - Activate/deactivate all adapters from the model.
    - Get a list of the active adapters.
    """

    # Runtime PEFT state — set during adapter load / hotswap setup.
    _hf_peft_config_loaded = False
    _lora_hotswap_kwargs: Optional[dict] = None

    # Per-model LoRA conversion config. Defaults to a no-op handler (only suffix normalization, no foreign-format
    # conversion). Models override by assigning ``_lora = FLUX_LORA`` (etc.) in their class body.
    _lora: LoRAHandler = LoRAHandler()

    @classmethod
    def _metadata(cls):
        """Contribute the ``lora_formats`` row to :class:`ModelMetadata` when foreign formats are registered."""
        from ..models.modeling_utils import DOCS_BASE

        formats = sorted(cls._lora.format_keys)
        if not formats:
            return {}
        return {
            "lora_formats": (
                ", ".join(formats),
                "Foreign LoRA formats this model converts to diffusers naming on load.",
                f"{DOCS_BASE}/training/lora",
            )
        }

    @staticmethod
    def normalize_lora_suffixes(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Rewrite ``.lora_down/.lora_up`` (kohya-ish) suffixes to ``.lora_A/.lora_B`` (diffusers).

        Universal — applies to every LoRA state dict regardless of model. Useful as a standalone utility for callers
        that want suffix normalization without running the full ``map_to_diffusers`` pipeline.
        """
        return _normalize_lora_suffixes(state_dict)

    def detect_lora_format(self, state_dict: Dict[str, "torch.Tensor"]) -> Optional[str]:
        """Return the foreign LoRA format name (``"kohya"`` / ``"xlabs"`` / ...) matched by ``state_dict``,
        or ``None`` if no registered format matches (e.g. it's already in diffusers naming).

        Reads ``self._lora.format_keys`` (the per-model registry of identifying key substrings).
        """
        format_keys = self._lora.format_keys
        if not format_keys:
            return None
        keys = set(state_dict)
        for fmt, fmt_keys in format_keys.items():
            if any(any(fk in k for k in keys) for fk in fmt_keys):
                return fmt
        return None

    @_requires_peft
    def load_adapter(
        self,
        adapter,
        adapter_name=None,
        prefix="transformer",
        hotswap: bool = False,
        **kwargs,
    ):
        r"""
        Add an adapter to the underlying model.

        ``source`` can be either:

        - A ``PeftConfig`` (e.g. ``LoraConfig``) — initializes a fresh adapter with random weights, suitable for
          training.
        - A repo id, local path, or pre-loaded ``state_dict`` — loads pretrained adapter weights, suitable for
          inference.

        For the config path, only ``adapter_name`` is used; ``prefix``, ``hotswap``, and the download/loading kwargs
        apply to the pretrained path.
        """
        adapter_name = adapter_name or get_adapter_name(self)
        if isinstance(adapter, PeftConfig):
            return self._load_adapter_from_config(adapter, adapter_name=adapter_name)

        return self._load_adapter_from_pretrained(
            adapter, adapter_name=adapter_name, prefix=prefix, hotswap=hotswap, **kwargs
        )

    def _load_adapter_from_config(self, adapter_config, adapter_name="default"):
        if self._hf_peft_config_loaded and adapter_name in getattr(self, "peft_config", {}):
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        # Unlike transformers, here we don't need to retrieve the name_or_path of the unet as the loading logic is
        # handled by the `load_lora_layers` or `StableDiffusionLoraLoaderMixin`. Therefore we set it to `None` here.
        adapter_config.base_model_name_or_path = None
        inject_adapter_in_model(adapter_config, self, adapter_name)
        self._hf_peft_config_loaded = True
        self.set_adapters(adapter_name)

    def _load_adapter_from_pretrained(
        self,
        pretrained_model_name_or_path_or_dict,
        adapter_name=None,
        prefix="transformer",
        hotswap: bool = False,
        **kwargs,
    ):
        r"""
        Loads a LoRA adapter into the underlying model.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            prefix (`str`, *optional*): Prefix to filter the state dict.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            network_alphas (`Dict[str, float]`):
                The value of the network alpha used for stable learning and preventing underflow. This value has the
                same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
                link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            hotswap : (`bool`, *optional*)
                Defaults to `False`. Whether to substitute an existing (LoRA) adapter with the newly loaded adapter
                in-place. This means that, instead of loading an additional adapter, this will take the existing
                adapter weights and replace them with the weights of the new adapter. This can be faster and more
                memory efficient. However, the main advantage of hotswapping is that when the model is compiled with
                torch.compile, loading the new adapter does not require recompilation of the model. When using
                hotswapping, the passed `adapter_name` should be the name of an already loaded adapter.

                If the new adapter and the old adapter have different ranks and/or LoRA alphas (i.e. scaling), you need
                to call an additional method before loading the adapter:

                ```py
                pipeline = ...  # load diffusers pipeline
                max_rank = ...  # the highest rank among all LoRAs that you want to load
                # call *before* compiling and loading the LoRA adapter
                pipeline.enable_lora_hotswap(target_rank=max_rank)
                pipeline.load_lora_weights(file_name)
                # optionally compile the model now
                ```

                Note that hotswapping adapters of the text encoder is not yet supported. There are some further
                limitations to this technique, which are documented here:
                https://huggingface.co/docs/peft/main/en/package_reference/hotswap
            metadata:
                LoRA adapter metadata. When supplied, the metadata inferred through the state dict isn't used to
                initialize `LoraConfig`.
        """
        hub_kwargs = {k: kwargs.pop(k, default) for k, default in HUB_KWARGS.items()}
        hub_kwargs["user_agent"] = {"file_type": "attn_procs_weights", "framework": "pytorch"}

        weight_name = kwargs.pop("weight_name", None)
        network_alphas = kwargs.pop("network_alphas", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        metadata = kwargs.pop("metadata", None)

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = pretrained_model_name_or_path_or_dict
        else:
            source = pretrained_model_name_or_path_or_dict
            name = weight_name or _best_guess_weight_name(source, ".safetensors", hub_kwargs.get("local_files_only"))
            model_file = _get_model_file(source, weights_name=name or LORA_WEIGHT_NAME_SAFE, **hub_kwargs)
            state_dict = load_state_dict(model_file)

        # Universal suffix normalization first (kohya-style ``.lora_down/.lora_up`` → ``.lora_A/.lora_B``), then
        # run the per-model foreign-format converter (no-op when none is registered).
        state_dict = self.normalize_lora_suffixes(state_dict)
        state_dict = self._lora.map_to_diffusers(state_dict)
        if not state_dict:
            model_class_name = self.__class__.__name__
            logger.warning(
                f"No LoRA keys associated to {model_class_name} found with the {prefix=}. "
                "This is safe to ignore if LoRA state dict didn't originally have any "
                f"{model_class_name} related params. You can also try specifying `prefix=None` "
                "to resolve the warning. Otherwise, open an issue if you think it's unexpected: "
                "https://github.com/huggingface/diffusers/issues/new"
            )
            return

        metadata = metadata or _fetch_lora_metadata(
            pretrained_model_name_or_path_or_dict, weight_name=weight_name, **hub_kwargs
        )

        if network_alphas is not None and prefix is None:
            raise ValueError("`network_alphas` cannot be None when `prefix` is None.")

        if network_alphas and metadata:
            raise ValueError("Both `network_alphas` and `metadata` cannot be specified.")

        if prefix is not None:
            state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
            if metadata is not None:
                metadata = {k.removeprefix(f"{prefix}."): v for k, v in metadata.items() if k.startswith(f"{prefix}.")}

        if adapter_name in getattr(self, "peft_config", {}) and not hotswap:
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the model - please select a new adapter name."
            )
        if adapter_name not in getattr(self, "peft_config", {}) and hotswap:
            raise ValueError(
                f"Trying to hotswap LoRA adapter '{adapter_name}' but there is no existing adapter by that name. "
                "Please choose an existing adapter name or set `hotswap=False` to prevent hotswapping."
            )

        rank = {}
        for key, val in state_dict.items():
            # Cannot figure out rank from lora layers that don't have at least 2 dimensions.
            # Bias layers in LoRA only have a single dimension
            if "lora_B" in key and val.ndim > 1:
                # See https://github.com/huggingface/peft/pull/2419 for the `^` symbol.
                # Disambiguates module names sharing a common prefix
                # (e.g. `proj_out.weight` vs `blocks.transformer.proj_out.weight`).
                rank[f"^{key}"] = val.shape[1]

        if network_alphas is not None and len(network_alphas) >= 1:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(f"{prefix}.")]
            network_alphas = {k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys}

        lora_config = _create_lora_config(state_dict, network_alphas, rank, metadata=metadata)

        # Mutating the model would otherwise fight with active offload hooks; the
        # context manager strips them for the duration and restores them on exit.
        peft_kwargs = {"low_cpu_mem_usage": low_cpu_mem_usage}
        with _offloading_disabled(self):
            if hotswap:
                self._hotswap_adapter(state_dict, lora_config, adapter_name)
                incompatible_keys = None

            else:
                incompatible_keys = self._inject_adapter(state_dict, lora_config, adapter_name, peft_kwargs)
                self._maybe_apply_deferred_hotswap_prep(lora_config)

        _maybe_warn_for_unhandled_keys(incompatible_keys, adapter_name)

    def _inject_adapter(self, state_dict, lora_config, adapter_name, peft_kwargs):
        """Inject a new adapter into ``self`` and load its weights.

        Returns the ``incompatible_keys`` reported by ``set_peft_model_state_dict``. On failure, rolls back any partial
        peft_config / adapter modules so the model is left in its prior state.
        """
        try:
            inject_adapter_in_model(lora_config, self, adapter_name=adapter_name, state_dict=state_dict, **peft_kwargs)
            incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name, **peft_kwargs)
            self._hf_peft_config_loaded = True

            return incompatible_keys

        except Exception as e:
            self._rollback_adapter(adapter_name, e)
            raise

    def _maybe_apply_deferred_hotswap_prep(self, lora_config):
        """If ``enable_lora_hotswap`` was called before the first adapter was loaded,
        we deferred ``prepare_model_for_compiled_hotswap`` until LoRA layers existed. Apply it now (after a successful
        inject) and clear the stash so it only fires once."""
        if self._lora_hotswap_kwargs is None:
            return
        prepare_model_for_compiled_hotswap(self, config=lora_config, **self._lora_hotswap_kwargs)
        self._lora_hotswap_kwargs = None

    def _hotswap_adapter(self, state_dict, lora_config, adapter_name):
        """Replace the weights of an already-loaded adapter in-place.

        ``hotswap_adapter_from_state_dict`` raises on incompatible keys; reaching the end of this function means the
        swap succeeded.
        """
        state_dict = _scope_state_dict_to_adapter(state_dict, adapter_name)
        check_hotswap_configs_compatible(self.peft_config[adapter_name], lora_config)
        try:
            hotswap_adapter_from_state_dict(
                model=self, state_dict=state_dict, adapter_name=adapter_name, config=lora_config
            )
        except Exception as e:
            logger.error(f"Hotswapping {adapter_name} was unsuccessful with the following error: \n{e}")
            self._rollback_adapter(adapter_name, e)
            raise

    def _rollback_adapter(self, adapter_name, error):
        """Remove ``adapter_name`` from ``self`` so failed loads don't leave partial state."""
        if hasattr(self, "peft_config"):
            for module in self.modules():
                if isinstance(module, BaseTunerLayer):
                    for active_adapter in module.active_adapters:
                        if adapter_name in active_adapter:
                            module.delete_adapter(adapter_name)
            self.peft_config.pop(adapter_name, None)

        logger.error(f"Loading {adapter_name} was unsuccessful with the following error: \n{error}")

    @_requires_peft
    def save_adapter(
        self,
        save_directory,
        adapter_name: str = "default",
        upcast_before_saving: bool = False,
        safe_serialization: bool = True,
        weight_name: Optional[str] = None,
    ):
        """Save the LoRA parameters corresponding to the underlying model.

        Args:
            save_directory: Directory to save LoRA parameters to. Created if missing.
            adapter_name: Name of the adapter to serialize. Useful when the model has
                multiple adapters loaded.
            upcast_before_saving: Whether to cast the underlying model to ``torch.float32``
                before serialization.
            safe_serialization: Save with ``safetensors`` (default) or pickled torch save.
            weight_name: Override the default filename.
        """
        if adapter_name not in getattr(self, "peft_config", {}):
            raise ValueError(f"Adapter name {adapter_name} not found in the model.")
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        state_dict = get_peft_model_state_dict(
            self.to(dtype=torch.float32 if upcast_before_saving else None), adapter_name=adapter_name
        )

        os.makedirs(save_directory, exist_ok=True)
        weight_name = weight_name or (LORA_WEIGHT_NAME_SAFE if safe_serialization else LORA_WEIGHT_NAME)
        save_path = Path(save_directory, weight_name).as_posix()

        if safe_serialization:
            metadata = {
                "format": "pt",
                LORA_ADAPTER_METADATA_KEY: _serialize_lora_adapter_metadata(self.peft_config[adapter_name]),
            }
            safetensors.torch.save_file(state_dict, save_path, metadata=metadata)
        else:
            torch.save(state_dict, save_path)

        logger.info(f"Model weights saved in {save_path}")

    def save_lora_adapter(self, *args, **kwargs):
        """Deprecated alias for :meth:`save_adapter`."""
        deprecate(
            "save_lora_adapter",
            "1.0.0",
            "`save_lora_adapter` is deprecated; use `save_adapter` instead.",
        )
        return self.save_adapter(*args, **kwargs)

    @_requires_peft
    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        """
        Set the currently active adapters for use in the diffusion network (e.g. unet, transformer, etc.).

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.unet.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # Expand weights into a list, one entry per adapter
        # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # Set None values to default of 1.0
        # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # e.g. [{...}, 7] -> [{expanded dict...}, 7]
        scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]
        weights = scale_expansion_fn(self, weights)

        set_weights_and_activate_adapters(self, adapter_names, weights)

    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        """Deprecated alias for :meth:`load_adapter` with a ``PeftConfig``."""
        deprecate(
            "add_adapter",
            "1.0.0",
            "`add_adapter` is deprecated; use `load_adapter(adapter_config)` instead.",
        )
        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )
        return self.load_adapter(adapter_config, adapter_name=adapter_name)

    def load_lora_adapter(
        self, pretrained_model_name_or_path_or_dict, prefix="transformer", hotswap: bool = False, **kwargs
    ):
        """Deprecated alias for :meth:`load_adapter`."""
        deprecate(
            "load_lora_adapter",
            "1.0.0",
            "`load_lora_adapter` is deprecated; use `load_adapter` instead.",
        )
        return self.load_adapter(pretrained_model_name_or_path_or_dict, prefix=prefix, hotswap=hotswap, **kwargs)

    def set_adapter(self, adapter_name: Union[str, List[str]]) -> None:
        """Deprecated alias for :meth:`set_adapters`.

        Note: ``set_adapters`` resets the per-adapter scale to ``1.0`` when no weights are passed; the original
        ``set_adapter`` left the previous scale untouched.
        """
        deprecate(
            "set_adapter",
            "1.0.0",
            "`set_adapter` is deprecated; use `set_adapters` instead. "
            "Note that `set_adapters(name)` resets the per-adapter scale to 1.0; "
            "pass `weights=...` to control it explicitly.",
        )
        return self.set_adapters(adapter_name)

    @_requires_peft
    def disable_adapters(self) -> None:
        r"""
        Disable all adapters attached to the model and fallback to inference with the base model only.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=False)

    @_requires_peft
    def enable_adapters(self) -> None:
        """
        Enable adapters that are attached to the model. The model uses `self.active_adapters()` to retrieve the list of
        adapters to enable.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=True)

    @_requires_peft
    def active_adapters(self) -> List[str]:
        """Return the sorted union of active adapter names across all PEFT layers."""
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        active = set()
        for module in self.modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            names = module.active_adapter
            active.update([names] if isinstance(names, str) else names)
        return sorted(active)

    @_requires_peft
    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        """Merge LoRA adapter weights into the base model in-place."""
        self.apply(
            partial(_fuse_lora_apply, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names)
        )

    @_requires_peft
    def unfuse_lora(self):
        """Reverse of :meth:`fuse_lora` — unmerge LoRA weights from the base model."""
        self.apply(_unfuse_lora_apply)

    @_requires_peft
    def delete_adapters(self, adapter_names: Optional[Union[List[str], str]] = None):
        """Remove adapter(s) from the model.

        Pass specific names to delete those adapters only — the PEFT wrapper layers (``lora_A`` / ``lora_B`` modules)
        stay in place, so a subsequent :meth:`load_adapter` call can reuse them without re-injecting.

        Pass ``None`` (the default) to remove every adapter *and* strip the wrapper layers themselves, returning the
        model to its pre-LoRA state.
        """
        if adapter_names is None:
            recurse_remove_peft_layers(self)
            if hasattr(self, "peft_config"):
                del self.peft_config

            self._hf_peft_config_loaded = False

        else:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]

            for adapter_name in adapter_names:
                delete_adapter_layers(self, adapter_name)
                if hasattr(self, "peft_config"):
                    self.peft_config.pop(adapter_name, None)

        # In-place mutation invalidates group-offload tensor refs; refresh them.
        _maybe_remove_and_reapply_group_offloading(self)

    def unload_lora(self):
        """Deprecated alias for :meth:`delete_adapters` (with no arguments)."""
        deprecate(
            "unload_lora",
            "1.0.0",
            "`unload_lora` is deprecated; use `delete_adapters()` (no args) for the same teardown.",
        )
        return self.delete_adapters()

    def enable_lora_hotswap(
        self, target_rank: int = 128, check_compiled: Literal["error", "warn", "ignore"] = "error"
    ) -> None:
        """Enables the possibility to hotswap LoRA adapters.

        Calling this method is only required when hotswapping adapters and if the model is compiled or if the ranks of
        the loaded adapters differ.

        Args:
            target_rank (`int`, *optional*, defaults to `128`):
                The highest rank among all the adapters that will be loaded.

            check_compiled (`str`, *optional*, defaults to `"error"`):
                How to handle the case when the model is already compiled, which should generally be avoided. The
                options are:
                  - "error" (default): raise an error
                  - "warn": issue a warning
                  - "ignore": do nothing
        """
        if check_compiled not in ("error", "warn", "ignore"):
            raise ValueError(
                f"check_compiled should be one of 'error', 'warn', or 'ignore', got '{check_compiled}' instead."
            )
        if getattr(self, "peft_config", {}):
            if check_compiled == "error":
                raise RuntimeError("Call `enable_lora_hotswap` before loading the first adapter.")
            if check_compiled == "warn":
                logger.warning(
                    "It is recommended to call `enable_lora_hotswap` before loading the first adapter to avoid recompilation."
                )
        self._lora_hotswap_kwargs = {"target_rank": target_rank, "check_compiled": check_compiled}


# Back-compat alias. Old name from the PEFT-only era; prefer ``LoRAModelMixin``.
PeftAdapterMixin = LoRAModelMixin
