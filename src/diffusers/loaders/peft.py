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
import inspect
import json
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import safetensors
import torch

from ..utils import (
    MIN_PEFT_VERSION,
    USE_PEFT_BACKEND,
    check_peft_version,
    convert_unet_state_dict_to_peft,
    delete_adapter_layers,
    get_adapter_name,
    is_peft_available,
    is_peft_version,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from ..utils.peft_utils import _create_lora_config, _maybe_warn_for_unhandled_keys
from .lora_base import _fetch_state_dict, _func_optionally_disable_offloading
from .unet_loader_utils import _maybe_expand_lora_scales


logger = logging.get_logger(__name__)

_SET_ADAPTER_SCALE_FN_MAPPING = {
    "UNet2DConditionModel": _maybe_expand_lora_scales,
    "UNetMotionModel": _maybe_expand_lora_scales,
    "SD3Transformer2DModel": lambda model_cls, weights: weights,
    "FluxTransformer2DModel": lambda model_cls, weights: weights,
    "CogVideoXTransformer3DModel": lambda model_cls, weights: weights,
    "ConsisIDTransformer3DModel": lambda model_cls, weights: weights,
    "MochiTransformer3DModel": lambda model_cls, weights: weights,
    "HunyuanVideoTransformer3DModel": lambda model_cls, weights: weights,
    "LTXVideoTransformer3DModel": lambda model_cls, weights: weights,
    "SanaTransformer2DModel": lambda model_cls, weights: weights,
    "AuraFlowTransformer2DModel": lambda model_cls, weights: weights,
    "Lumina2Transformer2DModel": lambda model_cls, weights: weights,
    "WanTransformer3DModel": lambda model_cls, weights: weights,
    "CogView4Transformer2DModel": lambda model_cls, weights: weights,
    "HiDreamImageTransformer2DModel": lambda model_cls, weights: weights,
    "HunyuanVideoFramepackTransformer3DModel": lambda model_cls, weights: weights,
    "WanVACETransformer3DModel": lambda model_cls, weights: weights,
    "ChromaTransformer2DModel": lambda model_cls, weights: weights,
    "QwenImageTransformer2DModel": lambda model_cls, weights: weights,
}


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
    more details about adapters and injecting them in a base model, check out the PEFT
    [documentation](https://huggingface.co/docs/peft/index).

    Install the latest version of PEFT, and use this mixin to:

    - Attach new adapters in the model.
    - Attach multiple adapters and iteratively activate/deactivate them.
    - Activate/deactivate all adapters from the model.
    - Get a list of the active adapters.
    """

    _hf_peft_config_loaded = False
    # kwargs for prepare_model_for_compiled_hotswap, if required
    _prepare_lora_hotswap_kwargs: Optional[dict] = None

    @classmethod
    # Copied from diffusers.loaders.lora_base.LoraBaseMixin._optionally_disable_offloading
    def _optionally_disable_offloading(cls, _pipeline):
        return _func_optionally_disable_offloading(_pipeline=_pipeline)

    def load_lora_adapter(
        self, pretrained_model_name_or_path_or_dict, prefix="transformer", hotswap: bool = False, **kwargs
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
        from peft import inject_adapter_in_model, set_peft_model_state_dict
        from peft.tuners.tuners_utils import BaseTunerLayer

        from ..hooks.group_offloading import _maybe_remove_and_reapply_group_offloading

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        adapter_name = kwargs.pop("adapter_name", None)
        network_alphas = kwargs.pop("network_alphas", None)
        _pipeline = kwargs.pop("_pipeline", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        metadata = kwargs.pop("metadata", None)
        allow_pickle = False

        if low_cpu_mem_usage and is_peft_version("<=", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}
        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
            metadata=metadata,
        )
        if network_alphas is not None and prefix is None:
            raise ValueError("`network_alphas` cannot be None when `prefix` is None.")
        if network_alphas and metadata:
            raise ValueError("Both `network_alphas` and `metadata` cannot be specified.")

        if prefix is not None:
            state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
            if metadata is not None:
                metadata = {k.removeprefix(f"{prefix}."): v for k, v in metadata.items() if k.startswith(f"{prefix}.")}

        if len(state_dict) > 0:
            if adapter_name in getattr(self, "peft_config", {}) and not hotswap:
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the model - please select a new adapter name."
                )
            elif adapter_name not in getattr(self, "peft_config", {}) and hotswap:
                raise ValueError(
                    f"Trying to hotswap LoRA adapter '{adapter_name}' but there is no existing adapter by that name. "
                    "Please choose an existing adapter name or set `hotswap=False` to prevent hotswapping."
                )

            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            if "lora_A" not in first_key:
                state_dict = convert_unet_state_dict_to_peft(state_dict)

            rank = {}
            for key, val in state_dict.items():
                # Cannot figure out rank from lora layers that don't have at least 2 dimensions.
                # Bias layers in LoRA only have a single dimension
                if "lora_B" in key and val.ndim > 1:
                    # Check out https://github.com/huggingface/peft/pull/2419 for the `^` symbol.
                    # We may run into some ambiguous configuration values when a model has module
                    # names, sharing a common prefix (`proj_out.weight` and `blocks.transformer.proj_out.weight`,
                    # for example) and they have different LoRA ranks.
                    rank[f"^{key}"] = val.shape[1]

            if network_alphas is not None and len(network_alphas) >= 1:
                alpha_keys = [k for k in network_alphas.keys() if k.startswith(f"{prefix}.")]
                network_alphas = {
                    k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys
                }

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(self)

            # create LoraConfig
            lora_config = _create_lora_config(
                state_dict,
                network_alphas,
                metadata,
                rank,
                model_state_dict=self.state_dict(),
                adapter_name=adapter_name,
            )

            # <Unsafe code
            # We can be sure that the following works as it just sets attention processors, lora layers and puts all in the same dtype
            # Now we remove any existing hooks to `_pipeline`.

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error.
            is_model_cpu_offload, is_sequential_cpu_offload, is_group_offload = self._optionally_disable_offloading(
                _pipeline
            )
            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            if hotswap or (self._prepare_lora_hotswap_kwargs is not None):
                if is_peft_version(">", "0.14.0"):
                    from peft.utils.hotswap import (
                        check_hotswap_configs_compatible,
                        hotswap_adapter_from_state_dict,
                        prepare_model_for_compiled_hotswap,
                    )
                else:
                    msg = (
                        "Hotswapping requires PEFT > v0.14. Please upgrade PEFT to a higher version or install it "
                        "from source."
                    )
                    raise ImportError(msg)

            if hotswap:

                def map_state_dict_for_hotswap(sd):
                    # For hotswapping, we need the adapter name to be present in the state dict keys
                    new_sd = {}
                    for k, v in sd.items():
                        if k.endswith("lora_A.weight") or k.endswith("lora_B.weight"):
                            k = k[: -len(".weight")] + f".{adapter_name}.weight"
                        elif k.endswith("lora_B.bias"):  # lora_bias=True option
                            k = k[: -len(".bias")] + f".{adapter_name}.bias"
                        new_sd[k] = v
                    return new_sd

            # To handle scenarios where we cannot successfully set state dict. If it's unsuccessful,
            # we should also delete the `peft_config` associated to the `adapter_name`.
            try:
                if hotswap:
                    state_dict = map_state_dict_for_hotswap(state_dict)
                    check_hotswap_configs_compatible(self.peft_config[adapter_name], lora_config)
                    try:
                        hotswap_adapter_from_state_dict(
                            model=self,
                            state_dict=state_dict,
                            adapter_name=adapter_name,
                            config=lora_config,
                        )
                    except Exception as e:
                        logger.error(f"Hotswapping {adapter_name} was unsuccessful with the following error: \n{e}")
                        raise
                    # the hotswap function raises if there are incompatible keys, so if we reach this point we can set
                    # it to None
                    incompatible_keys = None
                else:
                    inject_adapter_in_model(
                        lora_config, self, adapter_name=adapter_name, state_dict=state_dict, **peft_kwargs
                    )
                    incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name, **peft_kwargs)

                    if self._prepare_lora_hotswap_kwargs is not None:
                        # For hotswapping of compiled models or adapters with different ranks.
                        # If the user called enable_lora_hotswap, we need to ensure it is called:
                        # - after the first adapter was loaded
                        # - before the model is compiled and the 2nd adapter is being hotswapped in
                        # Therefore, it needs to be called here
                        prepare_model_for_compiled_hotswap(
                            self, config=lora_config, **self._prepare_lora_hotswap_kwargs
                        )
                        # We only want to call prepare_model_for_compiled_hotswap once
                        self._prepare_lora_hotswap_kwargs = None

                # Set peft config loaded flag to True if module has been successfully injected and incompatible keys retrieved
                if not self._hf_peft_config_loaded:
                    self._hf_peft_config_loaded = True
            except Exception as e:
                # In case `inject_adapter_in_model()` was unsuccessful even before injecting the `peft_config`.
                if hasattr(self, "peft_config"):
                    for module in self.modules():
                        if isinstance(module, BaseTunerLayer):
                            active_adapters = module.active_adapters
                            for active_adapter in active_adapters:
                                if adapter_name in active_adapter:
                                    module.delete_adapter(adapter_name)

                    self.peft_config.pop(adapter_name)
                logger.error(f"Loading {adapter_name} was unsuccessful with the following error: \n{e}")
                raise

            _maybe_warn_for_unhandled_keys(incompatible_keys, adapter_name)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            elif is_group_offload:
                for component in _pipeline.components.values():
                    if isinstance(component, torch.nn.Module):
                        _maybe_remove_and_reapply_group_offloading(component)
            # Unsafe code />

        if prefix is not None and not state_dict:
            model_class_name = self.__class__.__name__
            logger.warning(
                f"No LoRA keys associated to {model_class_name} found with the {prefix=}. "
                "This is safe to ignore if LoRA state dict didn't originally have any "
                f"{model_class_name} related params. You can also try specifying `prefix=None` "
                "to resolve the warning. Otherwise, open an issue if you think it's unexpected: "
                "https://github.com/huggingface/diffusers/issues/new"
            )

    def save_lora_adapter(
        self,
        save_directory,
        adapter_name: str = "default",
        upcast_before_saving: bool = False,
        safe_serialization: bool = True,
        weight_name: Optional[str] = None,
    ):
        """
        Save the LoRA parameters corresponding to the underlying model.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            adapter_name: (`str`, defaults to "default"): The name of the adapter to serialize. Useful when the
                underlying model has multiple adapters loaded.
            upcast_before_saving (`bool`, defaults to `False`):
                Whether to cast the underlying model to `torch.float32` before serialization.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            weight_name: (`str`, *optional*, defaults to `None`): Name of the file to serialize the state dict with.
        """
        from peft.utils import get_peft_model_state_dict

        from .lora_base import LORA_ADAPTER_METADATA_KEY, LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE

        if adapter_name is None:
            adapter_name = get_adapter_name(self)

        if adapter_name not in getattr(self, "peft_config", {}):
            raise ValueError(f"Adapter name {adapter_name} not found in the model.")

        lora_adapter_metadata = self.peft_config[adapter_name].to_dict()

        lora_layers_to_save = get_peft_model_state_dict(
            self.to(dtype=torch.float32 if upcast_before_saving else None), adapter_name=adapter_name
        )
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if safe_serialization:

            def save_function(weights, filename):
                # Inject framework format.
                metadata = {"format": "pt"}
                if lora_adapter_metadata is not None:
                    for key, value in lora_adapter_metadata.items():
                        if isinstance(value, set):
                            lora_adapter_metadata[key] = list(value)
                    metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(lora_adapter_metadata, indent=2, sort_keys=True)

                return safetensors.torch.save_file(weights, filename, metadata=metadata)

        else:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME

        save_path = Path(save_directory, weight_name).as_posix()
        save_function(lora_layers_to_save, save_path)
        logger.info(f"Model weights saved in {save_path}")

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
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

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
        r"""
        Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
        to the adapter to follow the convention of the PEFT library.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them in the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_config (`[~peft.PeftConfig]`):
                The configuration of the adapter to add; supported adapters are non-prefix tuning and adaption prompt
                methods.
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        from peft import PeftConfig, inject_adapter_in_model

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        # Unlike transformers, here we don't need to retrieve the name_or_path of the unet as the loading logic is
        # handled by the `load_lora_layers` or `StableDiffusionLoraLoaderMixin`. Therefore we set it to `None` here.
        adapter_config.base_model_name_or_path = None
        inject_adapter_in_model(adapter_config, self, adapter_name)
        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[str, List[str]]) -> None:
        """
        Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_name (Union[str, List[str]])):
                The list of adapters to set or the adapter name in the case of a single adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        missing = set(adapter_name) - set(self.peft_config)
        if len(missing) > 0:
            raise ValueError(
                f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                f" current loaded adapters are: {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                # Previous versions of PEFT does not support multi-adapter inference
                elif not hasattr(module, "set_adapter") and len(adapter_name) != 1:
                    raise ValueError(
                        "You are trying to set multiple adapters and you have a PEFT version that does not support multi-adapter inference. Please upgrade to the latest version of PEFT."
                        " `pip install -U peft` or `pip install -U git+https://github.com/huggingface/peft.git`"
                    )
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        Disable all adapters attached to the model and fallback to inference with the base model only.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        Enable adapters that are attached to the model. The model uses `self.active_adapters()` to retrieve the list of
        adapters to enable.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = False

    def active_adapters(self) -> List[str]:
        """
        Gets the current list of active adapters of the model.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                return module.active_adapter

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `fuse_lora()`.")

        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

    def _fuse_lora_apply(self, module, adapter_names=None):
        from peft.tuners.tuners_utils import BaseTunerLayer

        merge_kwargs = {"safe_merge": self._safe_fusing}

        if isinstance(module, BaseTunerLayer):
            if self.lora_scale != 1.0:
                module.scale_layer(self.lora_scale)

            # For BC with previous PEFT versions, we need to check the signature
            # of the `merge` method to see if it supports the `adapter_names` argument.
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                    " to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)

    def unfuse_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `unfuse_lora()`.")
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        from peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(module, BaseTunerLayer):
            module.unmerge()

    def unload_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `unload_lora()`.")

        from ..hooks.group_offloading import _maybe_remove_and_reapply_group_offloading
        from ..utils import recurse_remove_peft_layers

        recurse_remove_peft_layers(self)
        if hasattr(self, "peft_config"):
            del self.peft_config
        if hasattr(self, "_hf_peft_config_loaded"):
            self._hf_peft_config_loaded = None

        _maybe_remove_and_reapply_group_offloading(self)

    def disable_lora(self):
        """
        Disables the active LoRA layers of the underlying model.

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
        pipeline.unet.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=False)

    def enable_lora(self):
        """
        Enables the active LoRA layers of the underlying model.

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
        pipeline.unet.enable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=True)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the underlying model.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.unet.delete_adapters("cinematic")
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            delete_adapter_layers(self, adapter_name)

            # Pop also the corresponding adapter from the config
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)

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
        if getattr(self, "peft_config", {}):
            if check_compiled == "error":
                raise RuntimeError("Call `enable_lora_hotswap` before loading the first adapter.")
            elif check_compiled == "warn":
                logger.warning(
                    "It is recommended to call `enable_lora_hotswap` before loading the first adapter to avoid recompilation."
                )
            elif check_compiled != "ignore":
                raise ValueError(
                    f"check_compiles should be one of 'error', 'warn', or 'ignore', got '{check_compiled}' instead."
                )

        self._prepare_lora_hotswap_kwargs = {"target_rank": target_rank, "check_compiled": check_compiled}
