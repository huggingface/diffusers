# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import re
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from huggingface_hub.utils import validate_hf_hub_args
from typing_extensions import Self

from .. import __version__
from ..models.model_loading_utils import (
    _caching_allocator_warmup,
    _determine_device_map,
    _expand_device_map,
)
from ..quantizers import DiffusersAutoQuantizer
from ..utils import deprecate, is_accelerate_available, is_torch_version, logging
from ..utils.torch_utils import empty_device_cache
from .conversion.registry import CONVERSION_BUILDERS
from .single_file_utils import (
    SingleFileComponentError,
    convert_model_checkpoint,
    create_controlnet_diffusers_config_from_ldm,
    create_unet_diffusers_config_from_ldm,
    create_vae_diffusers_config_from_ldm,
    fetch_diffusers_config,
    fetch_original_config,
    load_single_file_checkpoint,
)


logger = logging.get_logger(__name__)


if is_accelerate_available():
    from accelerate import dispatch_model, init_empty_weights

    from ..models.model_loading_utils import load_model_dict_into_meta

if is_torch_version(">=", "1.9.0") and is_accelerate_available():
    _LOW_CPU_MEM_USAGE_DEFAULT = True
else:
    _LOW_CPU_MEM_USAGE_DEFAULT = False

# Models with established original-config mappings or checkpoint-based config inference.
# Other registered conversions require an explicit Diffusers component config.
SINGLE_FILE_CONFIGS = {
    "StableCascadeUNet": {},
    "UNet2DConditionModel": {
        "config_mapping_fn": create_unet_diffusers_config_from_ldm,
        "default_subfolder": "unet",
        "legacy_kwargs": {
            "num_in_channels": "in_channels",  # Legacy kwargs supported by `from_single_file` mapped to new args
        },
    },
    "AutoencoderKL": {
        "config_mapping_fn": create_vae_diffusers_config_from_ldm,
        "default_subfolder": "vae",
    },
    "ControlNetModel": {
        "config_mapping_fn": create_controlnet_diffusers_config_from_ldm,
    },
    "SD3Transformer2DModel": {
        "default_subfolder": "transformer",
    },
    "MotionAdapter": {},
    "SparseControlNetModel": {},
    "FluxTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "ChromaTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "ErnieImageTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "LTXVideoTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "AutoencoderKLLTXVideo": {
        "default_subfolder": "vae",
    },
    "AutoencoderDC": {},
    "MochiTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "HunyuanVideoTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "AuraFlowTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "Lumina2Transformer2DModel": {
        "default_subfolder": "transformer",
    },
    "SanaTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "SkyReelsV2Transformer3DModel": {
        "default_subfolder": "transformer",
    },
    "ChronoEditTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "WanTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "WanVACETransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "WanAnimateTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "WanAnimate2Transformer3DModel": {
        "default_subfolder": "transformer",
    },
    "AutoencoderKLWan": {
        "default_subfolder": "vae",
    },
    "HiDreamImageTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "CosmosTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "QwenImageTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "Flux2Transformer2DModel": {
        "default_subfolder": "transformer",
    },
    "ZImageTransformer2DModel": {
        "default_subfolder": "transformer",
    },
    "ZImageControlNetModel": {},
    "LTX2VideoTransformer3DModel": {
        "default_subfolder": "transformer",
    },
    "AutoencoderKLLTX2Video": {
        "default_subfolder": "vae",
    },
    "AutoencoderKLLTX2Audio": {
        "default_subfolder": "audio_vae",
    },
    "MotifVideoTransformer3DModel": {
        "default_subfolder": "transformer",
    },
}


def _should_convert_state_dict_to_diffusers(model_state_dict, checkpoint_state_dict):
    model_state_dict_keys = set(model_state_dict.keys())
    checkpoint_state_dict_keys = set(checkpoint_state_dict.keys())
    is_subset = model_state_dict_keys.issubset(checkpoint_state_dict_keys)
    is_match = model_state_dict_keys == checkpoint_state_dict_keys
    return not (is_subset and is_match)


def _get_single_file_loadable_mapping_class(cls):
    # Follow the MRO so a specialized model keeps its own conversion before considering a base class.
    return next(
        (
            base.__name__
            for base in cls.__mro__
            if base.__module__.startswith("diffusers.") and base.__name__ in CONVERSION_BUILDERS
        ),
        None,
    )


def _get_mapping_function_kwargs(mapping_fn, **kwargs):
    parameters = inspect.signature(mapping_fn).parameters

    mapping_kwargs = {}
    for parameter in parameters:
        if parameter in kwargs:
            mapping_kwargs[parameter] = kwargs[parameter]

    return mapping_kwargs


class FromOriginalModelMixin:
    """
    Load pretrained weights saved in the `.ckpt` or `.safetensors` format into a model.
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(
        cls, pretrained_model_link_or_path_or_dict: str | Mapping[str, Any] | None = None, **kwargs
    ) -> Self:
        r"""
        Instantiate a model from pretrained weights saved in the original `.ckpt` or `.safetensors` format. The model
        is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path_or_dict (`str`, *optional*):
                Can be either:
                    - A link to the `.safetensors` or `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A path to a local *file* containing the weights of the component model.
                    - A state dict containing the component model weights.
            config (`str` or `dict`, *optional*):
                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline hosted
                  on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline component
                  configs in Diffusers format.
                - A dictionary containing the matching Diffusers component configuration.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            original_config (`str`, *optional*):
                Dict or path to a yaml file containing the configuration for the model in its original format.
                    If a dict is provided, it will be used to initialize the model configuration.
            dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`str | os.PathLike`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 and
                is_accelerate_available() else `False`): Speed up model loading only loading the pretrained weights and
                not initializing the weights. This also tries to not use more than 1x model size in CPU memory
                (including peak memory) while loading the model. Only supported for PyTorch >= 1.9.0. If you are using
                an older version of PyTorch, setting this argument to `True` will raise an error.
            disable_mmap ('bool', *optional*, defaults to 'False'):
                Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        ```py
        >>> from diffusers import StableCascadeUNet

        >>> ckpt_path = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite.safetensors"
        >>> model = StableCascadeUNet.from_single_file(ckpt_path)
        ```
        """

        mapping_class_name = _get_single_file_loadable_mapping_class(cls)
        if mapping_class_name is None:
            raise ValueError(f"No original checkpoint conversion is registered for {cls.__name__}.")

        pretrained_model_link_or_path = kwargs.get("pretrained_model_link_or_path", None)
        if pretrained_model_link_or_path is not None:
            deprecation_message = (
                "Please use `pretrained_model_link_or_path_or_dict` argument instead for model classes"
            )
            deprecate("pretrained_model_link_or_path", "1.0.0", deprecation_message)
            pretrained_model_link_or_path_or_dict = pretrained_model_link_or_path

        config = kwargs.pop("config", None)
        original_config = kwargs.pop("original_config", None)

        if config is not None and original_config is not None:
            raise ValueError(
                "`from_single_file` cannot accept both `config` and `original_config` arguments. Please provide only one of these arguments"
            )

        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        subfolder = kwargs.pop("subfolder", None)
        revision = kwargs.pop("revision", None)
        config_revision = kwargs.pop("config_revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = dtype if dtype is not None else torch_dtype
        quantization_config = kwargs.pop("quantization_config", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        device = kwargs.pop("device", None)
        disable_mmap = kwargs.pop("disable_mmap", False)
        device_map = kwargs.pop("device_map", None)

        user_agent = {
            "diffusers": __version__,
            "file_type": "single_file",
            "framework": "pytorch",
        }
        # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
        if quantization_config is not None:
            user_agent["quant"] = quantization_config.quant_method.value

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            torch_dtype = torch.float32
            logger.warning(
                f"Passed `torch_dtype` {torch_dtype} is not a `torch.dtype`. Defaulting to `torch.float32`."
            )

        if isinstance(pretrained_model_link_or_path_or_dict, Mapping):
            checkpoint = pretrained_model_link_or_path_or_dict
        else:
            checkpoint = load_single_file_checkpoint(
                pretrained_model_link_or_path_or_dict,
                force_download=force_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                disable_mmap=disable_mmap,
                user_agent=user_agent,
            )
        if quantization_config is not None:
            hf_quantizer = DiffusersAutoQuantizer.from_config(quantization_config)
            hf_quantizer.validate_environment()
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)

        else:
            hf_quantizer = None

        loading_config = SINGLE_FILE_CONFIGS.get(mapping_class_name, {"config_required": True})

        if loading_config.get("config_required", False) and config is None and original_config is None:
            raise ValueError(
                f"{mapping_class_name} requires an explicit Diffusers `config` directory or repository when loading "
                "a single file. The original weights do not specify all settings of the model variant."
            )

        if original_config is not None:
            if "config_mapping_fn" in loading_config:
                config_mapping_fn = loading_config["config_mapping_fn"]
            else:
                config_mapping_fn = None

            if config_mapping_fn is None:
                raise ValueError(
                    (
                        f"`original_config` has been provided for {mapping_class_name} but no mapping function"
                        "was found to convert the original config to a Diffusers config in"
                        "`diffusers.loaders.single_file_utils`"
                    )
                )

            if isinstance(original_config, str):
                # If original_config is a URL or filepath fetch the original_config dict
                original_config = fetch_original_config(original_config, local_files_only=local_files_only)

            config_mapping_kwargs = _get_mapping_function_kwargs(config_mapping_fn, **kwargs)
            diffusers_model_config = config_mapping_fn(
                original_config=original_config,
                checkpoint=checkpoint,
                **config_mapping_kwargs,
            )
        else:
            config_is_mapping = isinstance(config, Mapping)
            if config_is_mapping:
                diffusers_model_config = dict(config)
            elif config is not None:
                if isinstance(config, str):
                    default_pretrained_model_config_name = config
                else:
                    raise ValueError(
                        (
                            "Invalid `config` argument. Please provide a string representing a repo id"
                            "or path to a local Diffusers model repo."
                        )
                    )

            else:
                config = fetch_diffusers_config(checkpoint)
                default_pretrained_model_config_name = config["pretrained_model_name_or_path"]

                if "default_subfolder" in loading_config:
                    subfolder = loading_config["default_subfolder"]

                subfolder = subfolder or config.pop(
                    "subfolder", None
                )  # some configs contain a subfolder key, e.g. StableCascadeUNet

            if not config_is_mapping:
                diffusers_model_config = cls.load_config(
                    pretrained_model_name_or_path=default_pretrained_model_config_name,
                    subfolder=subfolder,
                    local_files_only=local_files_only,
                    token=token,
                    revision=config_revision,
                )
            expected_kwargs, optional_kwargs = cls._get_signature_keys(cls)

            # Map legacy kwargs to new kwargs
            if "legacy_kwargs" in loading_config:
                legacy_kwargs = loading_config["legacy_kwargs"]
                for legacy_key, new_key in legacy_kwargs.items():
                    if legacy_key in kwargs:
                        kwargs[new_key] = kwargs.pop(legacy_key)

            model_kwargs = {k: kwargs.get(k) for k in kwargs if k in expected_kwargs or k in optional_kwargs}
            diffusers_model_config.update(model_kwargs)

        ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
        with ctx():
            model = cls.from_config(diffusers_model_config)

        model_state_dict = model.state_dict()

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
        )
        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]

        else:
            keep_in_fp32_modules = []

        # Now that the model is loaded, we can determine the `device_map`
        device_map = _determine_device_map(model, device_map, None, torch_dtype, keep_in_fp32_modules, hf_quantizer)
        if device_map is not None:
            expanded_device_map = _expand_device_map(device_map, model_state_dict.keys())
            _caching_allocator_warmup(model, expanded_device_map, torch_dtype, hf_quantizer)

        checkpoint_mapping_kwargs = _get_mapping_function_kwargs(convert_model_checkpoint, **kwargs)
        checkpoint_mapping_kwargs["return_config"] = True

        if _should_convert_state_dict_to_diffusers(model_state_dict, checkpoint):
            diffusers_format_checkpoint, conversion_config = convert_model_checkpoint(
                config=dict(diffusers_model_config, _class_name=mapping_class_name),
                checkpoint=checkpoint,
                **checkpoint_mapping_kwargs,
            )
            if "original_format" in conversion_config:
                model.register_to_config(original_format=conversion_config["original_format"])
        else:
            diffusers_format_checkpoint = checkpoint

        if not diffusers_format_checkpoint:
            raise SingleFileComponentError(
                f"Failed to load {mapping_class_name}. Weights for this component appear to be missing in the checkpoint."
            )

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model,
                device_map=None,
                state_dict=diffusers_format_checkpoint,
                keep_in_fp32_modules=keep_in_fp32_modules,
            )

        device_map = None
        if low_cpu_mem_usage:
            param_device = torch.device(device) if device else torch.device("cpu")
            empty_state_dict = model.state_dict()
            unexpected_keys = [
                param_name for param_name in diffusers_format_checkpoint if param_name not in empty_state_dict
            ]
            device_map = {"": param_device}
            load_model_dict_into_meta(
                model,
                diffusers_format_checkpoint,
                dtype=torch_dtype,
                device_map=device_map,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_modules=keep_in_fp32_modules,
                unexpected_keys=unexpected_keys,
            )
            empty_device_cache()
        else:
            _, unexpected_keys = model.load_state_dict(diffusers_format_checkpoint, strict=False)

        if model._keys_to_ignore_on_load_unexpected is not None:
            for pat in model._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if torch_dtype is not None and hf_quantizer is None:
            model.to(torch_dtype)

        model.eval()

        if device_map is not None:
            device_map_kwargs = {"device_map": device_map}
            dispatch_model(model, **device_map_kwargs)

        return model
