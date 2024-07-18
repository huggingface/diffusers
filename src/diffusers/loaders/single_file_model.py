# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import importlib
import inspect
import re
from contextlib import nullcontext
from typing import Optional

from huggingface_hub.utils import validate_hf_hub_args

from ..utils import deprecate, is_accelerate_available, logging
from .single_file_utils import (
    SingleFileComponentError,
    convert_animatediff_checkpoint_to_diffusers,
    convert_controlnet_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_sd3_transformer_checkpoint_to_diffusers,
    convert_stable_cascade_unet_single_file_to_diffusers,
    create_controlnet_diffusers_config_from_ldm,
    create_unet_diffusers_config_from_ldm,
    create_vae_diffusers_config_from_ldm,
    fetch_diffusers_config,
    fetch_original_config,
    load_single_file_checkpoint,
)


logger = logging.get_logger(__name__)


if is_accelerate_available():
    from accelerate import init_empty_weights

    from ..models.modeling_utils import load_model_dict_into_meta


SINGLE_FILE_LOADABLE_CLASSES = {
    "StableCascadeUNet": {
        "checkpoint_mapping_fn": convert_stable_cascade_unet_single_file_to_diffusers,
    },
    "UNet2DConditionModel": {
        "checkpoint_mapping_fn": convert_ldm_unet_checkpoint,
        "config_mapping_fn": create_unet_diffusers_config_from_ldm,
        "default_subfolder": "unet",
        "legacy_kwargs": {
            "num_in_channels": "in_channels",  # Legacy kwargs supported by `from_single_file` mapped to new args
        },
    },
    "AutoencoderKL": {
        "checkpoint_mapping_fn": convert_ldm_vae_checkpoint,
        "config_mapping_fn": create_vae_diffusers_config_from_ldm,
        "default_subfolder": "vae",
    },
    "ControlNetModel": {
        "checkpoint_mapping_fn": convert_controlnet_checkpoint,
        "config_mapping_fn": create_controlnet_diffusers_config_from_ldm,
    },
    "SD3Transformer2DModel": {
        "checkpoint_mapping_fn": convert_sd3_transformer_checkpoint_to_diffusers,
        "default_subfolder": "transformer",
    },
    "MotionAdapter": {
        "checkpoint_mapping_fn": convert_animatediff_checkpoint_to_diffusers,
    },
}


def _get_single_file_loadable_mapping_class(cls):
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    for loadable_class_str in SINGLE_FILE_LOADABLE_CLASSES:
        loadable_class = getattr(diffusers_module, loadable_class_str)

        if issubclass(cls, loadable_class):
            return loadable_class_str

    return None


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
    def from_single_file(cls, pretrained_model_link_or_path_or_dict: Optional[str] = None, **kwargs):
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
            config (`str`, *optional*):
                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline hosted
                  on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline component
                  configs in Diffusers format.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            original_config (`str`, *optional*):
                Dict or path to a yaml file containing the configuration for the model in its original format.
                    If a dict is provided, it will be used to initialize the model configuration.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
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
        # if class_name not in SINGLE_FILE_LOADABLE_CLASSES:
        if mapping_class_name is None:
            raise ValueError(
                f"FromOriginalModelMixin is currently only compatible with {', '.join(SINGLE_FILE_LOADABLE_CLASSES.keys())}"
            )

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
        torch_dtype = kwargs.pop("torch_dtype", None)

        if isinstance(pretrained_model_link_or_path_or_dict, dict):
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
            )

        mapping_functions = SINGLE_FILE_LOADABLE_CLASSES[mapping_class_name]

        checkpoint_mapping_fn = mapping_functions["checkpoint_mapping_fn"]
        if original_config:
            if "config_mapping_fn" in mapping_functions:
                config_mapping_fn = mapping_functions["config_mapping_fn"]
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
                original_config=original_config, checkpoint=checkpoint, **config_mapping_kwargs
            )
        else:
            if config:
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

                if "default_subfolder" in mapping_functions:
                    subfolder = mapping_functions["default_subfolder"]

                subfolder = subfolder or config.pop(
                    "subfolder", None
                )  # some configs contain a subfolder key, e.g. StableCascadeUNet

            diffusers_model_config = cls.load_config(
                pretrained_model_name_or_path=default_pretrained_model_config_name,
                subfolder=subfolder,
                local_files_only=local_files_only,
            )
            expected_kwargs, optional_kwargs = cls._get_signature_keys(cls)

            # Map legacy kwargs to new kwargs
            if "legacy_kwargs" in mapping_functions:
                legacy_kwargs = mapping_functions["legacy_kwargs"]
                for legacy_key, new_key in legacy_kwargs.items():
                    if legacy_key in kwargs:
                        kwargs[new_key] = kwargs.pop(legacy_key)

            model_kwargs = {k: kwargs.get(k) for k in kwargs if k in expected_kwargs or k in optional_kwargs}
            diffusers_model_config.update(model_kwargs)

        checkpoint_mapping_kwargs = _get_mapping_function_kwargs(checkpoint_mapping_fn, **kwargs)
        diffusers_format_checkpoint = checkpoint_mapping_fn(
            config=diffusers_model_config, checkpoint=checkpoint, **checkpoint_mapping_kwargs
        )
        if not diffusers_format_checkpoint:
            raise SingleFileComponentError(
                f"Failed to load {mapping_class_name}. Weights for this component appear to be missing in the checkpoint."
            )

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = cls.from_config(diffusers_model_config)

        if is_accelerate_available():
            unexpected_keys = load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)

        else:
            _, unexpected_keys = model.load_state_dict(diffusers_format_checkpoint, strict=False)

        if model._keys_to_ignore_on_load_unexpected is not None:
            for pat in model._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

        if torch_dtype is not None:
            model.to(torch_dtype)

        model.eval()

        return model
