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
from typing import Optional

from huggingface_hub.utils import validate_hf_hub_args

from .single_file_utils import (
    create_diffusers_controlnet_from_ldm,
    create_diffusers_unet_from_ldm,
    create_diffusers_unet_from_stable_cascade,
    create_diffusers_vae_from_ldm,
    load_single_file_model_checkpoint,
)


SINGLE_FILE_LOADABLE_CLASSES = {
    "StableCascadeUNet": create_diffusers_unet_from_stable_cascade,
    "UNet2DConditionModel": create_diffusers_unet_from_ldm,
    "AutoencoderKL": create_diffusers_vae_from_ldm,
    "ControlNetModel": create_diffusers_controlnet_from_ldm,
}


class FromOriginalModelMixin:
    """
    Load pretrained weights saved in the `.ckpt` or `.safetensors` format into a [`AutoencoderKL`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path: Optional[str] = None, **kwargs):
        r"""
        Instantiate a [`AutoencoderKL`] from pretrained ControlNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            config (`str`, *optional*):
                Dict or path to a json file containing the configuration for the model in diffusers format.
                    If a dict is provided, it will be used to initialize the model configuration.
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
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
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

        ```
        """

        class_name = cls.__name__
        if class_name not in SINGLE_FILE_LOADABLE_CLASSES:
            raise ValueError(
                f"FromOriginalModelMixin is currently only compatible with {', '.join(SINGLE_FILE_LOADABLE_CLASSES.keys())}"
            )

        checkpoint = kwargs.pop("checkpoint", None)
        if pretrained_model_link_or_path is None and checkpoint is None:
            raise ValueError(
                "Please provide either a `pretrained_model_link_or_path` or a `checkpoint` to load the model from."
            )

        config = kwargs.pop("config", None)
        original_config = kwargs.pop("original_config", None)

        if config is not None and original_config is not None:
            raise ValueError(
                "`from_single_file` cannot accept both `config` and `original_config` arguments. Please provide only one of these arguments"
            )

        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        if checkpoint is None:
            checkpoint = load_single_file_model_checkpoint(
                pretrained_model_link_or_path,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )

        model_loading_fn = SINGLE_FILE_LOADABLE_CLASSES[class_name]
        model = model_loading_fn(
            cls,
            checkpoint=checkpoint,
            config=config,
            original_config=original_config,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        return model
