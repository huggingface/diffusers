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

import os
from typing import Optional, Union

from huggingface_hub.utils import validate_hf_hub_args

from ..configuration_utils import ConfigMixin
from ..utils import logging
from ..utils.dynamic_modules_utils import get_class_from_dynamic_module, resolve_trust_remote_code


logger = logging.get_logger(__name__)


class AutoModel(ConfigMixin):
    config_name = "config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path: Optional[Union[str, os.PathLike]] = None, **kwargs):
        r"""
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device. Defaults to `None`, meaning that the model will be loaded on CPU.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if `device_map` contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.
            disable_mmap ('bool', *optional*, defaults to 'False'):
                Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.
            trust_remote_cocde (`bool`, *optional*, defaults to `False`):
                Whether to trust remote code

        > [!TIP] > To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in
        with `hf > auth login`. You can also activate the special >
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a >
        firewalled environment.

        Example:

        ```py
        from diffusers import AutoModel

        unet = AutoModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at stable-diffusion-v1-5/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        subfolder = kwargs.pop("subfolder", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name, None) for name in hub_kwargs_names}

        # load_config_kwargs uses the same hub kwargs minus subfolder and resume_download
        load_config_kwargs = {k: v for k, v in hub_kwargs.items() if k not in ["subfolder"]}

        library = None
        orig_class_name = None

        # Always attempt to fetch model_index.json first
        try:
            cls.config_name = "model_index.json"
            config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)

            if subfolder is not None and subfolder in config:
                library, orig_class_name = config[subfolder]
                load_config_kwargs.update({"subfolder": subfolder})

        except EnvironmentError as e:
            logger.debug(e)

        # Unable to load from model_index.json so fallback to loading from config
        if library is None and orig_class_name is None:
            cls.config_name = "config.json"
            config = cls.load_config(pretrained_model_or_path, subfolder=subfolder, **load_config_kwargs)

            if "_class_name" in config:
                # If we find a class name in the config, we can try to load the model as a diffusers model
                orig_class_name = config["_class_name"]
                library = "diffusers"
                load_config_kwargs.update({"subfolder": subfolder})
            elif "model_type" in config:
                orig_class_name = "AutoModel"
                library = "transformers"
                load_config_kwargs.update({"subfolder": "" if subfolder is None else subfolder})
            else:
                raise ValueError(f"Couldn't find model associated with the config file at {pretrained_model_or_path}.")

        has_remote_code = "auto_map" in config and cls.__name__ in config["auto_map"]
        trust_remote_code = resolve_trust_remote_code(trust_remote_code, pretrained_model_or_path, has_remote_code)
        if not has_remote_code and trust_remote_code:
            raise ValueError(
                "Selected model repository does not happear to have any custom code or does not have a valid `config.json` file."
            )

        if has_remote_code and trust_remote_code:
            class_ref = config["auto_map"][cls.__name__]
            module_file, class_name = class_ref.split(".")
            module_file = module_file + ".py"
            model_cls = get_class_from_dynamic_module(
                pretrained_model_or_path,
                subfolder=subfolder,
                module_file=module_file,
                class_name=class_name,
                **hub_kwargs,
            )
        else:
            from ..pipelines.pipeline_loading_utils import ALL_IMPORTABLE_CLASSES, get_class_obj_and_candidates

            model_cls, _ = get_class_obj_and_candidates(
                library_name=library,
                class_name=orig_class_name,
                importable_classes=ALL_IMPORTABLE_CLASSES,
                pipelines=None,
                is_pipeline_module=False,
            )

        if model_cls is None:
            raise ValueError(f"AutoModel can't find a model linked to {orig_class_name}.")

        kwargs = {**load_config_kwargs, **kwargs}
        return model_cls.from_pretrained(pretrained_model_or_path, **kwargs)
