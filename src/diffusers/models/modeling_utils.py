# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import inspect
import itertools
import json
import os
import re
from collections import OrderedDict
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import safetensors
import torch
from huggingface_hub import create_repo, split_torch_state_dict_into_shards
from huggingface_hub.utils import validate_hf_hub_args
from torch import Tensor, nn

from .. import __version__
from ..quantizers import DiffusersAutoQuantizer, DiffusersQuantizer
from ..quantizers.quantization_config import QuantizationMethod
from ..utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
    deprecate,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_bitsandbytes_version,
    is_torch_version,
    logging,
)
from ..utils.hub_utils import (
    PushToHubMixin,
    load_or_create_model_card,
    populate_model_card,
)
from .model_loading_utils import (
    _determine_device_map,
    _fetch_index_file,
    _fetch_index_file_legacy,
    _load_state_dict_into_model,
    _merge_sharded_checkpoints,
    load_model_dict_into_meta,
    load_state_dict,
)


logger = logging.get_logger(__name__)

_REGEX_SHARD = re.compile(r"(.*?)-\d{5}-of-\d{5}")


if is_torch_version(">=", "1.9.0"):
    _LOW_CPU_MEM_USAGE_DEFAULT = True
else:
    _LOW_CPU_MEM_USAGE_DEFAULT = False


if is_accelerate_available():
    import accelerate


def get_parameter_device(parameter: torch.nn.Module) -> torch.device:
    try:
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        try:
            return next(parameter.buffers()).dtype
        except StopIteration:
            # For torch.nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = parameter._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype


class ModelMixin(torch.nn.Module, PushToHubMixin):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the model configuration and provides methods for loading, downloading and
    saving models.

        - **config_name** ([`str`]) -- Filename to save a model to when calling [`~models.ModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = None
    _no_split_modules = None
    _keep_in_fp32_modules = None

    def __init__(self):
        super().__init__()

    def __getattr__(self, name: str) -> Any:
        """The only reason we overwrite `getattr` here is to gracefully deprecate accessing
        config attributes directly. See https://github.com/huggingface/diffusers/pull/3129 We need to overwrite
        __getattr__ here in addition so that we don't trigger `torch.nn.Module`'s __getattr__':
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'unet.config.{name}'."
            deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False, stacklevel=3)
            return self._internal_dict[name]

        # call PyTorch's https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        return super().__getattr__(name)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def enable_gradient_checkpointing(self) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self) -> None:
        """
        Deactivates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def set_use_npu_flash_attention(self, valid: bool) -> None:
        r"""
        Set the switch for the npu flash attention.
        """

        def fn_recursive_set_npu_flash_attention(module: torch.nn.Module):
            if hasattr(module, "set_use_npu_flash_attention"):
                module.set_use_npu_flash_attention(valid)

            for child in module.children():
                fn_recursive_set_npu_flash_attention(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_npu_flash_attention(module)

    def enable_npu_flash_attention(self) -> None:
        r"""
        Enable npu flash attention from torch_npu

        """
        self.set_use_npu_flash_attention(True)

    def disable_npu_flash_attention(self) -> None:
        r"""
        disable npu flash attention from torch_npu

        """
        self.set_use_npu_flash_attention(False)

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None) -> None:
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up during
        inference. Speed up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import UNet2DConditionModel
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> model = UNet2DConditionModel.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", subfolder="unet", torch_dtype=torch.float16
        ... )
        >>> model = model.to("cuda")
        >>> model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self) -> None:
        r"""
        Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "10GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory so that it can be reloaded using the
        [`~models.ModelMixin.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a model and its configuration file to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            max_shard_size (`int` or `str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5GB"`).
                If expressed as an integer, the unit is bytes. Note that this limit will be decreased after a certain
                period of time (starting from Oct 2024) to allow users to upgrade to the latest version of `diffusers`.
                This is to establish a common default size for this argument across different libraries in the Hugging
                Face ecosystem (`transformers`, and `accelerate`, for example).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        hf_quantizer = getattr(self, "hf_quantizer", None)
        if hf_quantizer is not None:
            quantization_serializable = (
                hf_quantizer is not None
                and isinstance(hf_quantizer, DiffusersQuantizer)
                and hf_quantizer.is_serializable
            )
            if not quantization_serializable:
                raise ValueError(
                    f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                    " the logger on the traceback to understand the reason why the quantized model is not serializable."
                )

        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)
        weights_name_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
            ".safetensors", "{suffix}.safetensors"
        )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # Save the model
        state_dict = model_to_save.state_dict()

        # Save the model
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, max_shard_size=max_shard_size, filename_pattern=weights_name_pattern
        )

        # Clean the folder from a previous save
        if is_main_process:
            for filename in os.listdir(save_directory):
                if filename in state_dict_split.filename_to_tensors.keys():
                    continue
                full_filename = os.path.join(save_directory, filename)
                if not os.path.isfile(full_filename):
                    continue
                weights_without_ext = weights_name_pattern.replace(".bin", "").replace(".safetensors", "")
                weights_without_ext = weights_without_ext.replace("{suffix}", "")
                filename_without_ext = filename.replace(".bin", "").replace(".safetensors", "")
                # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                if (
                    filename.startswith(weights_without_ext)
                    and _REGEX_SHARD.fullmatch(filename_without_ext) is not None
                ):
                    os.remove(full_filename)

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            filepath = os.path.join(save_directory, filename)
            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safetensors.torch.save_file(shard, filepath, metadata={"format": "pt"})
            else:
                torch.save(shard, filepath)

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
        else:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token)
            model_card = populate_model_card(model_card)
            model_card.save(Path(save_directory, "README.md").as_posix())

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    def dequantize(self):
        """
        Potentially dequantize the model in case it has been quantized by a quantization method that support
        dequantization.
        """
        hf_quantizer = getattr(self, "hf_quantizer", None)

        if hf_quantizer is None:
            raise ValueError("You need to first quantize your model in order to dequantize it")

        return hf_quantizer.dequantize(self)

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
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
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
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

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        Example:

        ```py
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        quantization_config = kwargs.pop("quantization_config", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set"
                " `device_map=None`. You can install accelerate with `pip install accelerate`."
            )

        # Check if we can handle device_map and dispatching the weights
        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if low_cpu_mem_usage:
            if device_map is not None and not is_torch_version(">=", "1.10"):
                # The max memory utils require PyTorch >= 1.10 to have torch.cuda.mem_get_info.
                raise ValueError("`low_cpu_mem_usage` and `device_map` require PyTorch >= 1.10.")

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )
        # no in-place modification of the original config.
        config = copy.deepcopy(config)

        # determine initial quantization config.
        #######################################
        pre_quantized = "quantization_config" in config and config["quantization_config"] is not None
        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config["quantization_config"] = DiffusersAutoQuantizer.merge_quantization_configs(
                    config["quantization_config"], quantization_config
                )
            else:
                config["quantization_config"] = quantization_config
            hf_quantizer = DiffusersAutoQuantizer.from_config(
                config["quantization_config"], pre_quantized=pre_quantized
            )
        else:
            hf_quantizer = None

        if hf_quantizer is not None:
            if device_map is not None:
                raise NotImplementedError(
                    "Currently, `device_map` is automatically inferred for quantized models. Support for providing `device_map` as an input will be added in the future."
                )
            hf_quantizer.validate_environment(torch_dtype=torch_dtype, from_flax=from_flax, device_map=device_map)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `hf_quantizer` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
        )
        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]

            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
        else:
            keep_in_fp32_modules = []
        #######################################

        # Determine if we're loading from a directory of sharded checkpoints.
        is_sharded = False
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file_kwargs = {
            "is_local": is_local,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "subfolder": subfolder or "",
            "use_safetensors": use_safetensors,
            "cache_dir": cache_dir,
            "variant": variant,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "user_agent": user_agent,
            "commit_hash": commit_hash,
        }
        index_file = _fetch_index_file(**index_file_kwargs)
        # In case the index file was not found we still have to consider the legacy format.
        # this becomes applicable when the variant is not None.
        if variant is not None and (index_file is None or not os.path.exists(index_file)):
            index_file = _fetch_index_file_legacy(**index_file_kwargs)
        if index_file is not None and index_file.is_file():
            is_sharded = True

        if is_sharded and from_flax:
            raise ValueError("Loading of sharded checkpoints is not supported when `from_flax=True`.")

        # load model
        model_file = None
        if from_flax:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=FLAX_WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            model = cls.from_config(config, **unused_kwargs)

            # Convert the weights
            from .modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model

            model = load_flax_checkpoint_in_pytorch_model(model, model_file)
        else:
            if is_sharded:
                sharded_ckpt_cached_folder, sharded_metadata = _get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    index_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder or "",
                )
                if hf_quantizer is not None:
                    model_file = _merge_sharded_checkpoints(sharded_ckpt_cached_folder, sharded_metadata)
                    logger.info("Merged sharded checkpoints as `hf_quantizer` is not None.")
                    is_sharded = False

            elif use_safetensors and not is_sharded:
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )

                except IOError as e:
                    logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                    if not allow_pickle:
                        raise
                    logger.warning(
                        "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                    )

            if model_file is None and not is_sharded:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )

            if low_cpu_mem_usage:
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **unused_kwargs)

                if hf_quantizer is not None:
                    hf_quantizer.preprocess_model(
                        model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
                    )

                # if device_map is None, load the state dict and move the params from meta device to the cpu
                if device_map is None and not is_sharded:
                    # `torch.cuda.current_device()` is fine here when `hf_quantizer` is not None.
                    # It would error out during the `validate_environment()` call above in the absence of cuda.
                    is_quant_method_bnb = (
                        getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
                    )
                    if hf_quantizer is None:
                        param_device = "cpu"
                    # TODO (sayakpaul,  SunMarc): remove this after model loading refactor
                    elif is_quant_method_bnb:
                        param_device = torch.cuda.current_device()
                    state_dict = load_state_dict(model_file, variant=variant)
                    model._convert_deprecated_attention_blocks(state_dict)

                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if hf_quantizer is not None:
                        missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix="")
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_name_or_path,
                        hf_quantizer=hf_quantizer,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        logger.warning(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )

                else:  # else let accelerate handle loading and dispatching.
                    # Load weights and dispatch according to the device_map
                    # by default the device_map is None and the weights are loaded on the CPU
                    force_hook = True
                    device_map = _determine_device_map(
                        model, device_map, max_memory, torch_dtype, keep_in_fp32_modules, hf_quantizer
                    )
                    if device_map is None and is_sharded:
                        # we load the parameters on the cpu
                        device_map = {"": "cpu"}
                        force_hook = False
                    try:
                        accelerate.load_checkpoint_and_dispatch(
                            model,
                            model_file if not is_sharded else index_file,
                            device_map,
                            max_memory=max_memory,
                            offload_folder=offload_folder,
                            offload_state_dict=offload_state_dict,
                            dtype=torch_dtype,
                            force_hooks=force_hook,
                            strict=True,
                        )
                    except AttributeError as e:
                        # When using accelerate loading, we do not have the ability to load the state
                        # dict and rename the weight names manually. Additionally, accelerate skips
                        # torch loading conventions and directly writes into `module.{_buffers, _parameters}`
                        # (which look like they should be private variables?), so we can't use the standard hooks
                        # to rename parameters on load. We need to mimic the original weight names so the correct
                        # attributes are available. After we have loaded the weights, we convert the deprecated
                        # names to the new non-deprecated names. Then we _greatly encourage_ the user to convert
                        # the weights so we don't have to do this again.

                        if "'Attention' object has no attribute" in str(e):
                            logger.warning(
                                f"Taking `{str(e)}` while using `accelerate.load_checkpoint_and_dispatch` to mean {pretrained_model_name_or_path}"
                                " was saved with deprecated attention block weight names. We will load it with the deprecated attention block"
                                " names and convert them on the fly to the new attention block format. Please re-save the model after this conversion,"
                                " so we don't have to do the on the fly renaming in the future. If the model is from a hub checkpoint,"
                                " please also re-upload it or open a PR on the original repository."
                            )
                            model._temp_convert_self_to_deprecated_attention_blocks()
                            accelerate.load_checkpoint_and_dispatch(
                                model,
                                model_file if not is_sharded else index_file,
                                device_map,
                                max_memory=max_memory,
                                offload_folder=offload_folder,
                                offload_state_dict=offload_state_dict,
                                dtype=torch_dtype,
                                force_hooks=force_hook,
                                strict=True,
                            )
                            model._undo_temp_convert_self_to_deprecated_attention_blocks()
                        else:
                            raise e

                loading_info = {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            else:
                model = cls.from_config(config, **unused_kwargs)

                state_dict = load_state_dict(model_file, variant=variant)
                model._convert_deprecated_attention_blocks(state_dict)

                model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                    model,
                    state_dict,
                    model_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )

                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        # When using `use_keep_in_fp32_modules` if we do a global `to()` here, then we will
        # completely lose the effectivity of `use_keep_in_fp32_modules`.
        elif torch_dtype is not None and hf_quantizer is None and not use_keep_in_fp32_modules:
            model = model.to(torch_dtype)

        if hf_quantizer is not None:
            # We also make sure to purge `_pre_quantization_dtype` when we serialize
            # the model config because `_pre_quantization_dtype` is `torch.dtype`, not JSON serializable.
            model.register_to_config(_name_or_path=pretrained_model_name_or_path, _pre_quantization_dtype=torch_dtype)
        else:
            model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model

    # Adapted from `transformers`.
    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "Calling `cuda()` is not supported for `8-bit` quantized models. "
                    " Please use the model as it is, since the model has already been set to the correct devices."
                )
            elif is_bitsandbytes_version("<", "0.43.2"):
                raise ValueError(
                    "Calling `cuda()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        return super().cuda(*args, **kwargs)

    # Adapted from `transformers`.
    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a bitsandbytes model in a new `dtype`. Make sure to load the model using `from_pretrained` using the"
                    " desired `dtype` by passing the correct `torch_dtype` argument."
                )

            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "`.to` is not supported for `8-bit` bitsandbytes models. Please use the model as it is, since the"
                    " model has already been set to the correct devices and casted to the correct `dtype`."
                )
            elif is_bitsandbytes_version("<", "0.43.2"):
                raise ValueError(
                    "Calling `to()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        return super().to(*args, **kwargs)

    # Taken from `transformers`.
    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been cast to the correct `dtype`."
            )
        else:
            return super().half(*args)

    # Taken from `transformers`.
    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been cast to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: OrderedDict,
        resolved_archive_file,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ignore_mismatched_sizes: bool = False,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = list(state_dict.keys())

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        return expected_modules, optional_parameters

    # Adapted from `transformers` modeling_utils.py
    def _get_no_split_modules(self, device_map: str):
        """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            # if the module does not appear in _no_split_modules, we also check the children
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, ModelMixin):
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                            "class needs to implement the `_no_split_modules` attribute."
                        )
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                modules_to_check += list(module.children())
        return list(_no_split_modules)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (trainable or non-embedding) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters.
            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embedding parameters.

        Returns:
            `int`: The number of parameters.

        Example:

        ```py
        from diffusers import UNet2DConditionModel

        model_id = "runwayml/stable-diffusion-v1-5"
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        unet.num_parameters(only_trainable=True)
        859520964
        ```
        """
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        if is_loaded_in_4bit:
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                raise ValueError(
                    "bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong"
                    " make sure to install bitsandbytes with `pip install bitsandbytes`. You also need a GPU. "
                )

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []

        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, "element_size"):
                        num_bytes = param.element_size()
                    elif hasattr(param, "quant_storage"):
                        num_bytes = param.quant_storage.itemsize
                    else:
                        num_bytes = 1
                    total_numel.append(param.numel() * 2 * num_bytes)
                else:
                    total_numel.append(param.numel())

        return sum(total_numel)

    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem

    def _convert_deprecated_attention_blocks(self, state_dict: OrderedDict) -> None:
        deprecated_attention_block_paths = []

        def recursive_find_attn_block(name, module):
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_paths.append(name)

            for sub_name, sub_module in module.named_children():
                sub_name = sub_name if name == "" else f"{name}.{sub_name}"
                recursive_find_attn_block(sub_name, sub_module)

        recursive_find_attn_block("", self)

        # NOTE: we have to check if the deprecated parameters are in the state dict
        # because it is possible we are loading from a state dict that was already
        # converted

        for path in deprecated_attention_block_paths:
            # group_norm path stays the same

            # query -> to_q
            if f"{path}.query.weight" in state_dict:
                state_dict[f"{path}.to_q.weight"] = state_dict.pop(f"{path}.query.weight")
            if f"{path}.query.bias" in state_dict:
                state_dict[f"{path}.to_q.bias"] = state_dict.pop(f"{path}.query.bias")

            # key -> to_k
            if f"{path}.key.weight" in state_dict:
                state_dict[f"{path}.to_k.weight"] = state_dict.pop(f"{path}.key.weight")
            if f"{path}.key.bias" in state_dict:
                state_dict[f"{path}.to_k.bias"] = state_dict.pop(f"{path}.key.bias")

            # value -> to_v
            if f"{path}.value.weight" in state_dict:
                state_dict[f"{path}.to_v.weight"] = state_dict.pop(f"{path}.value.weight")
            if f"{path}.value.bias" in state_dict:
                state_dict[f"{path}.to_v.bias"] = state_dict.pop(f"{path}.value.bias")

            # proj_attn -> to_out.0
            if f"{path}.proj_attn.weight" in state_dict:
                state_dict[f"{path}.to_out.0.weight"] = state_dict.pop(f"{path}.proj_attn.weight")
            if f"{path}.proj_attn.bias" in state_dict:
                state_dict[f"{path}.to_out.0.bias"] = state_dict.pop(f"{path}.proj_attn.bias")

    def _temp_convert_self_to_deprecated_attention_blocks(self) -> None:
        deprecated_attention_block_modules = []

        def recursive_find_attn_block(module):
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_modules.append(module)

            for sub_module in module.children():
                recursive_find_attn_block(sub_module)

        recursive_find_attn_block(self)

        for module in deprecated_attention_block_modules:
            module.query = module.to_q
            module.key = module.to_k
            module.value = module.to_v
            module.proj_attn = module.to_out[0]

            # We don't _have_ to delete the old attributes, but it's helpful to ensure
            # that _all_ the weights are loaded into the new attributes and we're not
            # making an incorrect assumption that this model should be converted when
            # it really shouldn't be.
            del module.to_q
            del module.to_k
            del module.to_v
            del module.to_out

    def _undo_temp_convert_self_to_deprecated_attention_blocks(self) -> None:
        deprecated_attention_block_modules = []

        def recursive_find_attn_block(module) -> None:
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_modules.append(module)

            for sub_module in module.children():
                recursive_find_attn_block(sub_module)

        recursive_find_attn_block(self)

        for module in deprecated_attention_block_modules:
            module.to_q = module.query
            module.to_k = module.key
            module.to_v = module.value
            module.to_out = nn.ModuleList([module.proj_attn, nn.Dropout(module.dropout)])

            del module.query
            del module.key
            del module.value
            del module.proj_attn


class LegacyModelMixin(ModelMixin):
    r"""
    A subclass of `ModelMixin` to resolve class mapping from legacy classes (like `Transformer2DModel`) to more
    pipeline-specific classes (like `DiTTransformer2DModel`).
    """

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # To prevent dependency import problem.
        from .model_loading_utils import _fetch_remapped_cls_from_config

        # Create a copy of the kwargs so that we don't mess with the keyword arguments in the downstream calls.
        kwargs_copy = kwargs.copy()

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, _, _ = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )
        # resolve remapping
        remapped_class = _fetch_remapped_cls_from_config(config, cls)

        return remapped_class.from_pretrained(pretrained_model_name_or_path, **kwargs_copy)
