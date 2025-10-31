# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
import functools
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Type, Union

import safetensors
import torch
import torch.utils.checkpoint
from huggingface_hub import DDUFEntry, create_repo, split_torch_state_dict_into_shards
from huggingface_hub.utils import validate_hf_hub_args
from torch import Tensor, nn
from typing_extensions import Self

from .. import __version__
from ..quantizers import DiffusersAutoQuantizer, DiffusersQuantizer
from ..quantizers.quantization_config import QuantizationMethod
from ..utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    HF_ENABLE_PARALLEL_LOADING,
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
    is_peft_available,
    is_torch_version,
    logging,
)
from ..utils.hub_utils import (
    PushToHubMixin,
    load_or_create_model_card,
    populate_model_card,
)
from ..utils.torch_utils import empty_device_cache
from ._modeling_parallel import ContextParallelConfig, ContextParallelModelPlan, ParallelConfig
from .model_loading_utils import (
    _caching_allocator_warmup,
    _determine_device_map,
    _expand_device_map,
    _fetch_index_file,
    _fetch_index_file_legacy,
    _load_shard_file,
    _load_shard_files_with_threadpool,
    load_state_dict,
)


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


logger = logging.get_logger(__name__)

_REGEX_SHARD = re.compile(r"(.*?)-\d{5}-of-\d{5}")

TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}

if is_torch_version(">=", "1.9.0"):
    _LOW_CPU_MEM_USAGE_DEFAULT = True
else:
    _LOW_CPU_MEM_USAGE_DEFAULT = False


if is_accelerate_available():
    import accelerate
    from accelerate import dispatch_model
    from accelerate.utils import load_offloaded_weights, save_offload_index


def get_parameter_device(parameter: torch.nn.Module) -> torch.device:
    from ..hooks.group_offloading import _get_group_onload_device

    try:
        # Try to get the onload device from the group offloading hook
        return _get_group_onload_device(parameter)
    except ValueError:
        pass

    try:
        # If the onload device is not available due to no group offloading hooks, try to get the device
        # from the first parameter or buffer
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
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    # 1. Check if we have attached any dtype modifying hooks (eg. layerwise casting)
    if isinstance(parameter, nn.Module):
        for name, submodule in parameter.named_modules():
            if not hasattr(submodule, "_diffusers_hook"):
                continue
            registry = submodule._diffusers_hook
            hook = registry.get_hook("layerwise_casting")
            if hook is not None:
                return hook.compute_dtype

    # 2. If no dtype modifying hooks are attached, return the dtype of the first floating point parameter/buffer
    last_dtype = None

    for name, param in parameter.named_parameters():
        last_dtype = param.dtype
        if (
            hasattr(parameter, "_keep_in_fp32_modules")
            and parameter._keep_in_fp32_modules
            and any(m in name for m in parameter._keep_in_fp32_modules)
        ):
            continue

        if param.is_floating_point():
            return param.dtype

    for buffer in parameter.buffers():
        last_dtype = buffer.dtype
        if buffer.is_floating_point():
            return buffer.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype


@contextmanager
def no_init_weights():
    """
    Context manager to globally disable weight initialization to speed up loading large models. To do that, all the
    torch.nn.init function are all replaced with skip.
    """

    def _skip_init(*args, **kwargs):
        pass

    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        # Restore the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)


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
    _skip_layerwise_casting_patterns = None
    _supports_group_offloading = True
    _repeated_blocks = []
    _parallel_config = None
    _cp_plan = None
    _skip_keys = None

    def __init__(self):
        super().__init__()

        self._gradient_checkpointing_func = None

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

    def enable_gradient_checkpointing(self, gradient_checkpointing_func: Optional[Callable] = None) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).

        Args:
            gradient_checkpointing_func (`Callable`, *optional*):
                The function to use for gradient checkpointing. If `None`, the default PyTorch checkpointing function
                is used (`torch.utils.checkpoint.checkpoint`).
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing. Please make sure to set the boolean attribute "
                f"`_supports_gradient_checkpointing` to `True` in the class definition."
            )

        if gradient_checkpointing_func is None:

            def _gradient_checkpointing_func(module, *args):
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                return torch.utils.checkpoint.checkpoint(
                    module.__call__,
                    *args,
                    **ckpt_kwargs,
                )

            gradient_checkpointing_func = _gradient_checkpointing_func

        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def disable_gradient_checkpointing(self) -> None:
        """
        Deactivates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if self._supports_gradient_checkpointing:
            self._set_gradient_checkpointing(enable=False)

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

    def set_use_xla_flash_attention(
        self, use_xla_flash_attention: bool, partition_spec: Optional[Callable] = None, **kwargs
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_xla_flash_attention method
        # gets the message
        def fn_recursive_set_flash_attention(module: torch.nn.Module):
            if hasattr(module, "set_use_xla_flash_attention"):
                module.set_use_xla_flash_attention(use_xla_flash_attention, partition_spec, **kwargs)

            for child in module.children():
                fn_recursive_set_flash_attention(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_flash_attention(module)

    def enable_xla_flash_attention(self, partition_spec: Optional[Callable] = None, **kwargs):
        r"""
        Enable the flash attention pallals kernel for torch_xla.
        """
        self.set_use_xla_flash_attention(True, partition_spec, **kwargs)

    def disable_xla_flash_attention(self):
        r"""
        Disable the flash attention pallals kernel for torch_xla.
        """
        self.set_use_xla_flash_attention(False)

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

        > [!WARNING] > ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient
        attention takes > precedent.

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

    def enable_layerwise_casting(
        self,
        storage_dtype: torch.dtype = torch.float8_e4m3fn,
        compute_dtype: Optional[torch.dtype] = None,
        skip_modules_pattern: Optional[Tuple[str, ...]] = None,
        skip_modules_classes: Optional[Tuple[Type[torch.nn.Module], ...]] = None,
        non_blocking: bool = False,
    ) -> None:
        r"""
        Activates layerwise casting for the current model.

        Layerwise casting is a technique that casts the model weights to a lower precision dtype for storage but
        upcasts them on-the-fly to a higher precision dtype for computation. This process can significantly reduce the
        memory footprint from model weights, but may lead to some quality degradation in the outputs. Most degradations
        are negligible, mostly stemming from weight casting in normalization and modulation layers.

        By default, most models in diffusers set the `_skip_layerwise_casting_patterns` attribute to ignore patch
        embedding, positional embedding and normalization layers. This is because these layers are most likely
        precision-critical for quality. If you wish to change this behavior, you can set the
        `_skip_layerwise_casting_patterns` attribute to `None`, or call
        [`~hooks.layerwise_casting.apply_layerwise_casting`] with custom arguments.

        Example:
            Using [`~models.ModelMixin.enable_layerwise_casting`]:

            ```python
            >>> from diffusers import CogVideoXTransformer3DModel

            >>> transformer = CogVideoXTransformer3DModel.from_pretrained(
            ...     "THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16
            ... )

            >>> # Enable layerwise casting via the model, which ignores certain modules by default
            >>> transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            ```

        Args:
            storage_dtype (`torch.dtype`):
                The dtype to which the model should be cast for storage.
            compute_dtype (`torch.dtype`):
                The dtype to which the model weights should be cast during the forward pass.
            skip_modules_pattern (`Tuple[str, ...]`, *optional*):
                A list of patterns to match the names of the modules to skip during the layerwise casting process. If
                set to `None`, default skip patterns are used to ignore certain internal layers of modules and PEFT
                layers.
            skip_modules_classes (`Tuple[Type[torch.nn.Module], ...]`, *optional*):
                A list of module classes to skip during the layerwise casting process.
            non_blocking (`bool`, *optional*, defaults to `False`):
                If `True`, the weight casting operations are non-blocking.
        """
        from ..hooks import apply_layerwise_casting

        user_provided_patterns = True
        if skip_modules_pattern is None:
            from ..hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

            skip_modules_pattern = DEFAULT_SKIP_MODULES_PATTERN
            user_provided_patterns = False
        if self._keep_in_fp32_modules is not None:
            skip_modules_pattern += tuple(self._keep_in_fp32_modules)
        if self._skip_layerwise_casting_patterns is not None:
            skip_modules_pattern += tuple(self._skip_layerwise_casting_patterns)
        skip_modules_pattern = tuple(set(skip_modules_pattern))

        if is_peft_available() and not user_provided_patterns:
            # By default, we want to skip all peft layers because they have a very low memory footprint.
            # If users want to apply layerwise casting on peft layers as well, they can utilize the
            # `~diffusers.hooks.layerwise_casting.apply_layerwise_casting` function which provides
            # them with more flexibility and control.

            from peft.tuners.loha.layer import LoHaLayer
            from peft.tuners.lokr.layer import LoKrLayer
            from peft.tuners.lora.layer import LoraLayer

            for layer in (LoHaLayer, LoKrLayer, LoraLayer):
                skip_modules_pattern += tuple(layer.adapter_layer_names)

        if compute_dtype is None:
            logger.info("`compute_dtype` not provided when enabling layerwise casting. Using dtype of the model.")
            compute_dtype = self.dtype

        apply_layerwise_casting(
            self, storage_dtype, compute_dtype, skip_modules_pattern, skip_modules_classes, non_blocking
        )

    def enable_group_offload(
        self,
        onload_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
        offload_type: str = "block_level",
        num_blocks_per_group: Optional[int] = None,
        non_blocking: bool = False,
        use_stream: bool = False,
        record_stream: bool = False,
        low_cpu_mem_usage=False,
        offload_to_disk_path: Optional[str] = None,
    ) -> None:
        r"""
        Activates group offloading for the current model.

        See [`~hooks.group_offloading.apply_group_offloading`] for more information.

        Example:

            ```python
            >>> from diffusers import CogVideoXTransformer3DModel

            >>> transformer = CogVideoXTransformer3DModel.from_pretrained(
            ...     "THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16
            ... )

            >>> transformer.enable_group_offload(
            ...     onload_device=torch.device("cuda"),
            ...     offload_device=torch.device("cpu"),
            ...     offload_type="leaf_level",
            ...     use_stream=True,
            ... )
            ```
        """
        from ..hooks import apply_group_offloading

        if getattr(self, "enable_tiling", None) is not None and getattr(self, "use_tiling", False) and use_stream:
            msg = (
                "Applying group offloading on autoencoders, with CUDA streams, may not work as expected if the first "
                "forward pass is executed with tiling enabled. Please make sure to either:\n"
                "1. Run a forward pass with small input shapes.\n"
                "2. Or, run a forward pass with tiling disabled (can still use small dummy inputs)."
            )
            logger.warning(msg)
        if not self._supports_group_offloading:
            raise ValueError(
                f"{self.__class__.__name__} does not support group offloading. Please make sure to set the boolean attribute "
                f"`_supports_group_offloading` to `True` in the class definition. If you believe this is a mistake, please "
                f"open an issue at https://github.com/huggingface/diffusers/issues."
            )
        apply_group_offloading(
            module=self,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type=offload_type,
            num_blocks_per_group=num_blocks_per_group,
            non_blocking=non_blocking,
            use_stream=use_stream,
            record_stream=record_stream,
            low_cpu_mem_usage=low_cpu_mem_usage,
            offload_to_disk_path=offload_to_disk_path,
        )

    def set_attention_backend(self, backend: str) -> None:
        """
        Set the attention backend for the model.

        Args:
            backend (`str`):
                The name of the backend to set. Must be one of the available backends defined in
                `AttentionBackendName`. Available backends can be found in
                `diffusers.attention_dispatch.AttentionBackendName`. Defaults to torch native scaled dot product
                attention as backend.
        """
        from .attention import AttentionModuleMixin
        from .attention_dispatch import AttentionBackendName, _check_attention_backend_requirements

        # TODO: the following will not be required when everything is refactored to AttentionModuleMixin
        from .attention_processor import Attention, MochiAttention

        logger.warning("Attention backends are an experimental feature and the API may be subject to change.")

        backend = backend.lower()
        available_backends = {x.value for x in AttentionBackendName.__members__.values()}
        if backend not in available_backends:
            raise ValueError(f"`{backend=}` must be one of the following: " + ", ".join(available_backends))
        backend = AttentionBackendName(backend)
        _check_attention_backend_requirements(backend)

        attention_classes = (Attention, MochiAttention, AttentionModuleMixin)
        for module in self.modules():
            if not isinstance(module, attention_classes):
                continue
            processor = module.processor
            if processor is None or not hasattr(processor, "_attention_backend"):
                continue
            processor._attention_backend = backend

    def reset_attention_backend(self) -> None:
        """
        Resets the attention backend for the model. Following calls to `forward` will use the environment default, if
        set, or the torch native scaled dot product attention.
        """
        from .attention import AttentionModuleMixin
        from .attention_processor import Attention, MochiAttention

        logger.warning("Attention backends are an experimental feature and the API may be subject to change.")

        attention_classes = (Attention, MochiAttention, AttentionModuleMixin)
        for module in self.modules():
            if not isinstance(module, attention_classes):
                continue
            processor = module.processor
            if processor is None or not hasattr(processor, "_attention_backend"):
                continue
            processor._attention_backend = None

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
            private = kwargs.pop("private", None)
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
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
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
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> Self:
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
            device_map (`Union[int, str, torch.device]` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device. Defaults to `None`, meaning that the model will be loaded on CPU.

                Examples:

                ```py
                >>> from diffusers import AutoModel
                >>> import torch

                >>> # This works.
                >>> model = AutoModel.from_pretrained(
                ...     "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", device_map="cuda"
                ... )
                >>> # This also works (integer accelerator device ID).
                >>> model = AutoModel.from_pretrained(
                ...     "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", device_map=0
                ... )
                >>> # Specifying a supported offloading strategy like "auto" also works.
                >>> model = AutoModel.from_pretrained(
                ...     "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", device_map="auto"
                ... )
                >>> # Specifying a dictionary as `device_map` also works.
                >>> model = AutoModel.from_pretrained(
                ...     "stabilityai/stable-diffusion-xl-base-1.0",
                ...     subfolder="unet",
                ...     device_map={"": torch.device("cuda")},
                ... )
                ```

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#the-devicemap). You
                can also refer to the [Diffusers-specific
                documentation](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#model-sharding)
                for more concrete examples.
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

        > [!TIP] > To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in
        with `hf > auth login`. You can also activate the special >
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a >
        firewalled environment.

        Example:

        ```py
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at stable-diffusion-v1-5/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
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
        offload_state_dict = kwargs.pop("offload_state_dict", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        quantization_config = kwargs.pop("quantization_config", None)
        dduf_entries: Optional[Dict[str, DDUFEntry]] = kwargs.pop("dduf_entries", None)
        disable_mmap = kwargs.pop("disable_mmap", False)
        parallel_config: Optional[Union[ParallelConfig, ContextParallelConfig]] = kwargs.pop("parallel_config", None)

        is_parallel_loading_enabled = HF_ENABLE_PARALLEL_LOADING
        if is_parallel_loading_enabled and not low_cpu_mem_usage:
            raise NotImplementedError("Parallel loading is not supported when not using `low_cpu_mem_usage`.")

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            torch_dtype = torch.float32
            logger.warning(
                f"Passed `torch_dtype` {torch_dtype} is not a `torch.dtype`. Defaulting to `torch.float32`."
            )

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

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }
        unused_kwargs = {}

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

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
            dduf_entries=dduf_entries,
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
            hf_quantizer.validate_environment(torch_dtype=torch_dtype, from_flax=from_flax, device_map=device_map)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            device_map = hf_quantizer.update_device_map(device_map)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `hf_quantizer` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = cls._keep_in_fp32_modules is not None and (
            hf_quantizer is None or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
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

        is_sharded = False
        resolved_model_file = None

        # Determine if we're loading from a directory of sharded checkpoints.
        sharded_metadata = None
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
            "dduf_entries": dduf_entries,
        }
        index_file = _fetch_index_file(**index_file_kwargs)
        # In case the index file was not found we still have to consider the legacy format.
        # this becomes applicable when the variant is not None.
        if variant is not None and (index_file is None or not os.path.exists(index_file)):
            index_file = _fetch_index_file_legacy(**index_file_kwargs)
        if index_file is not None and (dduf_entries or index_file.is_file()):
            is_sharded = True

        if is_sharded and from_flax:
            raise ValueError("Loading of sharded checkpoints is not supported when `from_flax=True`.")

        # load model
        if from_flax:
            resolved_model_file = _get_model_file(
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

            model = load_flax_checkpoint_in_pytorch_model(model, resolved_model_file)
        else:
            # in the case it is sharded, we have already the index
            if is_sharded:
                resolved_model_file, sharded_metadata = _get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    index_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder or "",
                    dduf_entries=dduf_entries,
                )
            elif use_safetensors:
                try:
                    resolved_model_file = _get_model_file(
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
                        dduf_entries=dduf_entries,
                    )

                except IOError as e:
                    logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                    if not allow_pickle:
                        raise
                    logger.warning(
                        "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                    )

            if resolved_model_file is None and not is_sharded:
                resolved_model_file = _get_model_file(
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
                    dduf_entries=dduf_entries,
                )

        if not isinstance(resolved_model_file, list):
            resolved_model_file = [resolved_model_file]

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is float8, we don't use _set_default_torch_dtype and we downcast after loading the model
        dtype_orig = None
        if torch_dtype is not None and not torch_dtype == getattr(torch, "float8_e4m3fn", None):
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(
                    f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
                )
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        init_contexts = [no_init_weights()]

        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights())

        with ContextManagers(init_contexts):
            model = cls.from_config(config, **unused_kwargs)

        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        state_dict = None
        if not is_sharded:
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_model_file[0], disable_mmap=disable_mmap, dduf_entries=dduf_entries)
            # We only fix it for non sharded checkpoints as we don't need it yet for sharded one.
            model._fix_state_dict_keys_on_load(state_dict)

        if is_sharded:
            loaded_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_keys = list(state_dict.keys())

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
            )

        # Now that the model is loaded, we can determine the device_map
        device_map = _determine_device_map(
            model, device_map, max_memory, torch_dtype, keep_in_fp32_modules, hf_quantizer
        )
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
            resolved_model_file,
            pretrained_model_name_or_path,
            loaded_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=keep_in_fp32_modules,
            dduf_entries=dduf_entries,
            is_parallel_loading_enabled=is_parallel_loading_enabled,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        # Dispatch model with hooks on all devices if necessary
        if device_map is not None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
            }
            dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if (
            torch_dtype is not None
            and torch_dtype == getattr(torch, "float8_e4m3fn", None)
            and hf_quantizer is None
            and not use_keep_in_fp32_modules
        ):
            model = model.to(torch_dtype)

        if hf_quantizer is not None:
            # We also make sure to purge `_pre_quantization_dtype` when we serialize
            # the model config because `_pre_quantization_dtype` is `torch.dtype`, not JSON serializable.
            model.register_to_config(_name_or_path=pretrained_model_name_or_path, _pre_quantization_dtype=torch_dtype)
        else:
            model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if parallel_config is not None:
            model.enable_parallelism(config=parallel_config)

        if output_loading_info:
            return model, loading_info

        return model

    # Adapted from `transformers`.
    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        from ..hooks.group_offloading import _is_group_offload_enabled

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

        # Checks if group offloading is enabled
        if _is_group_offload_enabled(self):
            logger.warning(
                f"The module '{self.__class__.__name__}' is group offloaded and moving it using `.cuda()` is not supported."
            )
            return self

        return super().cuda(*args, **kwargs)

    # Adapted from `transformers`.
    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        from ..hooks.group_offloading import _is_group_offload_enabled

        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present_in_args = "dtype" in kwargs

        # Try converting arguments to torch.device in case they are passed as strings
        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "is_quantized", False):
            if dtype_present_in_args:
                raise ValueError(
                    "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please "
                    "use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`"
                )

        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
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

        if _is_group_offload_enabled(self) and device_arg_or_kwarg_present:
            logger.warning(
                f"The module '{self.__class__.__name__}' is group offloaded and moving it using `.to()` is not supported."
            )
            return self

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

    def compile_repeated_blocks(self, *args, **kwargs):
        """
        Compiles *only* the frequently repeated sub-modules of a model (e.g. the Transformer layers) instead of
        compiling the entire model. This technique—often called **regional compilation** (see the PyTorch recipe
        https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) can reduce end-to-end compile time
        substantially, while preserving the runtime speed-ups you would expect from a full `torch.compile`.

        The set of sub-modules to compile is discovered by the presence of **`_repeated_blocks`** attribute in the
        model definition. Define this attribute on your model subclass as a list/tuple of class names (strings). Every
        module whose class name matches will be compiled.

        Once discovered, each matching sub-module is compiled by calling `submodule.compile(*args, **kwargs)`. Any
        positional or keyword arguments you supply to `compile_repeated_blocks` are forwarded verbatim to
        `torch.compile`.
        """
        repeated_blocks = getattr(self, "_repeated_blocks", None)

        if not repeated_blocks:
            raise ValueError(
                "`_repeated_blocks` attribute is empty. "
                f"Set `_repeated_blocks` for the class `{self.__class__.__name__}` to benefit from faster compilation. "
            )
        has_compiled_region = False
        for submod in self.modules():
            if submod.__class__.__name__ in repeated_blocks:
                submod.compile(*args, **kwargs)
                has_compiled_region = True

        if not has_compiled_region:
            raise ValueError(
                f"Regional compilation failed because {repeated_blocks} classes are not found in the model. "
            )

    def enable_parallelism(
        self,
        *,
        config: Union[ParallelConfig, ContextParallelConfig],
        cp_plan: Optional[Dict[str, ContextParallelModelPlan]] = None,
    ):
        logger.warning(
            "`enable_parallelism` is an experimental feature. The API may change in the future and breaking changes may be introduced at any time without warning."
        )

        if not torch.distributed.is_available() and not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed must be available and initialized before calling `enable_parallelism`."
            )

        from ..hooks.context_parallel import apply_context_parallel
        from .attention import AttentionModuleMixin
        from .attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
        from .attention_processor import Attention, MochiAttention

        if isinstance(config, ContextParallelConfig):
            config = ParallelConfig(context_parallel_config=config)

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device_type = torch._C._get_accelerator().type
        device_module = torch.get_device_module(device_type)
        device = torch.device(device_type, rank % device_module.device_count())

        attention_classes = (Attention, MochiAttention, AttentionModuleMixin)

        if config.context_parallel_config is not None:
            for module in self.modules():
                if not isinstance(module, attention_classes):
                    continue

                processor = module.processor
                if processor is None or not hasattr(processor, "_attention_backend"):
                    continue

                attention_backend = processor._attention_backend
                if attention_backend is None:
                    attention_backend, _ = _AttentionBackendRegistry.get_active_backend()
                else:
                    attention_backend = AttentionBackendName(attention_backend)

                if not _AttentionBackendRegistry._is_context_parallel_available(attention_backend):
                    compatible_backends = sorted(_AttentionBackendRegistry._supports_context_parallel)
                    raise ValueError(
                        f"Context parallelism is enabled but the attention processor '{processor.__class__.__name__}' "
                        f"is using backend '{attention_backend.value}' which does not support context parallelism. "
                        f"Please set a compatible attention backend: {compatible_backends} using `model.set_attention_backend()` before "
                        f"calling `enable_parallelism()`."
                    )

                # All modules use the same attention processor and backend. We don't need to
                # iterate over all modules after checking the first processor
                break

        mesh = None
        if config.context_parallel_config is not None:
            cp_config = config.context_parallel_config
            mesh = torch.distributed.device_mesh.init_device_mesh(
                device_type=device_type,
                mesh_shape=cp_config.mesh_shape,
                mesh_dim_names=cp_config.mesh_dim_names,
            )

        config.setup(rank, world_size, device, mesh=mesh)
        self._parallel_config = config

        for module in self.modules():
            if not isinstance(module, attention_classes):
                continue
            processor = module.processor
            if processor is None or not hasattr(processor, "_parallel_config"):
                continue
            processor._parallel_config = config

        if config.context_parallel_config is not None:
            if cp_plan is None and self._cp_plan is None:
                raise ValueError(
                    "`cp_plan` must be provided either as an argument or set in the model's `_cp_plan` attribute."
                )
            cp_plan = cp_plan if cp_plan is not None else self._cp_plan
            apply_context_parallel(self, config.context_parallel_config, cp_plan)

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: OrderedDict,
        resolved_model_file: List[str],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        loaded_keys: List[str],
        ignore_mismatched_sizes: bool = False,
        assign_to_params_buffers: bool = False,
        hf_quantizer: Optional[DiffusersQuantizer] = None,
        low_cpu_mem_usage: bool = True,
        dtype: Optional[Union[str, torch.dtype]] = None,
        keep_in_fp32_modules: Optional[List[str]] = None,
        device_map: Union[str, int, torch.device, Dict[str, Union[int, str, torch.device]]] = None,
        offload_state_dict: Optional[bool] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = None,
        dduf_entries: Optional[Dict[str, DDUFEntry]] = None,
        is_parallel_loading_enabled: Optional[bool] = False,
    ):
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        if hf_quantizer is not None:
            missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix="")
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        mismatched_keys = []
        error_msgs = []

        # Deal with offload
        if device_map is not None and "disk" in device_map.values():
            if offload_folder is None:
                raise ValueError(
                    "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                    " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                    " offers the weights in this format."
                )
            else:
                os.makedirs(offload_folder, exist_ok=True)
            if offload_state_dict is None:
                offload_state_dict = True

        # If a device map has been used, we can speedup the load time by warming up the device caching allocator.
        # If we don't warmup, each tensor allocation on device calls to the allocator for memory (effectively, a
        # lot of individual calls to device malloc). We can, however, preallocate the memory required by the
        # tensors using their expected shape and not performing any initialization of the memory (empty data).
        # When the actual device allocations happen, the allocator already has a pool of unused device memory
        # that it can re-use for faster loading of the model.
        if device_map is not None:
            expanded_device_map = _expand_device_map(device_map, expected_keys)
            _caching_allocator_warmup(model, expanded_device_map, dtype, hf_quantizer)

        offload_index = {} if device_map is not None and "disk" in device_map.values() else None
        state_dict_folder, state_dict_index = None, None
        if offload_state_dict:
            state_dict_folder = tempfile.mkdtemp()
            state_dict_index = {}

        if state_dict is not None:
            # load_state_dict will manage the case where we pass a dict instead of a file
            # if state dict is not None, it means that we don't need to read the files from resolved_model_file also
            resolved_model_file = [state_dict]

        # Prepare the loading function sharing the attributes shared between them.
        load_fn = functools.partial(
            _load_shard_files_with_threadpool if is_parallel_loading_enabled else _load_shard_file,
            model=model,
            model_state_dict=model_state_dict,
            device_map=device_map,
            dtype=dtype,
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=keep_in_fp32_modules,
            dduf_entries=dduf_entries,
            loaded_keys=loaded_keys,
            unexpected_keys=unexpected_keys,
            offload_index=offload_index,
            offload_folder=offload_folder,
            state_dict_index=state_dict_index,
            state_dict_folder=state_dict_folder,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        if is_parallel_loading_enabled:
            offload_index, state_dict_index, _mismatched_keys, _error_msgs = load_fn(resolved_model_file)
            error_msgs += _error_msgs
            mismatched_keys += _mismatched_keys
        else:
            shard_files = resolved_model_file
            if len(resolved_model_file) > 1:
                shard_files = logging.tqdm(resolved_model_file, desc="Loading checkpoint shards")

            for shard_file in shard_files:
                offload_index, state_dict_index, _mismatched_keys, _error_msgs = load_fn(shard_file)
                error_msgs += _error_msgs
                mismatched_keys += _mismatched_keys

        empty_device_cache()

        if offload_index is not None and len(offload_index) > 0:
            save_offload_index(offload_index, offload_folder)
            offload_index = None

            if offload_state_dict:
                load_offloaded_weights(model, state_dict_index, state_dict_folder)
                shutil.rmtree(state_dict_folder)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
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

        return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs

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
        Get the modules of the model that should not be split when using device_map. We iterate through the modules to
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

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model
        under specific dtype.

        Args:
            dtype (`torch.dtype`):
                a floating dtype to set to.

        Returns:
            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

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

        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
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

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func: Callable = torch.utils.checkpoint.checkpoint
    ) -> None:
        is_gradient_checkpointing_set = False

        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                logger.debug(f"Setting `gradient_checkpointing={enable}` for '{name}'")
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"The module {self.__class__.__name__} does not support gradient checkpointing. Please make sure to "
                f"use a module that supports gradient checkpointing by creating a boolean attribute `gradient_checkpointing`."
            )

    def _fix_state_dict_keys_on_load(self, state_dict: OrderedDict) -> None:
        """
        This function fix the state dict of the model to take into account some changes that were made in the model
        architecture:
        - deprecated attention blocks (happened before we introduced sharded checkpoint,
        so this is why we apply this method only when loading non sharded checkpoints for now)
        """
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
        return state_dict


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

        if remapped_class is cls:
            return super(LegacyModelMixin, remapped_class).from_pretrained(
                pretrained_model_name_or_path, **kwargs_copy
            )
        else:
            return remapped_class.from_pretrained(pretrained_model_name_or_path, **kwargs_copy)
