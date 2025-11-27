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
import fnmatch
import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin

import httpx
import numpy as np
import PIL.Image
import requests
import torch
from huggingface_hub import (
    DDUFEntry,
    ModelCard,
    create_repo,
    hf_hub_download,
    model_info,
    read_dduf_file,
    snapshot_download,
)
from huggingface_hub.utils import HfHubHTTPError, OfflineModeIsEnabled, validate_hf_hub_args
from packaging import version
from tqdm.auto import tqdm
from typing_extensions import Self

from .. import __version__
from ..configuration_utils import ConfigMixin
from ..models import AutoencoderKL
from ..models.attention_processor import FusedAttnProcessor2_0
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, ModelMixin
from ..quantizers import PipelineQuantizationConfig
from ..quantizers.bitsandbytes.utils import _check_bnb_status
from ..schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from ..utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    BaseOutput,
    PushToHubMixin,
    _get_detailed_type,
    _is_valid_type,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    is_hpu_available,
    is_torch_npu_available,
    is_torch_version,
    is_transformers_version,
    logging,
    numpy_to_pil,
)
from ..utils.hub_utils import _check_legacy_sharding_variant_format, load_or_create_model_card, populate_model_card
from ..utils.torch_utils import empty_device_cache, get_device, is_compiled_module


if is_torch_npu_available():
    import torch_npu  # noqa: F401

from .pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    CONNECTED_PIPES_KEYS,
    CUSTOM_PIPELINE_FILE_NAME,
    LOADABLE_CLASSES,
    _download_dduf_file,
    _fetch_class_library_tuple,
    _get_custom_components_and_folders,
    _get_custom_pipeline_class,
    _get_final_device_map,
    _get_ignore_patterns,
    _get_pipeline_class,
    _identify_model_variants,
    _maybe_raise_error_for_incorrect_transformers,
    _maybe_raise_warning_for_inpainting,
    _maybe_warn_for_wrong_component_in_quant_config,
    _resolve_custom_pipeline_and_cls,
    _unwrap_model,
    _update_init_kwargs_with_connected_pipeline,
    filter_model_files,
    load_sub_model,
    maybe_raise_or_warn,
    variant_compatible_siblings,
    warn_deprecated_model_variant,
)


if is_accelerate_available():
    import accelerate


LIBRARIES = []
for library in LOADABLE_CLASSES:
    LIBRARIES.append(library)

SUPPORTED_DEVICE_MAP = ["balanced"] + [get_device()]

logger = logging.get_logger(__name__)


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised audio samples of a NumPy array of shape `(batch_size, num_channels, sample_rate)`.
    """

    audios: np.ndarray


class DeprecatedPipelineMixin:
    """
    A mixin that can be used to mark a pipeline as deprecated.

    Pipelines inheriting from this mixin will raise a warning when instantiated, indicating that they are deprecated
    and won't receive updates past the specified version. Tests will be skipped for pipelines that inherit from this
    mixin.

    Example usage:
    ```python
    class MyDeprecatedPipeline(DeprecatedPipelineMixin, DiffusionPipeline):
        _last_supported_version = "0.20.0"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    ```
    """

    # Override this in the inheriting class to specify the last version that will support this pipeline
    _last_supported_version = None

    def __init__(self, *args, **kwargs):
        # Get the class name for the warning message
        class_name = self.__class__.__name__

        # Get the last supported version or use the current version if not specified
        version_info = getattr(self.__class__, "_last_supported_version", __version__)

        # Raise a warning that this pipeline is deprecated
        logger.warning(
            f"The {class_name} has been deprecated and will not receive bug fixes or feature updates after Diffusers version {version_info}. "
        )

        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)


class DiffusionPipeline(ConfigMixin, PushToHubMixin):
    r"""
    Base class for all pipelines.

    [`DiffusionPipeline`] stores all components (models, schedulers, and processors) for diffusion pipelines and
    provides methods for loading, downloading and saving models. It also includes methods to:

        - move all PyTorch modules to the device of your choice
        - enable/disable the progress bar for the denoising iteration

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.
        - **_optional_components** (`List[str]`) -- List of all optional components that don't have to be passed to the
          pipeline to function (should be overridden by subclasses).
    """

    config_name = "model_index.json"
    model_cpu_offload_seq = None
    hf_device_map = None
    _optional_components = []
    _exclude_from_cpu_offload = []
    _load_connected_pipes = False
    _is_onnx = False

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__ and hasattr(self.config, name):
            # We need to overwrite the config if name exists in config
            if isinstance(getattr(self.config, name), (tuple, list)):
                if value is not None and self.config[name][0] is not None:
                    class_library_tuple = _fetch_class_library_tuple(value)
                else:
                    class_library_tuple = (None, None)

                self.register_to_config(**{name: class_library_tuple})
            else:
                self.register_to_config(**{name: value})

        super().__setattr__(name, value)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Optional[Union[int, str]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~DiffusionPipeline.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a pipeline to. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            max_shard_size (`int` or `str`, defaults to `None`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5GB"`).
                If expressed as an integer, the unit is bytes. Note that this limit will be decreased after a certain
                period of time (starting from Oct 2024) to allow users to upgrade to the latest version of `diffusers`.
                This is to establish a common default size for this argument across different libraries in the Hugging
                Face ecosystem (`transformers`, and `accelerate`, for example).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).

            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            # Dynamo wraps the original model in a private class.
            # I didn't find a public API to get the original class.
            if is_compiled_module(sub_model):
                sub_model = _unwrap_model(sub_model)
                model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)
                else:
                    logger.info(
                        f"{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}"
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                logger.warning(
                    f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved."
                )
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters
            save_method_accept_max_shard_size = "max_shard_size" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant
            if save_method_accept_max_shard_size and max_shard_size is not None:
                # max_shard_size is expected to not be None in ModelMixin
                save_kwargs["max_shard_size"] = max_shard_size

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

        # finally save the config
        self.save_config(save_directory)

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token, is_pipeline=True)
            model_card = populate_model_card(model_card)
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    def to(self, *args, **kwargs) -> Self:
        r"""
        Performs Pipeline dtype and/or device conversion. A torch.dtype and torch.device are inferred from the
        arguments of `self.to(*args, **kwargs).`

        > [!TIP] > If the pipeline already has the correct torch.dtype and torch.device, then it is returned as is.
        Otherwise, > the returned pipeline is a copy of self with the desired torch.dtype and torch.device.


        Here are the ways to call `to`:

        - `to(dtype, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
        - `to(device, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        - `to(device=None, dtype=None, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the
          specified [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) and
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)

        Arguments:
            dtype (`torch.dtype`, *optional*):
                Returns a pipeline with the specified
                [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
            device (`torch.Device`, *optional*):
                Returns a pipeline with the specified
                [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
            silence_dtype_warnings (`str`, *optional*, defaults to `False`):
                Whether to omit warnings if the target `dtype` is not compatible with the target `device`.

        Returns:
            [`DiffusionPipeline`]: The pipeline converted to specified `dtype` and/or `dtype`.
        """
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        silence_dtype_warnings = kwargs.pop("silence_dtype_warnings", False)

        dtype_arg = None
        device_arg = None
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg
        device_type = torch.device(device).type if device is not None else None
        pipeline_has_bnb = any(any((_check_bnb_status(module))) for _, module in self.components.items())

        # throw warning if pipeline is in "offloaded"-mode but user tries to manually set to GPU.
        def module_is_sequentially_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.14.0"):
                return False

            _, _, is_loaded_in_8bit_bnb = _check_bnb_status(module)

            if is_loaded_in_8bit_bnb:
                return False

            return hasattr(module, "_hf_hook") and (
                isinstance(module._hf_hook, accelerate.hooks.AlignDevicesHook)
                or hasattr(module._hf_hook, "hooks")
                and isinstance(module._hf_hook.hooks[0], accelerate.hooks.AlignDevicesHook)
            )

        def module_is_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.17.0.dev0"):
                return False

            return hasattr(module, "_hf_hook") and isinstance(module._hf_hook, accelerate.hooks.CpuOffload)

        # .to("cuda") would raise an error if the pipeline is sequentially offloaded, so we raise our own to make it clearer
        pipeline_is_sequentially_offloaded = any(
            module_is_sequentially_offloaded(module) for _, module in self.components.items()
        )

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline which doesn't allow explicit device placement using `to()`. You can call `reset_device_map()` to remove the existing device map from the pipeline."
            )

        if device_type in ["cuda", "xpu"]:
            if pipeline_is_sequentially_offloaded and not pipeline_has_bnb:
                raise ValueError(
                    "It seems like you have activated sequential model offloading by calling `enable_sequential_cpu_offload`, but are now attempting to move the pipeline to GPU. This is not compatible with offloading. Please, move your pipeline `.to('cpu')` or consider removing the move altogether if you use sequential offloading."
                )
            # PR: https://github.com/huggingface/accelerate/pull/3223/
            elif pipeline_has_bnb and is_accelerate_version("<", "1.1.0.dev0"):
                raise ValueError(
                    "You are trying to call `.to('cuda')` on a pipeline that has models quantized with `bitsandbytes`. Your current `accelerate` installation does not support it. Please upgrade the installation."
                )

        # Display a warning in this case (the operation succeeds but the benefits are lost)
        pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in self.components.items())
        if pipeline_is_offloaded and device_type in ["cuda", "xpu"]:
            logger.warning(
                f"It seems like you have activated model offloading by calling `enable_model_cpu_offload`, but are now manually moving the pipeline to GPU. It is strongly recommended against doing so as memory gains from offloading are likely to be lost. Offloading automatically takes care of moving the individual components {', '.join(self.components.keys())} to GPU when needed. To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading."
            )

        # Enable generic support for Intel Gaudi accelerator using GPU/HPU migration
        if device_type == "hpu" and kwargs.pop("hpu_migration", True) and is_hpu_available():
            os.environ["PT_HPU_GPU_MIGRATION"] = "1"
            logger.debug("Environment variable set: PT_HPU_GPU_MIGRATION=1")

            import habana_frameworks.torch  # noqa: F401

            # HPU hardware check
            if not (hasattr(torch, "hpu") and torch.hpu.is_available()):
                raise ValueError("You are trying to call `.to('hpu')` but HPU device is unavailable.")

            os.environ["PT_HPU_MAX_COMPOUND_OP_SIZE"] = "1"
            logger.debug("Environment variable set: PT_HPU_MAX_COMPOUND_OP_SIZE=1")

            if dtype in (torch.bfloat16, None) and kwargs.pop("sdp_on_bf16", True):
                if hasattr(torch._C, "_set_math_sdp_allow_fp16_bf16_reduction"):
                    torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)
                    logger.warning(
                        "Enabled SDP with BF16 precision on HPU. To disable, please use `.to('hpu', sdp_on_bf16=False)`"
                    )

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
        for module in modules:
            _, is_loaded_in_4bit_bnb, is_loaded_in_8bit_bnb = _check_bnb_status(module)
            is_group_offloaded = self._maybe_raise_error_if_group_offload_active(module=module)

            if (is_loaded_in_4bit_bnb or is_loaded_in_8bit_bnb) and dtype is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in `bitsandbytes` {'4bit' if is_loaded_in_4bit_bnb else '8bit'} and conversion to {dtype} is not supported. Module is still in {'4bit' if is_loaded_in_4bit_bnb else '8bit'} precision."
                )

            if is_loaded_in_8bit_bnb and device is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in `bitsandbytes` 8bit and moving it to {device} via `.to()` is not supported. Module is still on {module.device}."
                )

            # Note: we also handle this at the ModelMixin level. The reason for doing it here too is that modeling
            # components can be from outside diffusers too, but still have group offloading enabled.
            if (
                self._maybe_raise_error_if_group_offload_active(raise_error=False, module=module)
                and device is not None
            ):
                logger.warning(
                    f"The module '{module.__class__.__name__}' is group offloaded and moving it to {device} via `.to()` is not supported."
                )

            # This can happen for `transformer` models. CPU placement was added in
            # https://github.com/huggingface/transformers/pull/33122. So, we guard this accordingly.
            if is_loaded_in_4bit_bnb and device is not None and is_transformers_version(">", "4.44.0"):
                module.to(device=device)
            elif not is_loaded_in_4bit_bnb and not is_loaded_in_8bit_bnb and not is_group_offloaded:
                module.to(device, dtype)

            if (
                module.dtype == torch.float16
                and str(device) in ["cpu"]
                and not silence_dtype_warnings
                and not is_offloaded
            ):
                logger.warning(
                    "Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device. It"
                    " is not recommended to move them to `cpu` as running them will fail. Please make"
                    " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                    " support for`float16` operations on this device in PyTorch. Please, remove the"
                    " `torch_dtype=torch.float16` argument, or use another device for inference."
                )
        return self

    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.device

        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns:
            `torch.dtype`: The torch dtype on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.dtype

        return torch.float32

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> Self:
        r"""
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at stable-diffusion-v1-5/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing a dduf file
            torch_dtype (`torch.dtype` or `dict[str, Union[str, torch.dtype]]`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. To load submodels with
                different dtype pass a `dict` (for example `{'transformer': torch.bfloat16, 'vae': torch.float16}`).
                Set the default dtype for unspecified components with `default` (for example `{'transformer':
                torch.bfloat16, 'default': torch.float16}`). If a component is not specified and no default is set,
                `torch.float32` is used.
            custom_pipeline (`str`, *optional*):

                > [!WARNING] > 🧪 This is an experimental feature and may change in the future.

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers
                version.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str`, *optional*):
                Strategy that dictates how the different components of a pipeline should be placed on available
                devices. Currently, only "balanced" `device_map` is supported. Check out
                [this](https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement)
                to know more.
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            dduf_file(`str`, *optional*):
                Load weights from the specified dduf file.

        > [!TIP] > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in
        with `hf > auth login`.

        Examples:

        ```py
        >>> from diffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
        # Copy the kwargs to re-use during loading connected pipeline.
        kwargs_copied = kwargs.copy()

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        provider_options = kwargs.pop("provider_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        dduf_file = kwargs.pop("dduf_file", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        quantization_config = kwargs.pop("quantization_config", None)

        if torch_dtype is not None and not isinstance(torch_dtype, dict) and not isinstance(torch_dtype, torch.dtype):
            torch_dtype = torch.float32
            logger.warning(
                f"Passed `torch_dtype` {torch_dtype} is not a `torch.dtype`. Defaulting to `torch.float32`."
            )

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if quantization_config is not None and not isinstance(quantization_config, PipelineQuantizationConfig):
            raise ValueError("`quantization_config` must be an instance of `PipelineQuantizationConfig`.")

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`."
            )

        if device_map is not None and not isinstance(device_map, str):
            raise ValueError("`device_map` must be a string.")

        if device_map is not None and device_map not in SUPPORTED_DEVICE_MAP:
            raise NotImplementedError(
                f"{device_map} not supported. Supported strategies are: {', '.join(SUPPORTED_DEVICE_MAP)}"
            )

        if device_map is not None and device_map in SUPPORTED_DEVICE_MAP:
            if is_accelerate_version("<", "0.28.0"):
                raise NotImplementedError("Device placement requires `accelerate` version `0.28.0` or later.")

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        if dduf_file:
            if custom_pipeline:
                raise NotImplementedError("Custom pipelines are not supported with DDUF at the moment.")
            if load_connected_pipeline:
                raise NotImplementedError("Connected pipelines are not supported with DDUF at the moment.")

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                from_flax=from_flax,
                use_safetensors=use_safetensors,
                use_onnx=use_onnx,
                custom_pipeline=custom_pipeline,
                custom_revision=custom_revision,
                variant=variant,
                dduf_file=dduf_file,
                load_connected_pipeline=load_connected_pipeline,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        # The variant filenames can have the legacy sharding checkpoint format that we check and throw
        # a warning if detected.
        if variant is not None and _check_legacy_sharding_variant_format(folder=cached_folder, variant=variant):
            warn_msg = (
                f"Warning: The repository contains sharded checkpoints for variant '{variant}' maybe in a deprecated format. "
                "Please check your files carefully:\n\n"
                "- Correct format example: diffusion_pytorch_model.fp16-00003-of-00003.safetensors\n"
                "- Deprecated format example: diffusion_pytorch_model-00001-of-00002.fp16.safetensors\n\n"
                "If you find any files in the deprecated format:\n"
                "1. Remove all existing checkpoint files for this variant.\n"
                "2. Re-obtain the correct files by running `save_pretrained()`.\n\n"
                "This will ensure you're using the most up-to-date and compatible checkpoint format."
            )
            logger.warning(warn_msg)

        dduf_entries = None
        if dduf_file:
            dduf_file_path = os.path.join(cached_folder, dduf_file)
            dduf_entries = read_dduf_file(dduf_file_path)
            # The reader contains already all the files needed, no need to check it again
            cached_folder = ""

        config_dict = cls.load_config(cached_folder, dduf_entries=dduf_entries)

        if dduf_file:
            _maybe_raise_error_for_incorrect_transformers(config_dict)

        # pop out "_ignore_files" as it is only needed for download
        config_dict.pop("_ignore_files", None)

        # 2. Define which model components should load variants
        # We retrieve the information by matching whether variant model checkpoints exist in the subfolders.
        # Example: `diffusion_pytorch_model.safetensors` -> `diffusion_pytorch_model.fp16.safetensors`
        # with variant being `"fp16"`.
        model_variants = _identify_model_variants(folder=cached_folder, variant=variant, config=config_dict)
        if len(model_variants) == 0 and variant is not None:
            error_message = f"You are trying to load the model files of the `variant={variant}`, but no such modeling files are available."
            raise ValueError(error_message)

        # 3. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        custom_pipeline, custom_class_name = _resolve_custom_pipeline_and_cls(
            folder=cached_folder, config=config_dict, custom_pipeline=custom_pipeline
        )
        pipeline_class = _get_pipeline_class(
            cls,
            config=config_dict,
            load_connected_pipeline=load_connected_pipeline,
            custom_pipeline=custom_pipeline,
            class_name=custom_class_name,
            cache_dir=cache_dir,
            revision=custom_revision,
        )

        if device_map is not None and pipeline_class._load_connected_pipes:
            raise NotImplementedError("`device_map` is not yet supported for connected pipelines.")

        # DEPRECATED: To be removed in 1.0.0
        # we are deprecating the `StableDiffusionInpaintPipelineLegacy` pipeline which gets loaded
        # when a user requests for a `StableDiffusionInpaintPipeline` with `diffusers` version being <= 0.5.1.
        _maybe_raise_warning_for_inpainting(
            pipeline_class=pipeline_class,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config_dict,
        )

        # 4. Define expected modules given pipeline signature
        # and define non-None initialized modules (=`init_kwargs`)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        expected_types = pipeline_class._get_signature_types()
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs and make sure that optional component modules are filtered out
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Special case: safety_checker must be loaded separately when using `from_flax`
        if from_flax and "safety_checker" in init_dict and "safety_checker" not in passed_class_obj:
            raise NotImplementedError(
                "The safety checker cannot be automatically loaded when loading weights `from_flax`."
                " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
                " separately if you need it."
            )

        # 5. Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        # import it here to avoid circular import
        from diffusers import pipelines

        # 6. device map delegation
        final_device_map = None
        if device_map is not None:
            final_device_map = _get_final_device_map(
                device_map=device_map,
                pipeline_class=pipeline_class,
                passed_class_obj=passed_class_obj,
                init_dict=init_dict,
                library=library,
                max_memory=max_memory,
                torch_dtype=torch_dtype,
                cached_folder=cached_folder,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )

        # 7. Load each module in the pipeline
        current_device_map = None
        _maybe_warn_for_wrong_component_in_quant_config(init_dict, quantization_config)
        for name, (library_name, class_name) in logging.tqdm(init_dict.items(), desc="Loading pipeline components..."):
            # 7.1 device_map shenanigans
            if final_device_map is not None:
                if isinstance(final_device_map, dict) and len(final_device_map) > 0:
                    component_device = final_device_map.get(name, None)
                    if component_device is not None:
                        current_device_map = {"": component_device}
                    else:
                        current_device_map = None
                elif isinstance(final_device_map, str):
                    current_device_map = final_device_map

            # 7.2 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            class_name = class_name[4:] if class_name.startswith("Flax") else class_name

            # 7.3 Define all importable classes
            is_pipeline_module = hasattr(pipelines, library_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            loaded_sub_model = None

            # 7.4 Use passed sub model or load class_name from library_name
            if name in passed_class_obj:
                # if the model is in a pipeline module, then we load it from the pipeline
                # check that passed_class_obj has correct parent class
                maybe_raise_or_warn(
                    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
                )

                loaded_sub_model = passed_class_obj[name]
            else:
                # load sub model
                sub_model_dtype = (
                    torch_dtype.get(name, torch_dtype.get("default", torch.float32))
                    if isinstance(torch_dtype, dict)
                    else torch_dtype
                )
                loaded_sub_model = load_sub_model(
                    library_name=library_name,
                    class_name=class_name,
                    importable_classes=importable_classes,
                    pipelines=pipelines,
                    is_pipeline_module=is_pipeline_module,
                    pipeline_class=pipeline_class,
                    torch_dtype=sub_model_dtype,
                    provider=provider,
                    sess_options=sess_options,
                    device_map=current_device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    model_variants=model_variants,
                    name=name,
                    from_flax=from_flax,
                    variant=variant,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    cached_folder=cached_folder,
                    use_safetensors=use_safetensors,
                    dduf_entries=dduf_entries,
                    provider_options=provider_options,
                    quantization_config=quantization_config,
                )
                logger.info(
                    f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
                )

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        # 8. Handle connected pipelines.
        if pipeline_class._load_connected_pipes and os.path.isfile(os.path.join(cached_folder, "README.md")):
            init_kwargs = _update_init_kwargs_with_connected_pipeline(
                init_kwargs=init_kwargs,
                passed_pipe_kwargs=passed_pipe_kwargs,
                passed_class_objs=passed_class_obj,
                folder=cached_folder,
                **kwargs_copied,
            )

        # 9. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - set(optional_kwargs)
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 10. Type checking init arguments
        for kw, arg in init_kwargs.items():
            # Too complex to validate with type annotation alone
            if "scheduler" in kw:
                continue
            # Many tokenizer annotations don't include its "Fast" variant, so skip this
            # e.g T5Tokenizer but not T5TokenizerFast
            elif "tokenizer" in kw:
                continue
            elif (
                arg is not None  # Skip if None
                and not expected_types[kw] == (inspect.Signature.empty,)  # Skip if no type annotations
                and not _is_valid_type(arg, expected_types[kw])  # Check type
            ):
                logger.warning(f"Expected types for {kw}: {expected_types[kw]}, got {_get_detailed_type(arg)}.")

        # 11. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # 12. Save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        if device_map is not None:
            setattr(model, "hf_device_map", final_device_map)
        if quantization_config is not None:
            setattr(model, "quantization_config", quantization_config)
        return model

    @property
    def name_or_path(self) -> str:
        return getattr(self.config, "_name_or_path", None)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        from ..hooks.group_offloading import _get_group_onload_device

        # When apply group offloading at the leaf_level, we're in the same situation as accelerate's sequential
        # offloading. We need to return the onload device of the group offloading hooks so that the intermediates
        # required for computation (latents, prompt embeddings, etc.) can be created on the correct device.
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue
            try:
                return _get_group_onload_device(model)
            except ValueError:
                pass

        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def remove_all_hooks(self):
        r"""
        Removes all hooks that were added when using `enable_sequential_cpu_offload` or `enable_model_cpu_offload`.
        """
        for _, model in self.components.items():
            if isinstance(model, torch.nn.Module) and hasattr(model, "_hf_hook"):
                accelerate.hooks.remove_hook_from_module(model, recurse=True)
        self._all_hooks = []

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = None):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the accelerator when its
        `forward` method is called, and the model remains in accelerator until the next model runs. Memory savings are
        lower than with `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution
        of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to None):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                automatically detect the available accelerator and use.
        """
        self._maybe_raise_error_if_group_offload_active(raise_error=True)

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_model_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_model_cpu_offload()`."
            )

        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        self.remove_all_hooks()

        if device is None:
            device = get_device()
            if device == "cpu":
                raise RuntimeError("`enable_model_cpu_offload` requires accelerator, but not found")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        self.to("cpu", silence_dtype_warnings=True)
        empty_device_cache(device.type)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)

            if not isinstance(model, torch.nn.Module):
                continue

            # This is because the model would already be placed on a CUDA device.
            _, _, is_loaded_in_8bit_bnb = _check_bnb_status(model)
            if is_loaded_in_8bit_bnb:
                logger.info(
                    f"Skipping the hook placement for the {model.__class__.__name__} as it is loaded in `bitsandbytes` 8bit."
                )
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Method that performs the following:
        - Offloads all components.
        - Removes all model hooks that were added when using `enable_model_cpu_offload`, and then applies them again.
          In case the model has not been offloaded, this function is a no-op.
        - Resets stateful diffusers hooks of denoiser components if they were added with
          [`~hooks.HookRegistry.register_hook`].

        Make sure to add this function to the end of the `__call__` function of your pipeline so that it functions
        correctly when applying `enable_model_cpu_offload`.
        """
        for component in self.components.values():
            if hasattr(component, "_reset_stateful_cache"):
                component._reset_stateful_cache()

        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload(device=getattr(self, "_offload_device", "cuda"))

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = None):
        r"""
        Offloads all models to CPU using 🤗 Accelerate, significantly reducing memory usage. When called, the state
        dicts of all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are saved to CPU
        and then moved to `torch.device('meta')` and loaded to accelerator only when their specific submodule has its
        `forward` method called. Offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to None):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                automatically detect the available accelerator and use.
        """
        self._maybe_raise_error_if_group_offload_active(raise_error=True)

        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")
        self.remove_all_hooks()

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_sequential_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_sequential_cpu_offload()`."
            )

        if device is None:
            device = get_device()
            if device == "cpu":
                raise RuntimeError("`enable_sequential_cpu_offload` requires accelerator, but not found")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        if self.device.type != "cpu":
            orig_device_type = self.device.type
            self.to("cpu", silence_dtype_warnings=True)
            empty_device_cache(orig_device_type)

        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                # make sure to offload buffers if not all high level weights
                # are of type nn.Module
                offload_buffers = len(model._parameters) > 0
                cpu_offload(model, device, offload_buffers=offload_buffers)

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
        exclude_modules: Optional[Union[str, List[str]]] = None,
    ) -> None:
        r"""
        Applies group offloading to the internal layers of a torch.nn.Module. To understand what group offloading is,
        and where it is beneficial, we need to first provide some context on how other supported offloading methods
        work.

        Typically, offloading is done at two levels:
        - Module-level: In Diffusers, this can be enabled using the `ModelMixin::enable_model_cpu_offload()` method. It
        works by offloading each component of a pipeline to the CPU for storage, and onloading to the accelerator
        device when needed for computation. This method is more memory-efficient than keeping all components on the
        accelerator, but the memory requirements are still quite high. For this method to work, one needs memory
        equivalent to size of the model in runtime dtype + size of largest intermediate activation tensors to be able
        to complete the forward pass.
        - Leaf-level: In Diffusers, this can be enabled using the `ModelMixin::enable_sequential_cpu_offload()` method.
          It
        works by offloading the lowest leaf-level parameters of the computation graph to the CPU for storage, and
        onloading only the leafs to the accelerator device for computation. This uses the lowest amount of accelerator
        memory, but can be slower due to the excessive number of device synchronizations.

        Group offloading is a middle ground between the two methods. It works by offloading groups of internal layers,
        (either `torch.nn.ModuleList` or `torch.nn.Sequential`). This method uses lower memory than module-level
        offloading. It is also faster than leaf-level/sequential offloading, as the number of device synchronizations
        is reduced.

        Another supported feature (for CUDA devices with support for asynchronous data transfer streams) is the ability
        to overlap data transfer and computation to reduce the overall execution time compared to sequential
        offloading. This is enabled using layer prefetching with streams, i.e., the layer that is to be executed next
        starts onloading to the accelerator device while the current layer is being executed - this increases the
        memory requirements slightly. Note that this implementation also supports leaf-level offloading but can be made
        much faster when using streams.

        Args:
            onload_device (`torch.device`):
                The device to which the group of modules are onloaded.
            offload_device (`torch.device`, defaults to `torch.device("cpu")`):
                The device to which the group of modules are offloaded. This should typically be the CPU. Default is
                CPU.
            offload_type (`str` or `GroupOffloadingType`, defaults to "block_level"):
                The type of offloading to be applied. Can be one of "block_level" or "leaf_level". Default is
                "block_level".
            offload_to_disk_path (`str`, *optional*, defaults to `None`):
                The path to the directory where parameters will be offloaded. Setting this option can be useful in
                limited RAM environment settings where a reasonable speed-memory trade-off is desired.
            num_blocks_per_group (`int`, *optional*):
                The number of blocks per group when using offload_type="block_level". This is required when using
                offload_type="block_level".
            non_blocking (`bool`, defaults to `False`):
                If True, offloading and onloading is done with non-blocking data transfer.
            use_stream (`bool`, defaults to `False`):
                If True, offloading and onloading is done asynchronously using a CUDA stream. This can be useful for
                overlapping computation and data transfer.
            record_stream (`bool`, defaults to `False`): When enabled with `use_stream`, it marks the current tensor
                as having been used by this stream. It is faster at the expense of slightly more memory usage. Refer to
                the [PyTorch official docs](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
                more details.
            low_cpu_mem_usage (`bool`, defaults to `False`):
                If True, the CPU memory usage is minimized by pinning tensors on-the-fly instead of pre-pinning them.
                This option only matters when using streamed CPU offloading (i.e. `use_stream=True`). This can be
                useful when the CPU memory is a bottleneck but may counteract the benefits of using streams.
            exclude_modules (`Union[str, List[str]]`, defaults to `None`): List of modules to exclude from offloading.

        Example:
            ```python
            >>> from diffusers import DiffusionPipeline
            >>> import torch

            >>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)

            >>> pipe.enable_group_offload(
            ...     onload_device=torch.device("cuda"),
            ...     offload_device=torch.device("cpu"),
            ...     offload_type="leaf_level",
            ...     use_stream=True,
            ... )
            >>> image = pipe("a beautiful sunset").images[0]
            ```
        """
        from ..hooks import apply_group_offloading

        if isinstance(exclude_modules, str):
            exclude_modules = [exclude_modules]
        elif exclude_modules is None:
            exclude_modules = []

        unknown = set(exclude_modules) - self.components.keys()
        if unknown:
            logger.info(
                f"The following modules are not present in pipeline: {', '.join(unknown)}. Ignore if this is expected."
            )

        group_offload_kwargs = {
            "onload_device": onload_device,
            "offload_device": offload_device,
            "offload_type": offload_type,
            "num_blocks_per_group": num_blocks_per_group,
            "non_blocking": non_blocking,
            "use_stream": use_stream,
            "record_stream": record_stream,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "offload_to_disk_path": offload_to_disk_path,
        }
        for name, component in self.components.items():
            if name not in exclude_modules and isinstance(component, torch.nn.Module):
                if hasattr(component, "enable_group_offload"):
                    component.enable_group_offload(**group_offload_kwargs)
                else:
                    apply_group_offloading(module=component, **group_offload_kwargs)

        if exclude_modules:
            for module_name in exclude_modules:
                module = getattr(self, module_name, None)
                if module is not None and isinstance(module, torch.nn.Module):
                    module.to(onload_device)
                    logger.debug(f"Placed `{module_name}` on {onload_device} device as it was in `exclude_modules`.")

    def reset_device_map(self):
        r"""
        Resets the device maps (if any) to None.
        """
        if self.hf_device_map is None:
            return
        else:
            self.remove_all_hooks()
            for name, component in self.components.items():
                if isinstance(component, torch.nn.Module):
                    component.to("cpu")
            self.hf_device_map = None

    @classmethod
    @validate_hf_hub_args
    def download(cls, pretrained_model_name, **kwargs) -> Union[str, os.PathLike]:
        r"""
        Download and cache a PyTorch diffusion pipeline from pretrained pipeline weights.

        Parameters:
            pretrained_model_name (`str` or `os.PathLike`, *optional*):
                A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                hosted on the Hub.
            custom_pipeline (`str`, *optional*):
                Can be either:

                    - A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained
                      pipeline hosted on the Hub. The repository must contain a file called `pipeline.py` that defines
                      the custom pipeline.

                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current `main` branch of GitHub.

                    - A path to a *directory* (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                > [!WARNING] > 🧪 This is an experimental feature and may change in the future.

                For more information on how to load and create custom pipelines, take a look at [How to contribute a
                community pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline).

            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            dduf_file(`str`, *optional*):
                Load weights from the specified DDUF file.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `False`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom pipelines and components defined on the Hub in their own files. This
                option should only be set to `True` for repositories you trust and in which you have read the code, as
                it will execute code present on the Hub on your local machine.

        Returns:
            `os.PathLike`:
                A path to the downloaded pipeline.

        > [!TIP] > To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in
        with `hf > auth login

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        dduf_file: Optional[Dict[str, DDUFEntry]] = kwargs.pop("dduf_file", None)

        if dduf_file:
            if custom_pipeline:
                raise NotImplementedError("Custom pipelines are not supported with DDUF at the moment.")
            if load_connected_pipeline:
                raise NotImplementedError("Connected pipelines are not supported with DDUF at the moment.")
            return _download_dduf_file(
                pretrained_model_name=pretrained_model_name,
                dduf_file=dduf_file,
                pipeline_class_name=cls.__name__,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )

        allow_pickle = True if (use_safetensors is None or use_safetensors is False) else False
        use_safetensors = use_safetensors if use_safetensors is not None else True

        allow_patterns = None
        ignore_patterns = None

        model_info_call_error: Optional[Exception] = None
        if not local_files_only:
            try:
                info = model_info(pretrained_model_name, token=token, revision=revision)
            except (HfHubHTTPError, OfflineModeIsEnabled, requests.ConnectionError, httpx.NetworkError) as e:
                logger.warning(f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache.")
                local_files_only = True
                model_info_call_error = e  # save error to reraise it if model is not cached locally

        if not local_files_only:
            config_file = hf_hub_download(
                pretrained_model_name,
                cls.config_name,
                cache_dir=cache_dir,
                revision=revision,
                proxies=proxies,
                force_download=force_download,
                token=token,
            )
            config_dict = cls._dict_from_json_file(config_file)
            ignore_filenames = config_dict.pop("_ignore_files", [])

            filenames = {sibling.rfilename for sibling in info.siblings}
            if variant is not None and _check_legacy_sharding_variant_format(filenames=filenames, variant=variant):
                warn_msg = (
                    f"Warning: The repository contains sharded checkpoints for variant '{variant}' maybe in a deprecated format. "
                    "Please check your files carefully:\n\n"
                    "- Correct format example: diffusion_pytorch_model.fp16-00003-of-00003.safetensors\n"
                    "- Deprecated format example: diffusion_pytorch_model-00001-of-00002.fp16.safetensors\n\n"
                    "If you find any files in the deprecated format:\n"
                    "1. Remove all existing checkpoint files for this variant.\n"
                    "2. Re-obtain the correct files by running `save_pretrained()`.\n\n"
                    "This will ensure you're using the most up-to-date and compatible checkpoint format."
                )
                logger.warning(warn_msg)

            filenames = set(filenames) - set(ignore_filenames)
            if revision in DEPRECATED_REVISION_ARGS and version.parse(
                version.parse(__version__).base_version
            ) >= version.parse("0.22.0"):
                warn_deprecated_model_variant(pretrained_model_name, token, variant, revision, filenames)

            custom_components, folder_names = _get_custom_components_and_folders(
                pretrained_model_name, config_dict, filenames, variant
            )
            custom_class_name = None
            if custom_pipeline is None and isinstance(config_dict["_class_name"], (list, tuple)):
                custom_pipeline = config_dict["_class_name"][0]
                custom_class_name = config_dict["_class_name"][1]

            load_pipe_from_hub = custom_pipeline is not None and f"{custom_pipeline}.py" in filenames
            load_components_from_hub = len(custom_components) > 0

            if load_pipe_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {custom_pipeline}.py which must be executed to correctly "
                    f"load the model. You can inspect the repository content at https://hf.co/{pretrained_model_name}/blob/main/{custom_pipeline}.py.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            if load_components_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {'.py, '.join([os.path.join(k, v) for k, v in custom_components.items()])} which must be executed to correctly "
                    f"load the model. You can inspect the repository content at {', '.join([f'https://hf.co/{pretrained_model_name}/{k}/{v}.py' for k, v in custom_components.items()])}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            # retrieve passed components that should not be downloaded
            pipeline_class = _get_pipeline_class(
                cls,
                config_dict,
                load_connected_pipeline=load_connected_pipeline,
                custom_pipeline=custom_pipeline,
                repo_id=pretrained_model_name if load_pipe_from_hub else None,
                hub_revision=revision,
                class_name=custom_class_name,
                cache_dir=cache_dir,
                revision=custom_revision,
            )
            expected_components, _ = cls._get_signature_keys(pipeline_class)
            passed_components = [k for k in expected_components if k in kwargs]

            # retrieve the names of the folders containing model weights
            model_folder_names = {
                os.path.split(f)[0] for f in filter_model_files(filenames) if os.path.split(f)[0] in folder_names
            }
            # retrieve all patterns that should not be downloaded and error out when needed
            ignore_patterns = _get_ignore_patterns(
                passed_components,
                model_folder_names,
                filenames,
                use_safetensors,
                from_flax,
                allow_pickle,
                use_onnx,
                pipeline_class._is_onnx,
                variant,
            )

            model_filenames, variant_filenames = variant_compatible_siblings(
                filenames, variant=variant, ignore_patterns=ignore_patterns
            )

            # all filenames compatible with variant will be added
            allow_patterns = list(model_filenames)

            # allow all patterns from non-model folders
            # this enables downloading schedulers, tokenizers, ...
            allow_patterns += [f"{k}/*" for k in folder_names if k not in model_folder_names]
            # add custom component files
            allow_patterns += [f"{k}/{f}.py" for k, f in custom_components.items()]
            # add custom pipeline file
            allow_patterns += [f"{custom_pipeline}.py"] if f"{custom_pipeline}.py" in filenames else []
            # also allow downloading config.json files with the model
            allow_patterns += [os.path.join(k, "config.json") for k in model_folder_names]
            allow_patterns += [
                SCHEDULER_CONFIG_NAME,
                CONFIG_NAME,
                cls.config_name,
                CUSTOM_PIPELINE_FILE_NAME,
            ]

            # Don't download any objects that are passed
            allow_patterns = [
                p for p in allow_patterns if not (len(p.split("/")) == 2 and p.split("/")[0] in passed_components)
            ]

            if pipeline_class._load_connected_pipes:
                allow_patterns.append("README.md")

            # Don't download index files of forbidden patterns either
            ignore_patterns = ignore_patterns + [f"{i}.index.*json" for i in ignore_patterns]
            re_ignore_pattern = [re.compile(fnmatch.translate(p)) for p in ignore_patterns]
            re_allow_pattern = [re.compile(fnmatch.translate(p)) for p in allow_patterns]

            expected_files = [f for f in filenames if not any(p.match(f) for p in re_ignore_pattern)]
            expected_files = [f for f in expected_files if any(p.match(f) for p in re_allow_pattern)]

            snapshot_folder = Path(config_file).parent
            pipeline_is_cached = all((snapshot_folder / f).is_file() for f in expected_files)

            if pipeline_is_cached and not force_download:
                # if the pipeline is cached, we can directly return it
                # else call snapshot_download
                return snapshot_folder

        user_agent = {"pipeline_class": cls.__name__}
        if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
            user_agent["custom_pipeline"] = custom_pipeline

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = snapshot_download(
                pretrained_model_name,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )

            cls_name = cls.load_config(os.path.join(cached_folder, "model_index.json")).get("_class_name", None)
            cls_name = cls_name[4:] if isinstance(cls_name, str) and cls_name.startswith("Flax") else cls_name

            diffusers_module = importlib.import_module(__name__.split(".")[0])
            pipeline_class = getattr(diffusers_module, cls_name, None) if isinstance(cls_name, str) else None

            if pipeline_class is not None and pipeline_class._load_connected_pipes:
                modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
                connected_pipes = sum([getattr(modelcard.data, k, []) for k in CONNECTED_PIPES_KEYS], [])
                for connected_pipe_repo_id in connected_pipes:
                    download_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "local_files_only": local_files_only,
                        "token": token,
                        "variant": variant,
                        "use_safetensors": use_safetensors,
                    }
                    DiffusionPipeline.download(connected_pipe_repo_id, **download_kwargs)

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise EnvironmentError(
                    f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        optional_names = list(optional_parameters)
        for name in optional_names:
            if name in cls._optional_components:
                expected_modules.add(name)
                optional_parameters.remove(name)

        return sorted(expected_modules), sorted(optional_parameters)

    @classmethod
    def _get_signature_types(cls):
        signature_types = {}
        for k, v in inspect.signature(cls.__init__).parameters.items():
            if inspect.isclass(v.annotation):
                signature_types[k] = (v.annotation,)
            elif get_origin(v.annotation) == Union:
                signature_types[k] = get_args(v.annotation)
            elif get_origin(v.annotation) in [List, Dict, list, dict]:
                signature_types[k] = (v.annotation,)
            else:
                logger.warning(f"cannot get type annotation for Parameter {k} of {cls}.")
        return signature_types

    @property
    def parameters(self) -> Dict[str, Any]:
        r"""
        The `self.parameters` property can be useful to run different pipelines with the same weights and
        configurations without reallocating additional memory.

        Returns (`dict`):
            A dictionary containing all the optional parameters needed to initialize the pipeline.

        Examples:

        ```py
        >>> from diffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components, **text2img.parameters)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components, **text2img.parameters)
        ```
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        pipeline_parameters = {
            k: self.config[k] for k in self.config.keys() if not k.startswith("_") and k in optional_parameters
        }

        return pipeline_parameters

    @property
    def components(self) -> Dict[str, Any]:
        r"""
        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations without reallocating additional memory.

        Returns (`dict`):
            A dictionary containing all the modules needed to initialize the pipeline.

        Examples:

        ```py
        >>> from diffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        ```
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        actual = sorted(set(components.keys()))
        expected = sorted(expected_modules)
        if actual != expected:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected} to be defined, but {actual} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        return numpy_to_pil(images)

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/). When this
        option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
        up during training is not guaranteed.

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
        >>> from diffusers import DiffusionPipeline
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        >>> # Workaround for not accepting attention shape using VAE for Flash Attention
        >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
        """
        self.set_use_memory_efficient_attention_xformers(False)

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

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            fn_recursive_set_mem_eff(module)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
        in slices to compute attention in several steps. For more than one attention head, the computation is performed
        sequentially over each head. This is useful to save some memory in exchange for a small speed decrease.

        > [!WARNING] > ⚠️ Don't enable attention slicing if you're already using `scaled_dot_product_attention` (SDPA)
        from PyTorch > 2.0 or xFormers. These attention computations are already very memory efficient so you won't
        need to enable > this function. If you enable attention slicing with SDPA or xFormers, it can lead to serious
        slow downs!

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ...     torch_dtype=torch.float16,
        ...     use_safetensors=True,
        ... )

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> pipe.enable_attention_slicing()
        >>> image = pipe(prompt).images[0]
        ```
        """
        self.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously called, attention is
        computed in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def set_attention_slice(self, slice_size: Optional[int]):
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module) and hasattr(m, "set_attention_slice")]

        for module in modules:
            module.set_attention_slice(slice_size)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Create a new pipeline from a given pipeline. This method is useful to create a new pipeline from the existing
        pipeline components without reallocating additional memory.

        Arguments:
            pipeline (`DiffusionPipeline`):
                The pipeline from which to create a new pipeline.

        Returns:
            `DiffusionPipeline`:
                A new pipeline with the same weights and configurations as `pipeline`.

        Examples:

        ```py
        >>> from diffusers import StableDiffusionPipeline, StableDiffusionSAGPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        >>> new_pipe = StableDiffusionSAGPipeline.from_pipe(pipe)
        ```
        """

        original_config = dict(pipeline.config)
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)

        # derive the pipeline class to instantiate
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)

        if custom_pipeline is not None:
            pipeline_class = _get_custom_pipeline_class(custom_pipeline, revision=custom_revision)
        else:
            pipeline_class = cls

        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        # true_optional_modules are optional components with default value in signature so it is ok not to pass them to `__init__`
        # e.g. `image_encoder` for StableDiffusionPipeline
        parameters = inspect.signature(cls.__init__).parameters
        true_optional_modules = set(
            {k for k, v in parameters.items() if v.default != inspect._empty and k in expected_modules}
        )

        # get the class of each component based on its type hint
        # e.g. {"unet": UNet2DConditionModel, "text_encoder": CLIPTextMode}
        component_types = pipeline_class._get_signature_types()

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)
        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}

        original_class_obj = {}
        for name, component in pipeline.components.items():
            if name in expected_modules and name not in passed_class_obj:
                # for model components, we will not switch over if the class does not matches the type hint in the new pipeline's signature
                if (
                    not isinstance(component, ModelMixin)
                    or type(component) in component_types[name]
                    or (component is None and name in cls._optional_components)
                ):
                    original_class_obj[name] = component
                else:
                    logger.warning(
                        f"component {name} is not switched over to new pipeline because type does not match the expected."
                        f" {name} is {type(component)} while the new pipeline expect {component_types[name]}."
                        f" please pass the component of the correct type to the new pipeline. `from_pipe(..., {name}={name})`"
                    )

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k in original_config.keys()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config attribute that were not expected by pipeline is stored as its private attribute
        # (i.e. when the original pipeline was also instantiated with `from_pipe` from another pipeline that has this config)
        # in this case, we will pass them as optional arguments if they can be accepted by the new pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        pipeline_kwargs = {
            **passed_class_obj,
            **original_class_obj,
            **passed_pipe_kwargs,
            **original_pipe_kwargs,
            **kwargs,
        }

        # store unused config as private attribute in the new pipeline
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": v for k, v in original_config.items() if k not in pipeline_kwargs
        }

        optional_components = (
            pipeline._optional_components
            if hasattr(pipeline, "_optional_components") and pipeline._optional_components
            else []
        )
        missing_modules = (
            set(expected_modules) - set(optional_components) - set(pipeline_kwargs.keys()) - set(true_optional_modules)
        )

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        new_pipeline = pipeline_class(**pipeline_kwargs)
        if pretrained_model_name_or_path is not None:
            new_pipeline.register_to_config(_name_or_path=pretrained_model_name_or_path)
        new_pipeline.register_to_config(**unused_original_config)

        if torch_dtype is not None:
            new_pipeline.to(dtype=torch_dtype)

        return new_pipeline

    def _maybe_raise_error_if_group_offload_active(
        self, raise_error: bool = False, module: Optional[torch.nn.Module] = None
    ) -> bool:
        from ..hooks.group_offloading import _is_group_offload_enabled

        components = self.components.values() if module is None else [module]
        components = [component for component in components if isinstance(component, torch.nn.Module)]
        for component in components:
            if _is_group_offload_enabled(component):
                if raise_error:
                    raise ValueError(
                        "You are trying to apply model/sequential CPU offloading to a pipeline that contains components "
                        "with group offloading enabled. This is not supported. Please disable group offloading for "
                        "components of the pipeline to use other offloading methods."
                    )
                return True
        return False


class StableDiffusionMixin:
    r"""
    Helper for DiffusionPipeline with vae and unet.(mainly for LDM such as stable diffusion)
    """

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://huggingface.co/papers/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    def fuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        > [!WARNING] > This API is 🧪 experimental.

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_unet = False
        self.fusing_vae = False

        if unet:
            self.fusing_unet = True
            self.unet.fuse_qkv_projections()
            self.unet.set_attn_processor(FusedAttnProcessor2_0())

        if vae:
            if not isinstance(self.vae, AutoencoderKL):
                raise ValueError("`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.")

            self.fusing_vae = True
            self.vae.fuse_qkv_projections()
            self.vae.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        > [!WARNING] > This API is 🧪 experimental.

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if unet:
            if not self.fusing_unet:
                logger.warning("The UNet was not initially fused for QKV projections. Doing nothing.")
            else:
                self.unet.unfuse_qkv_projections()
                self.fusing_unet = False

        if vae:
            if not self.fusing_vae:
                logger.warning("The VAE was not initially fused for QKV projections. Doing nothing.")
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False
