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
import inspect
import os
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch
import torch.nn.functional as F
from huggingface_hub.utils import validate_hf_hub_args
from torch import nn

from ..models.embeddings import (
    ImageProjection,
    IPAdapterFaceIDImageProjection,
    IPAdapterFaceIDPlusImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)
from ..models.modeling_utils import load_model_dict_into_meta, load_state_dict
from ..utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    convert_unet_state_dict_to_peft,
    delete_adapter_layers,
    get_adapter_name,
    get_peft_kwargs,
    is_accelerate_available,
    is_peft_version,
    is_torch_version,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from .lora import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE, TEXT_ENCODER_NAME, UNET_NAME
from .unet_loader_utils import _maybe_expand_lora_scales
from .utils import AttnProcsLayers


if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)


CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


class UNet2DConditionLoadersMixin:
    """
    Load LoRA layers into a [`UNet2DCondtionModel`].
    """

    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    @validate_hf_hub_args
    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
        and be a `torch.nn.Module` class. Currently supported: LoRA, Custom Diffusion. For LoRA, one must install
        `peft`: `pip install -U peft`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible. Will be removed in v1
                of Diffusers.
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
            adapter_name (`str`, *optional*, defaults to None):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            weight_name (`str`, *optional*, defaults to None):
                Name of the serialized state dict file.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.unet.load_attn_procs(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        adapter_name = kwargs.pop("adapter_name", None)
        _pipeline = kwargs.pop("_pipeline", None)
        network_alphas = kwargs.pop("network_alphas", None)
        allow_pickle = False

        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = load_state_dict(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())
        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        if is_custom_diffusion:
            attn_processors = self._process_custom_diffusion(state_dict=state_dict)
        elif is_lora:
            is_model_cpu_offload, is_sequential_cpu_offload = self._process_lora(
                state_dict=state_dict,
                unet_identifier_key=self.unet_name,
                network_alphas=network_alphas,
                adapter_name=adapter_name,
                _pipeline=_pipeline,
            )
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by Custom Diffusion training."
            )

        # <Unsafe code
        # We can be sure that the following works as it just sets attention processors, lora layers and puts all in the same dtype
        # Now we remove any existing hooks to `_pipeline`.

        # For LoRA, the UNet is already offloaded at this stage as it is handled inside `_process_lora`.
        if is_custom_diffusion and _pipeline is not None:
            is_model_cpu_offload, is_sequential_cpu_offload = self._optionally_disable_offloading(_pipeline=_pipeline)

            # only custom diffusion needs to set attn processors
            self.set_attn_processor(attn_processors)
            self.to(dtype=self.dtype, device=self.device)

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />

    def _process_custom_diffusion(self, state_dict):
        from ..models.attention_processor import CustomDiffusionAttnProcessor

        attn_processors = {}
        custom_diffusion_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            if len(value) == 0:
                custom_diffusion_grouped_dict[key] = {}
            else:
                if "to_out" in key:
                    attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                else:
                    attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

        for key, value_dict in custom_diffusion_grouped_dict.items():
            if len(value_dict) == 0:
                attn_processors[key] = CustomDiffusionAttnProcessor(
                    train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                )
            else:
                cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                attn_processors[key] = CustomDiffusionAttnProcessor(
                    train_kv=True,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
                attn_processors[key].load_state_dict(value_dict)

        return attn_processors

    def _process_lora(self, state_dict, unet_identifier_key, network_alphas, adapter_name, _pipeline):
        # This method does the following things:
        # 1. Filters the `state_dict` with keys matching  `unet_identifier_key` when using the non-legacy
        #    format. For legacy format no filtering is applied.
        # 2. Converts the `state_dict` to the `peft` compatible format.
        # 3. Creates a `LoraConfig` and then injects the converted `state_dict` into the UNet per the
        #    `LoraConfig` specs.
        # 4. It also reports if the underlying `_pipeline` has any kind of offloading inside of it.
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        keys = list(state_dict.keys())

        unet_keys = [k for k in keys if k.startswith(unet_identifier_key)]
        unet_state_dict = {
            k.replace(f"{unet_identifier_key}.", ""): v for k, v in state_dict.items() if k in unet_keys
        }

        if network_alphas is not None:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(unet_identifier_key)]
            network_alphas = {
                k.replace(f"{unet_identifier_key}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
            }

        is_model_cpu_offload = False
        is_sequential_cpu_offload = False
        state_dict_to_be_used = unet_state_dict if len(unet_state_dict) > 0 else state_dict

        if len(state_dict_to_be_used) > 0:
            if adapter_name in getattr(self, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the Unet - please select a new adapter name."
                )

            state_dict = convert_unet_state_dict_to_peft(state_dict_to_be_used)

            if network_alphas is not None:
                # The alphas state dict have the same structure as Unet, thus we convert it to peft format using
                # `convert_unet_state_dict_to_peft` method.
                network_alphas = convert_unet_state_dict_to_peft(network_alphas)

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict, is_unet=True)
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"]:
                    if is_peft_version("<", "0.9.0"):
                        raise ValueError(
                            "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                        )
                else:
                    if is_peft_version("<", "0.9.0"):
                        lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(self)

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error
            is_model_cpu_offload, is_sequential_cpu_offload = self._optionally_disable_offloading(_pipeline)

            inject_adapter_in_model(lora_config, self, adapter_name=adapter_name)
            incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name)

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        return is_model_cpu_offload, is_sequential_cpu_offload

    @classmethod
    # Copied from diffusers.loaders.lora.LoraLoaderMixin._optionally_disable_offloading
    def _optionally_disable_offloading(cls, _pipeline):
        """
        Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

        Args:
            _pipeline (`DiffusionPipeline`):
                The pipeline to disable offloading for.

        Returns:
            tuple:
                A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
        """
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        if _pipeline is not None and _pipeline.hf_device_map is None:
            for _, component in _pipeline.components.items():
                if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                    if not is_model_cpu_offload:
                        is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                    if not is_sequential_cpu_offload:
                        is_sequential_cpu_offload = (
                            isinstance(component._hf_hook, AlignDevicesHook)
                            or hasattr(component._hf_hook, "hooks")
                            and isinstance(component._hf_hook.hooks[0], AlignDevicesHook)
                        )

                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

        return (is_model_cpu_offload, is_sequential_cpu_offload)

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        r"""
        Save attention processor layers to a directory so that it can be reloaded with the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to (will be created if it doesn't exist).
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or with `pickle`.

        Example:

        ```py
        import torch
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        ).to("cuda")
        pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        pipeline.unet.save_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        ```
        """
        from ..models.attention_processor import (
            CustomDiffusionAttnProcessor,
            CustomDiffusionAttnProcessor2_0,
            CustomDiffusionXFormersAttnProcessor,
        )

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        is_custom_diffusion = any(
            isinstance(
                x,
                (CustomDiffusionAttnProcessor, CustomDiffusionAttnProcessor2_0, CustomDiffusionXFormersAttnProcessor),
            )
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            state_dict = self._get_custom_diffusion_state_dict()
            if save_function is None and safe_serialization:
                # safetensors does not support saving dicts with non-tensor values
                empty_state_dict = {k: v for k, v in state_dict.items() if not isinstance(v, torch.Tensor)}
                if len(empty_state_dict) > 0:
                    logger.warning(
                        f"Safetensors does not support saving dicts with non-tensor values. "
                        f"The following keys will be ignored: {empty_state_dict.keys()}"
                    )
                state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        else:
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for saving LoRAs using the `save_attn_procs()` method.")

            from peft.utils import get_peft_model_state_dict

            state_dict = get_peft_model_state_dict(self)

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if safe_serialization:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else LORA_WEIGHT_NAME

        # Save the model
        save_path = Path(save_directory, weight_name).as_posix()
        save_function(state_dict, save_path)
        logger.info(f"Model weights saved in {save_path}")

    def _get_custom_diffusion_state_dict(self):
        from ..models.attention_processor import (
            CustomDiffusionAttnProcessor,
            CustomDiffusionAttnProcessor2_0,
            CustomDiffusionXFormersAttnProcessor,
        )

        model_to_save = AttnProcsLayers(
            {
                y: x
                for (y, x) in self.attn_processors.items()
                if isinstance(
                    x,
                    (
                        CustomDiffusionAttnProcessor,
                        CustomDiffusionAttnProcessor2_0,
                        CustomDiffusionXFormersAttnProcessor,
                    ),
                )
            }
        )
        state_dict = model_to_save.state_dict()
        for name, attn in self.attn_processors.items():
            if len(attn.state_dict()) == 0:
                state_dict[name] = {}

        return state_dict

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

            # For BC with prevous PEFT versions, we need to check the signature
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

        from ..utils import recurse_remove_peft_layers

        recurse_remove_peft_layers(self)
        if hasattr(self, "peft_config"):
            del self.peft_config

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        """
        Set the currently active adapters for use in the UNet.

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
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
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
        weights = _maybe_expand_lora_scales(self, weights)

        set_weights_and_activate_adapters(self, adapter_names, weights)

    def disable_lora(self):
        """
        Disable the UNet's active LoRA layers.

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
        pipeline.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=False)

    def enable_lora(self):
        """
        Enable the UNet's active LoRA layers.

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
        pipeline.enable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=True)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the UNet.

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
        pipeline.delete_adapters("cinematic")
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

    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict, low_cpu_mem_usage=False):
        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        updated_state_dict = {}
        image_projection = None
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

            with init_context():
                image_projection = ImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=clip_embeddings_dim,
                    num_image_text_embeds=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        elif "proj.3.weight" in state_dict:
            # IP-Adapter Full
            clip_embeddings_dim = state_dict["proj.0.weight"].shape[0]
            cross_attention_dim = state_dict["proj.3.weight"].shape[0]

            with init_context():
                image_projection = IPAdapterFullImageProjection(
                    cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                diffusers_name = diffusers_name.replace("proj.3", "norm")
                updated_state_dict[diffusers_name] = value

        elif "perceiver_resampler.proj_in.weight" in state_dict:
            # IP-Adapter Face ID Plus
            id_embeddings_dim = state_dict["proj.0.weight"].shape[1]
            embed_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[0]
            hidden_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[1]
            output_dims = state_dict["perceiver_resampler.proj_out.weight"].shape[0]
            heads = state_dict["perceiver_resampler.layers.0.0.to_q.weight"].shape[0] // 64

            with init_context():
                image_projection = IPAdapterFaceIDPlusImageProjection(
                    embed_dims=embed_dims,
                    output_dims=output_dims,
                    hidden_dims=hidden_dims,
                    heads=heads,
                    id_embeddings_dim=id_embeddings_dim,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("perceiver_resampler.", "")
                diffusers_name = diffusers_name.replace("0.to", "attn.to")
                diffusers_name = diffusers_name.replace("0.1.0.", "0.ff.0.")
                diffusers_name = diffusers_name.replace("0.1.1.weight", "0.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("0.1.3.weight", "0.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("1.1.0.", "1.ff.0.")
                diffusers_name = diffusers_name.replace("1.1.1.weight", "1.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.1.3.weight", "1.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("2.1.0.", "2.ff.0.")
                diffusers_name = diffusers_name.replace("2.1.1.weight", "2.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("2.1.3.weight", "2.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("3.1.0.", "3.ff.0.")
                diffusers_name = diffusers_name.replace("3.1.1.weight", "3.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("3.1.3.weight", "3.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("layers.0.0", "layers.0.ln0")
                diffusers_name = diffusers_name.replace("layers.0.1", "layers.0.ln1")
                diffusers_name = diffusers_name.replace("layers.1.0", "layers.1.ln0")
                diffusers_name = diffusers_name.replace("layers.1.1", "layers.1.ln1")
                diffusers_name = diffusers_name.replace("layers.2.0", "layers.2.ln0")
                diffusers_name = diffusers_name.replace("layers.2.1", "layers.2.ln1")
                diffusers_name = diffusers_name.replace("layers.3.0", "layers.3.ln0")
                diffusers_name = diffusers_name.replace("layers.3.1", "layers.3.ln1")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                elif "proj.0.weight" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.weight"] = value
                elif "proj.0.bias" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.bias"] = value
                elif "proj.2.weight" == diffusers_name:
                    updated_state_dict["proj.net.2.weight"] = value
                elif "proj.2.bias" == diffusers_name:
                    updated_state_dict["proj.net.2.bias"] = value
                else:
                    updated_state_dict[diffusers_name] = value

        elif "norm.weight" in state_dict:
            # IP-Adapter Face ID
            id_embeddings_dim_in = state_dict["proj.0.weight"].shape[1]
            id_embeddings_dim_out = state_dict["proj.0.weight"].shape[0]
            multiplier = id_embeddings_dim_out // id_embeddings_dim_in
            norm_layer = "norm.weight"
            cross_attention_dim = state_dict[norm_layer].shape[0]
            num_tokens = state_dict["proj.2.weight"].shape[0] // cross_attention_dim

            with init_context():
                image_projection = IPAdapterFaceIDImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=id_embeddings_dim_in,
                    mult=multiplier,
                    num_tokens=num_tokens,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                updated_state_dict[diffusers_name] = value

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            attn_key_present = any("attn" in k for k in state_dict)
            heads = (
                state_dict["layers.0.attn.to_q.weight"].shape[0] // 64
                if attn_key_present
                else state_dict["layers.0.0.to_q.weight"].shape[0] // 64
            )

            with init_context():
                image_projection = IPAdapterPlusImageProjection(
                    embed_dims=embed_dims,
                    output_dims=output_dims,
                    hidden_dims=hidden_dims,
                    heads=heads,
                    num_queries=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")

                diffusers_name = diffusers_name.replace("0.0.norm1", "0.ln0")
                diffusers_name = diffusers_name.replace("0.0.norm2", "0.ln1")
                diffusers_name = diffusers_name.replace("1.0.norm1", "1.ln0")
                diffusers_name = diffusers_name.replace("1.0.norm2", "1.ln1")
                diffusers_name = diffusers_name.replace("2.0.norm1", "2.ln0")
                diffusers_name = diffusers_name.replace("2.0.norm2", "2.ln1")
                diffusers_name = diffusers_name.replace("3.0.norm1", "3.ln0")
                diffusers_name = diffusers_name.replace("3.0.norm2", "3.ln1")

                if "to_kv" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_q" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    updated_state_dict[diffusers_name] = value
                elif "to_out" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    diffusers_name = diffusers_name.replace("0.1.0", "0.ff.0")
                    diffusers_name = diffusers_name.replace("0.1.1", "0.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("0.1.3", "0.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("1.1.0", "1.ff.0")
                    diffusers_name = diffusers_name.replace("1.1.1", "1.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("1.1.3", "1.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("2.1.0", "2.ff.0")
                    diffusers_name = diffusers_name.replace("2.1.1", "2.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("2.1.3", "2.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("3.1.0", "3.ff.0")
                    diffusers_name = diffusers_name.replace("3.1.1", "3.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("3.1.3", "3.ff.1.net.2")
                    updated_state_dict[diffusers_name] = value

        if not low_cpu_mem_usage:
            image_projection.load_state_dict(updated_state_dict, strict=True)
        else:
            load_model_dict_into_meta(image_projection, updated_state_dict, device=self.device, dtype=self.dtype)

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=False):
        from ..models.attention_processor import (
            IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0,
        )

        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
        for name in self.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = self.attn_processors[name].__class__
                attn_procs[name] = attn_processor_class()

            else:
                attn_processor_class = (
                    IPAdapterAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else IPAdapterAttnProcessor
                )
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    elif "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID Plus
                        num_image_text_embeds += [4]
                    elif "norm.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID
                        num_image_text_embeds += [4]
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                with init_context():
                    attn_procs[name] = attn_processor_class(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_image_text_embeds,
                    )

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                if not low_cpu_mem_usage:
                    attn_procs[name].load_state_dict(value_dict)
                else:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(attn_procs[name], value_dict, device=device, dtype=dtype)

                key_id += 2

        return attn_procs

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=False):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]
        # Set encoder_hid_proj after loading ip_adapter weights,
        # because `IPAdapterPlusImageProjection` also has `attn_processors`.
        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
        self.set_attn_processor(attn_procs)

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"

        self.to(dtype=self.dtype, device=self.device)

    def _load_ip_adapter_loras(self, state_dicts):
        lora_dicts = {}
        for key_id, name in enumerate(self.attn_processors.keys()):
            for i, state_dict in enumerate(state_dicts):
                if f"{key_id}.to_k_lora.down.weight" in state_dict["ip_adapter"]:
                    if i not in lora_dicts:
                        lora_dicts[i] = {}
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_k_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_k_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_q_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_q_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_v_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_v_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_k_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_k_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_q_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_q_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_v_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_v_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.up.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.up.weight"
                            ]
                        }
                    )
        return lora_dicts
