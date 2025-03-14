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

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from huggingface_hub.utils import validate_hf_hub_args
from safetensors import safe_open

from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_state_dict
from ..utils import (
    USE_PEFT_BACKEND,
    _get_detailed_type,
    _get_model_file,
    _is_valid_type,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)
from .unet_loader_utils import _maybe_expand_lora_scales


if is_transformers_available():
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor, SiglipVisionModel

from ..models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    FluxAttnProcessor2_0,
    FluxIPAdapterJointAttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
    IPAdapterXFormersAttnProcessor,
    JointAttnProcessor2_0,
    SD3IPAdapterJointAttnProcessor2_0,
)


logger = logging.get_logger(__name__)


class IPAdapterMixin:
    """Mixin for handling IP Adapters."""

    @validate_hf_hub_args
    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `List[str]` or `os.PathLike` or `List[os.PathLike]` or `dict` or `List[dict]`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
            subfolder (`str` or `List[str]`):
                The subfolder location of a model file within a larger model repository on the Hub or locally. If a
                list is passed, it should have the same length as `weight_name`.
            weight_name (`str` or `List[str]`):
                The name of the weight file to load. If a list is passed, it should have the same length as
                `subfolder`.
            image_encoder_folder (`str`, *optional*, defaults to `image_encoder`):
                The subfolder location of the image encoder within a larger model repository on the Hub or locally.
                Pass `None` to not load the image encoder. If the image encoder is located in a folder inside
                `subfolder`, you only need to pass the name of the folder that contains image encoder weights, e.g.
                `image_encoder_folder="image_encoder"`. If the image encoder is located in a folder other than
                `subfolder`, you should pass the path to the folder that contains image encoder weights, for example,
                `image_encoder_folder="different_subfolder/image_encoder"`.
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
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
        """

        # handle the list inputs for multiple IP Adapters
        if not isinstance(weight_name, list):
            weight_name = [weight_name]

        if not isinstance(pretrained_model_name_or_path_or_dict, list):
            pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
        if len(pretrained_model_name_or_path_or_dict) == 1:
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

        if not isinstance(subfolder, list):
            subfolder = [subfolder]
        if len(subfolder) == 1:
            subfolder = subfolder * len(weight_name)

        if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
            raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

        if len(weight_name) != len(subfolder):
            raise ValueError("`weight_name` and `subfolder` must have the same length.")

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        if low_cpu_mem_usage and not is_accelerate_available():
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

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        state_dicts = []
        for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
            pretrained_model_name_or_path_or_dict, weight_name, subfolder
        ):
            if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                if weight_name.endswith(".safetensors"):
                    state_dict = {"image_proj": {}, "ip_adapter": {}}
                    with safe_open(model_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith("image_proj."):
                                state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                            elif key.startswith("ip_adapter."):
                                state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
                else:
                    state_dict = load_state_dict(model_file)
            else:
                state_dict = pretrained_model_name_or_path_or_dict

            keys = list(state_dict.keys())
            if "image_proj" not in keys and "ip_adapter" not in keys:
                raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

            state_dicts.append(state_dict)

            # load CLIP image encoder here if it has not been registered to the pipeline yet
            if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
                if image_encoder_folder is not None:
                    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                        logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
                        if image_encoder_folder.count("/") == 0:
                            image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
                        else:
                            image_encoder_subfolder = Path(image_encoder_folder).as_posix()

                        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                            pretrained_model_name_or_path_or_dict,
                            subfolder=image_encoder_subfolder,
                            low_cpu_mem_usage=low_cpu_mem_usage,
                            cache_dir=cache_dir,
                            local_files_only=local_files_only,
                            torch_dtype=self.dtype,
                        ).to(self.device)
                        self.register_modules(image_encoder=image_encoder)
                    else:
                        raise ValueError(
                            "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                        )
                else:
                    logger.warning(
                        "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                        "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
                    )

            # create feature extractor if it has not been registered to the pipeline yet
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
                # FaceID IP adapters don't need the image encoder so it's not present, in this case we default to 224
                default_clip_size = 224
                clip_image_size = (
                    self.image_encoder.config.image_size if self.image_encoder is not None else default_clip_size
                )
                feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
                self.register_modules(feature_extractor=feature_extractor)

        # load ip-adapter into unet
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)

        extra_loras = unet._load_ip_adapter_loras(state_dicts)
        if extra_loras != {}:
            if not USE_PEFT_BACKEND:
                logger.warning("PEFT backend is required to load these weights.")
            else:
                # apply the IP Adapter Face ID LoRA weights
                peft_config = getattr(unet, "peft_config", {})
                for k, lora in extra_loras.items():
                    if f"faceid_{k}" not in peft_config:
                        self.load_lora_weights(lora, adapter_name=f"faceid_{k}")
                        self.set_adapters([f"faceid_{k}"], adapter_weights=[1.0])

    def set_ip_adapter_scale(self, scale):
        """
        Set IP-Adapter scales per-transformer block. Input `scale` could be a single config or a list of configs for
        granular control over each IP-Adapter behavior. A config can be a float or a dictionary.

        Example:

        ```py
        # To use original IP-Adapter
        scale = 1.0
        pipeline.set_ip_adapter_scale(scale)

        # To use style block only
        scale = {
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)

        # To use style+layout blocks
        scale = {
            "down": {"block_2": [0.0, 1.0]},
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)

        # To use style and layout from 2 reference images
        scales = [{"down": {"block_2": [0.0, 1.0]}}, {"up": {"block_0": [0.0, 1.0, 0.0]}}]
        pipeline.set_ip_adapter_scale(scales)
        ```
        """
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        if not isinstance(scale, list):
            scale = [scale]
        scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)

        for attn_name, attn_processor in unet.attn_processors.items():
            if isinstance(
                attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor)
            ):
                if len(scale_configs) != len(attn_processor.scale):
                    raise ValueError(
                        f"Cannot assign {len(scale_configs)} scale_configs to "
                        f"{len(attn_processor.scale)} IP-Adapter."
                    )
                elif len(scale_configs) == 1:
                    scale_configs = scale_configs * len(attn_processor.scale)
                for i, scale_config in enumerate(scale_configs):
                    if isinstance(scale_config, dict):
                        for k, s in scale_config.items():
                            if attn_name.startswith(k):
                                attn_processor.scale[i] = s
                    else:
                        attn_processor.scale[i] = scale_config

    def unload_ip_adapter(self):
        """
        Unloads the IP Adapter weights

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the IP Adapter weights.
        >>> pipeline.unload_ip_adapter()
        >>> ...
        ```
        """
        # remove CLIP image encoder
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is not None:
            self.image_encoder = None
            self.register_to_config(image_encoder=[None, None])

        # remove feature extractor only when safety_checker is None as safety_checker uses
        # the feature_extractor later
        if not hasattr(self, "safety_checker"):
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is not None:
                self.feature_extractor = None
                self.register_to_config(feature_extractor=[None, None])

        # remove hidden encoder
        self.unet.encoder_hid_proj = None
        self.unet.config.encoder_hid_dim_type = None

        # Kolors: restore `encoder_hid_proj` with `text_encoder_hid_proj`
        if hasattr(self.unet, "text_encoder_hid_proj") and self.unet.text_encoder_hid_proj is not None:
            self.unet.encoder_hid_proj = self.unet.text_encoder_hid_proj
            self.unet.text_encoder_hid_proj = None
            self.unet.config.encoder_hid_dim_type = "text_proj"

        # restore original Unet attention processors layers
        attn_procs = {}
        for name, value in self.unet.attn_processors.items():
            attn_processor_class = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnProcessor()
            )
            attn_procs[name] = (
                attn_processor_class
                if isinstance(
                    value, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor)
                )
                else value.__class__()
            )
        self.unet.set_attn_processor(attn_procs)


class FluxIPAdapterMixin:
    """Mixin for handling Flux IP Adapters."""

    @validate_hf_hub_args
    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        weight_name: Union[str, List[str]],
        subfolder: Optional[Union[str, List[str]]] = "",
        image_encoder_pretrained_model_name_or_path: Optional[str] = "image_encoder",
        image_encoder_subfolder: Optional[str] = "",
        image_encoder_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `List[str]` or `os.PathLike` or `List[os.PathLike]` or `dict` or `List[dict]`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
            subfolder (`str` or `List[str]`):
                The subfolder location of a model file within a larger model repository on the Hub or locally. If a
                list is passed, it should have the same length as `weight_name`.
            weight_name (`str` or `List[str]`):
                The name of the weight file to load. If a list is passed, it should have the same length as
                `weight_name`.
            image_encoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `./image_encoder`):
                Can be either:

                    - A string, the *model id* (for example `openai/clip-vit-large-patch14`) of a pretrained model
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
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
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
        """

        # handle the list inputs for multiple IP Adapters
        if not isinstance(weight_name, list):
            weight_name = [weight_name]

        if not isinstance(pretrained_model_name_or_path_or_dict, list):
            pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
        if len(pretrained_model_name_or_path_or_dict) == 1:
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

        if not isinstance(subfolder, list):
            subfolder = [subfolder]
        if len(subfolder) == 1:
            subfolder = subfolder * len(weight_name)

        if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
            raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

        if len(weight_name) != len(subfolder):
            raise ValueError("`weight_name` and `subfolder` must have the same length.")

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        if low_cpu_mem_usage and not is_accelerate_available():
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

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        state_dicts = []
        for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
            pretrained_model_name_or_path_or_dict, weight_name, subfolder
        ):
            if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                if weight_name.endswith(".safetensors"):
                    state_dict = {"image_proj": {}, "ip_adapter": {}}
                    with safe_open(model_file, framework="pt", device="cpu") as f:
                        image_proj_keys = ["ip_adapter_proj_model.", "image_proj."]
                        ip_adapter_keys = ["double_blocks.", "ip_adapter."]
                        for key in f.keys():
                            if any(key.startswith(prefix) for prefix in image_proj_keys):
                                diffusers_name = ".".join(key.split(".")[1:])
                                state_dict["image_proj"][diffusers_name] = f.get_tensor(key)
                            elif any(key.startswith(prefix) for prefix in ip_adapter_keys):
                                diffusers_name = (
                                    ".".join(key.split(".")[1:])
                                    .replace("ip_adapter_double_stream_k_proj", "to_k_ip")
                                    .replace("ip_adapter_double_stream_v_proj", "to_v_ip")
                                    .replace("processor.", "")
                                )
                                state_dict["ip_adapter"][diffusers_name] = f.get_tensor(key)
                else:
                    state_dict = load_state_dict(model_file)
            else:
                state_dict = pretrained_model_name_or_path_or_dict

            keys = list(state_dict.keys())
            if keys != ["image_proj", "ip_adapter"]:
                raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

            state_dicts.append(state_dict)

            # load CLIP image encoder here if it has not been registered to the pipeline yet
            if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
                if image_encoder_pretrained_model_name_or_path is not None:
                    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                        logger.info(f"loading image_encoder from {image_encoder_pretrained_model_name_or_path}")
                        image_encoder = (
                            CLIPVisionModelWithProjection.from_pretrained(
                                image_encoder_pretrained_model_name_or_path,
                                subfolder=image_encoder_subfolder,
                                low_cpu_mem_usage=low_cpu_mem_usage,
                                cache_dir=cache_dir,
                                local_files_only=local_files_only,
                                dtype=image_encoder_dtype,
                            )
                            .to(self.device)
                            .eval()
                        )
                        self.register_modules(image_encoder=image_encoder)
                    else:
                        raise ValueError(
                            "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                        )
                else:
                    logger.warning(
                        "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                        "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
                    )

            # create feature extractor if it has not been registered to the pipeline yet
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
                # FaceID IP adapters don't need the image encoder so it's not present, in this case we default to 224
                default_clip_size = 224
                clip_image_size = (
                    self.image_encoder.config.image_size if self.image_encoder is not None else default_clip_size
                )
                feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
                self.register_modules(feature_extractor=feature_extractor)

        # load ip-adapter into transformer
        self.transformer._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)

    def set_ip_adapter_scale(self, scale: Union[float, List[float], List[List[float]]]):
        """
        Set IP-Adapter scales per-transformer block. Input `scale` could be a single config or a list of configs for
        granular control over each IP-Adapter behavior. A config can be a float or a list.

        `float` is converted to list and repeated for the number of blocks and the number of IP adapters. `List[float]`
        length match the number of blocks, it is repeated for each IP adapter. `List[List[float]]` must match the
        number of IP adapters and each must match the number of blocks.

        Example:

        ```py
        # To use original IP-Adapter
        scale = 1.0
        pipeline.set_ip_adapter_scale(scale)


        def LinearStrengthModel(start, finish, size):
            return [(start + (finish - start) * (i / (size - 1))) for i in range(size)]


        ip_strengths = LinearStrengthModel(0.3, 0.92, 19)
        pipeline.set_ip_adapter_scale(ip_strengths)
        ```
        """

        scale_type = Union[int, float]
        num_ip_adapters = self.transformer.encoder_hid_proj.num_ip_adapters
        num_layers = self.transformer.config.num_layers

        # Single value for all layers of all IP-Adapters
        if isinstance(scale, scale_type):
            scale = [scale for _ in range(num_ip_adapters)]
        # List of per-layer scales for a single IP-Adapter
        elif _is_valid_type(scale, List[scale_type]) and num_ip_adapters == 1:
            scale = [scale]
        # Invalid scale type
        elif not _is_valid_type(scale, List[Union[scale_type, List[scale_type]]]):
            raise TypeError(f"Unexpected type {_get_detailed_type(scale)} for scale.")

        if len(scale) != num_ip_adapters:
            raise ValueError(f"Cannot assign {len(scale)} scales to {num_ip_adapters} IP-Adapters.")

        if any(len(s) != num_layers for s in scale if isinstance(s, list)):
            invalid_scale_sizes = {len(s) for s in scale if isinstance(s, list)} - {num_layers}
            raise ValueError(
                f"Expected list of {num_layers} scales, got {', '.join(str(x) for x in invalid_scale_sizes)}."
            )

        # Scalars are transformed to lists with length num_layers
        scale_configs = [[s] * num_layers if isinstance(s, scale_type) else s for s in scale]

        # Set scales. zip over scale_configs prevents going into single transformer layers
        for attn_processor, *scale in zip(self.transformer.attn_processors.values(), *scale_configs):
            attn_processor.scale = scale

    def unload_ip_adapter(self):
        """
        Unloads the IP Adapter weights

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the IP Adapter weights.
        >>> pipeline.unload_ip_adapter()
        >>> ...
        ```
        """
        # remove CLIP image encoder
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is not None:
            self.image_encoder = None
            self.register_to_config(image_encoder=[None, None])

        # remove feature extractor only when safety_checker is None as safety_checker uses
        # the feature_extractor later
        if not hasattr(self, "safety_checker"):
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is not None:
                self.feature_extractor = None
                self.register_to_config(feature_extractor=[None, None])

        # remove hidden encoder
        self.transformer.encoder_hid_proj = None
        self.transformer.config.encoder_hid_dim_type = None

        # restore original Transformer attention processors layers
        attn_procs = {}
        for name, value in self.transformer.attn_processors.items():
            attn_processor_class = FluxAttnProcessor2_0()
            attn_procs[name] = (
                attn_processor_class if isinstance(value, (FluxIPAdapterJointAttnProcessor2_0)) else value.__class__()
            )
        self.transformer.set_attn_processor(attn_procs)


class SD3IPAdapterMixin:
    """Mixin for handling StableDiffusion 3 IP Adapters."""

    @property
    def is_ip_adapter_active(self) -> bool:
        """Checks if IP-Adapter is loaded and scale > 0.

        IP-Adapter scale controls the influence of the image prompt versus text prompt. When this value is set to 0,
        the image context is irrelevant.

        Returns:
            `bool`: True when IP-Adapter is loaded and any layer has scale > 0.
        """
        scales = [
            attn_proc.scale
            for attn_proc in self.transformer.attn_processors.values()
            if isinstance(attn_proc, SD3IPAdapterJointAttnProcessor2_0)
        ]

        return len(scales) > 0 and any(scale > 0 for scale in scales)

    @validate_hf_hub_args
    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str = "ip-adapter.safetensors",
        subfolder: Optional[str] = None,
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,
    ) -> None:
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:
                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
            weight_name (`str`, defaults to "ip-adapter.safetensors"):
                The name of the weight file to load. If a list is passed, it should have the same length as
                `subfolder`.
            subfolder (`str`, *optional*):
                The subfolder location of a model file within a larger model repository on the Hub or locally. If a
                list is passed, it should have the same length as `weight_name`.
            image_encoder_folder (`str`, *optional*, defaults to `image_encoder`):
                The subfolder location of the image encoder within a larger model repository on the Hub or locally.
                Pass `None` to not load the image encoder. If the image encoder is located in a folder inside
                `subfolder`, you only need to pass the name of the folder that contains image encoder weights, e.g.
                `image_encoder_folder="image_encoder"`. If the image encoder is located in a folder other than
                `subfolder`, you should pass the path to the folder that contains image encoder weights, for example,
                `image_encoder_folder="different_subfolder/image_encoder"`.
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
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
        """
        # Load the main state dict first
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        if low_cpu_mem_usage and not is_accelerate_available():
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

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = load_state_dict(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if "image_proj" not in keys and "ip_adapter" not in keys:
            raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

        # Load image_encoder and feature_extractor here if they haven't been registered to the pipeline yet
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
            if image_encoder_folder is not None:
                if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                    logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
                    if image_encoder_folder.count("/") == 0:
                        image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
                    else:
                        image_encoder_subfolder = Path(image_encoder_folder).as_posix()

                    # Commons args for loading image encoder and image processor
                    kwargs = {
                        "low_cpu_mem_usage": low_cpu_mem_usage,
                        "cache_dir": cache_dir,
                        "local_files_only": local_files_only,
                    }

                    self.register_modules(
                        feature_extractor=SiglipImageProcessor.from_pretrained(image_encoder_subfolder, **kwargs),
                        image_encoder=SiglipVisionModel.from_pretrained(
                            image_encoder_subfolder, torch_dtype=self.dtype, **kwargs
                        ).to(self.device),
                    )
                else:
                    raise ValueError(
                        "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                    )
            else:
                logger.warning(
                    "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                    "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
                )

        # Load IP-Adapter into transformer
        self.transformer._load_ip_adapter_weights(state_dict, low_cpu_mem_usage=low_cpu_mem_usage)

    def set_ip_adapter_scale(self, scale: float) -> None:
        """
        Set IP-Adapter scale, which controls image prompt conditioning. A value of 1.0 means the model is only
        conditioned on the image prompt, and 0.0 only conditioned by the text prompt. Lowering this value encourages
        the model to produce more diverse images, but they may not be as aligned with the image prompt.

        Example:

        ```python
        >>> # Assuming `pipeline` is already loaded with the IP Adapter weights.
        >>> pipeline.set_ip_adapter_scale(0.6)
        >>> ...
        ```

        Args:
            scale (float):
                IP-Adapter scale to be set.

        """
        for attn_processor in self.transformer.attn_processors.values():
            if isinstance(attn_processor, SD3IPAdapterJointAttnProcessor2_0):
                attn_processor.scale = scale

    def unload_ip_adapter(self) -> None:
        """
        Unloads the IP Adapter weights.

        Example:

        ```python
        >>> # Assuming `pipeline` is already loaded with the IP Adapter weights.
        >>> pipeline.unload_ip_adapter()
        >>> ...
        ```
        """
        # Remove image encoder
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is not None:
            self.image_encoder = None
            self.register_to_config(image_encoder=None)

        # Remove feature extractor
        if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is not None:
            self.feature_extractor = None
            self.register_to_config(feature_extractor=None)

        # Remove image projection
        self.transformer.image_proj = None

        # Restore original attention processors layers
        attn_procs = {
            name: (
                JointAttnProcessor2_0() if isinstance(value, SD3IPAdapterJointAttnProcessor2_0) else value.__class__()
            )
            for name, value in self.transformer.attn_processors.items()
        }
        self.transformer.set_attn_processor(attn_procs)
