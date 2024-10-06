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

import functools
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from huggingface_hub.utils import validate_hf_hub_args

from ..configuration_utils import ConfigMixin
from ..models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    GeneralizedLinearAttnProcessor,
)
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, ModelMixin
from ..utils import (
    is_accelerate_available,
    is_torch_version,
    logging,
)


logger = logging.get_logger(__name__)

model_dict = model_dict = {
    "CompVis/stable-diffusion-v1-4": "Yuanshi/LinFusion-1-5",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "Yuanshi/LinFusion-1-5",
    "SG161222/Realistic_Vision_V5.1_noVAE": "Yuanshi/LinFusion-1-5",
    "Lykon/dreamshaper-8": "Yuanshi/LinFusion-1-5",
    "timbrooks/instruct-pix2pix": "Yuanshi/LinFusion-1-5",
    "stabilityai/stable-diffusion-2-1": "Yuanshi/LinFusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0": "Yuanshi/LinFusion-XL",
    "stabilityai/sdxl-turbo": "Yuanshi/LinFusion-XL",
    "diffusers/sdxl-instructpix2pix-768": "Yuanshi/LinFusion-XL",
}


class LinFusionMixin(ModelMixin, ConfigMixin):
    """Mixin for handling LinFusion."""

    def __init__(self, modules_list, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
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

        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext

        self.modules_dict = {}
        self.register_to_config(modules_list=modules_list)

        for i, attention_config in enumerate(modules_list):
            dim_n = attention_config["dim_n"]
            heads = attention_config["heads"]
            projection_mid_dim = attention_config["projection_mid_dim"]

            with init_context():
                processor = GeneralizedLinearAttnProcessor(
                    dim_n=dim_n, heads=heads, projection_mid_dim=projection_mid_dim
                )

            self.add_module(f"{i}", processor)
            self.modules_dict[attention_config["module_name"]] = processor

    @classmethod
    def get_default_config(
        cls,
        pipeline=None,
        unet=None,
    ):
        """
        Get the default configuration for the LinFusion model.
        (The `projection_mid_dim` is same as the `query_dim` by default.)
        """
        assert unet is not None or pipeline.unet is not None
        unet = unet or pipeline.unet
        modules_list = []
        for module_name, module in unet.named_modules():
            if not isinstance(module, Attention):
                continue
            if module.is_cross_attention:
                continue
            dim_n = module.to_q.weight.shape[0]
            modules_list.append(
                {
                    "module_name": module_name,
                    "dim_n": dim_n,
                    "heads": module.heads,
                    "projection_mid_dim": None,
                }
            )
        return {"modules_list": modules_list}

    @classmethod
    @validate_hf_hub_args
    def load_linfusion(
        cls,
        pipeline,
        pretrained_model_name_or_path_or_dict: Optional[Union[str, Path, Dict]] = None,
        pipeline_name_or_path: Optional[Union[str, Path]] = None,
        load_pretrained: Optional[bool] = True,
        **kwargs,
    ):
        r"""
        Load LinFusion modules into [`UNet2DConditionModel`].
        """
        unet = (
            getattr(pipeline, pipeline.unet_name)
            if not hasattr(pipeline, "unet")
            else pipeline.unet
        )
        if load_pretrained or not isinstance(
            pretrained_model_name_or_path_or_dict, dict
        ):
            if not pretrained_model_name_or_path_or_dict:
                pipeline_name_or_path = (
                    pipeline_name_or_path or pipeline._internal_dict._name_or_path
                )
                pretrained_model_name_or_path_or_dict = model_dict.get(
                    pipeline_name_or_path, None
                )
                if not pretrained_model_name_or_path_or_dict:
                    raise ValueError(
                        f"LinFusion not found for pipeline [{pipeline_name_or_path}]. "
                        "Try specify `pretrained_model_name_or_path` explicitly."
                    )
                else:
                    logger.info(f"{pipeline_name_or_path} matches LinFusion checkpoint "
                                f"{pretrained_model_name_or_path_or_dict}")
            if (
                pretrained_model_name_or_path_or_dict == "Yuanshi/LinFusion-2-1"
                and unet.dtype == torch.float16
            ):
                logger.warning(
                    "`Yuanshi/LinFusion-2-1` may cause numerical instability under fp16. "
                    "torch.bfloat16 or torch.float32 is recommended for this pipeline."
                )

            linfusion = (
                LinFusionMixin.from_pretrained(
                    pretrained_model_name_or_path_or_dict, **kwargs
                )
                .to(unet.device)
                .to(unet.dtype)
            )
        else:
            default_config = LinFusionMixin.get_default_config(unet=unet)
            linfusion = LinFusionMixin(**default_config).to(unet.device).to(unet.dtype)
        linfusion.mount_to(
            unet,
            modules_dict=(
                pretrained_model_name_or_path_or_dict
                if isinstance(pretrained_model_name_or_path_or_dict, dict)
                else None
            ),
        )

    def mount_to(self, unet, modules_dict=None):
        r"""
        Mounts the modules in the `modules_dict` to the given `unet`.
        """
        modules_dict = modules_dict or self.modules_dict
        for module_name, module in modules_dict.items():
            parent_module = functools.reduce(getattr, module_name.split("."), unet)
            parent_module.set_processor(module)

        self.to(unet.device).to(unet.dtype)

    @classmethod
    def unload_linfusion(cls, pipeline):
        r"""
        Unload the LinFusion weights
        """
        unet = (
            getattr(pipeline, pipeline.unet_name)
            if not hasattr(pipeline, "unet")
            else pipeline.unet
        )
        attn_procs = {}
        for name, value in unet.attn_processors.items():
            attn_processor_class = (
                AttnProcessor2_0()
                if hasattr(F, "scaled_dot_product_attention")
                else AttnProcessor()
            )
            attn_procs[name] = (
                attn_processor_class
                if isinstance(value, GeneralizedLinearAttnProcessor)
                else value.__class__()
            )
        unet.set_attn_processor(attn_procs)
