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
from contextlib import nullcontext

from ..models.embeddings import (
    ImageProjection,
    MultiIPAdapterImageProjection,
)
from ..models.model_loading_utils import load_model_dict_into_meta
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from ..utils import is_accelerate_available, is_torch_version, logging
from ..utils.torch_utils import empty_device_cache


if is_accelerate_available():
    pass

logger = logging.get_logger(__name__)


class FluxTransformer2DLoadersMixin:
    """
    Load layers into a [`FluxTransformer2DModel`].
    """

    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
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
            if state_dict["proj.weight"].shape[0] == 65536:
                num_image_text_embeds = 16
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // num_image_text_embeds

            with init_context():
                image_projection = ImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=clip_embeddings_dim,
                    num_image_text_embeds=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        if not low_cpu_mem_usage:
            image_projection.load_state_dict(updated_state_dict, strict=True)
        else:
            device_map = {"": self.device}
            load_model_dict_into_meta(image_projection, updated_state_dict, device_map=device_map, dtype=self.dtype)
            empty_device_cache()

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
        from ..models.transformers.transformer_flux import FluxIPAdapterAttnProcessor

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
        key_id = 0
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
        for name in self.attn_processors.keys():
            if name.startswith("single_transformer_blocks"):
                attn_processor_class = self.attn_processors[name].__class__
                attn_procs[name] = attn_processor_class()
            else:
                cross_attention_dim = self.config.joint_attention_dim
                hidden_size = self.inner_dim
                attn_processor_class = FluxIPAdapterAttnProcessor
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        num_image_text_embed = 4
                        if state_dict["image_proj"]["proj.weight"].shape[0] == 65536:
                            num_image_text_embed = 16
                        # IP-Adapter
                        num_image_text_embeds += [num_image_text_embed]

                with init_context():
                    attn_procs[name] = attn_processor_class(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_image_text_embeds,
                        dtype=self.dtype,
                        device=self.device,
                    )

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})
                    value_dict.update({f"to_k_ip.{i}.bias": state_dict["ip_adapter"][f"{key_id}.to_k_ip.bias"]})
                    value_dict.update({f"to_v_ip.{i}.bias": state_dict["ip_adapter"][f"{key_id}.to_v_ip.bias"]})

                if not low_cpu_mem_usage:
                    attn_procs[name].load_state_dict(value_dict)
                else:
                    device_map = {"": self.device}
                    dtype = self.dtype
                    load_model_dict_into_meta(attn_procs[name], value_dict, device_map=device_map, dtype=dtype)

                key_id += 1

        empty_device_cache()

        return attn_procs

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]

        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
        self.set_attn_processor(attn_procs)

        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"
