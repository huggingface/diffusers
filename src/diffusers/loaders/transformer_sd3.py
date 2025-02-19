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
from typing import Dict

from ..models.attention_processor import SD3IPAdapterJointAttnProcessor2_0
from ..models.embeddings import IPAdapterTimeImageProjection
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta


class SD3Transformer2DLoadersMixin:
    """Load IP-Adapters and LoRA layers into a `[SD3Transformer2DModel]`."""

    def _load_ip_adapter_weights(self, state_dict: Dict, low_cpu_mem_usage: bool = _LOW_CPU_MEM_USAGE_DEFAULT) -> None:
        """Sets IP-Adapter attention processors, image projection, and loads state_dict.

        Args:
            state_dict (`Dict`):
                State dict with keys "ip_adapter", which contains parameters for attention processors, and
                "image_proj", which contains parameters for image projection net.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
        """
        # IP-Adapter cross attention parameters
        hidden_size = self.config.attention_head_dim * self.config.num_attention_heads
        ip_hidden_states_dim = self.config.attention_head_dim * self.config.num_attention_heads
        timesteps_emb_dim = state_dict["ip_adapter"]["0.norm_ip.linear.weight"].shape[1]

        # Dict where key is transformer layer index, value is attention processor's state dict
        # ip_adapter state dict keys example: "0.norm_ip.linear.weight"
        layer_state_dict = {idx: {} for idx in range(len(self.attn_processors))}
        for key, weights in state_dict["ip_adapter"].items():
            idx, name = key.split(".", maxsplit=1)
            layer_state_dict[int(idx)][name] = weights

        # Create IP-Adapter attention processor
        attn_procs = {}
        for idx, name in enumerate(self.attn_processors.keys()):
            attn_procs[name] = SD3IPAdapterJointAttnProcessor2_0(
                hidden_size=hidden_size,
                ip_hidden_states_dim=ip_hidden_states_dim,
                head_dim=self.config.attention_head_dim,
                timesteps_emb_dim=timesteps_emb_dim,
            ).to(self.device, dtype=self.dtype)

            if not low_cpu_mem_usage:
                attn_procs[name].load_state_dict(layer_state_dict[idx], strict=True)
            else:
                load_model_dict_into_meta(
                    attn_procs[name], layer_state_dict[idx], device=self.device, dtype=self.dtype
                )

        self.set_attn_processor(attn_procs)

        # Image projetion parameters
        embed_dim = state_dict["image_proj"]["proj_in.weight"].shape[1]
        output_dim = state_dict["image_proj"]["proj_out.weight"].shape[0]
        hidden_dim = state_dict["image_proj"]["proj_in.weight"].shape[0]
        heads = state_dict["image_proj"]["layers.0.attn.to_q.weight"].shape[0] // 64
        num_queries = state_dict["image_proj"]["latents"].shape[1]
        timestep_in_dim = state_dict["image_proj"]["time_embedding.linear_1.weight"].shape[1]

        # Image projection
        self.image_proj = IPAdapterTimeImageProjection(
            embed_dim=embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            num_queries=num_queries,
            timestep_in_dim=timestep_in_dim,
        ).to(device=self.device, dtype=self.dtype)

        if not low_cpu_mem_usage:
            self.image_proj.load_state_dict(state_dict["image_proj"], strict=True)
        else:
            load_model_dict_into_meta(self.image_proj, state_dict["image_proj"], device=self.device, dtype=self.dtype)
