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
from contextlib import nullcontext
from typing import Dict

from ..models.attention_processor import SD3IPAdapterJointAttnProcessor2_0
from ..models.embeddings import IPAdapterTimeImageProjection
from ..models.modeling_utils import load_model_dict_into_meta
from ..utils import is_accelerate_available, is_torch_version, logging


logger = logging.get_logger(__name__)


class SD3Transformer2DLoadersMixin:
    """Load IP-Adapters and LoRA layers into a `[SD3Transformer2DModel]`."""

    def _convert_ip_adapter_attn_to_diffusers(self, state_dict: Dict) -> Dict:
        # Always use memory-efficient loading
        if not is_accelerate_available():
            raise ImportError(
                "Memory-efficient loading requires `accelerate`. Please install it with: \n```\npip install accelerate\n```\n."
            )

        if not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Memory-efficient loading requires PyTorch >= 1.9.0. Please update your PyTorch version."
            )

        from accelerate import init_empty_weights

        # IP-Adapter cross attention parameters
        hidden_size = self.config.attention_head_dim * self.config.num_attention_heads
        ip_hidden_states_dim = self.config.attention_head_dim * self.config.num_attention_heads
        timesteps_emb_dim = state_dict["0.norm_ip.linear.weight"].shape[1]

        # Dict where key is transformer layer index, value is attention processor's state dict
        # ip_adapter state dict keys example: "0.norm_ip.linear.weight"
        layer_state_dict = {idx: {} for idx in range(len(self.attn_processors))}
        for key, weights in state_dict.items():
            idx, name = key.split(".", maxsplit=1)
            layer_state_dict[int(idx)][name] = weights

        # Create IP-Adapter attention processor & load state_dict
        attn_procs = {}
        for idx, name in enumerate(self.attn_processors.keys()):
            with init_empty_weights():
                attn_procs[name] = SD3IPAdapterJointAttnProcessor2_0(
                    hidden_size=hidden_size,
                    ip_hidden_states_dim=ip_hidden_states_dim,
                    head_dim=self.config.attention_head_dim,
                    timesteps_emb_dim=timesteps_emb_dim,
                )

            # Always use memory-efficient loading
            device_map = {"": self.device}
            load_model_dict_into_meta(
                attn_procs[name], layer_state_dict[idx], device_map=device_map, dtype=self.dtype
            )

        return attn_procs

    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict: Dict) -> IPAdapterTimeImageProjection:
        # Always use memory-efficient loading
        if not is_accelerate_available():
            raise ImportError(
                "Memory-efficient loading requires `accelerate`. Please install it with: \n```\npip install accelerate\n```\n."
            )

        if not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Memory-efficient loading requires PyTorch >= 1.9.0. Please update your PyTorch version."
            )

        from accelerate import init_empty_weights

        # Convert to diffusers
        updated_state_dict = {}
        for key, value in state_dict.items():
            # InstantX/SD3.5-Large-IP-Adapter
            if key.startswith("layers."):
                idx = key.split(".")[1]
                key = key.replace(f"layers.{idx}.0.norm1", f"layers.{idx}.ln0")
                key = key.replace(f"layers.{idx}.0.norm2", f"layers.{idx}.ln1")
                key = key.replace(f"layers.{idx}.0.to_q", f"layers.{idx}.attn.to_q")
                key = key.replace(f"layers.{idx}.0.to_kv", f"layers.{idx}.attn.to_kv")
                key = key.replace(f"layers.{idx}.0.to_out", f"layers.{idx}.attn.to_out.0")
                key = key.replace(f"layers.{idx}.1.0", f"layers.{idx}.adaln_norm")
                key = key.replace(f"layers.{idx}.1.1", f"layers.{idx}.ff.net.0.proj")
                key = key.replace(f"layers.{idx}.1.3", f"layers.{idx}.ff.net.2")
                key = key.replace(f"layers.{idx}.2.1", f"layers.{idx}.adaln_proj")
            updated_state_dict[key] = value

        # Image projection parameters
        embed_dim = updated_state_dict["proj_in.weight"].shape[1]
        output_dim = updated_state_dict["proj_out.weight"].shape[0]
        hidden_dim = updated_state_dict["proj_in.weight"].shape[0]
        heads = updated_state_dict["layers.0.attn.to_q.weight"].shape[0] // 64
        num_queries = updated_state_dict["latents"].shape[1]
        timestep_in_dim = updated_state_dict["time_embedding.linear_1.weight"].shape[1]

        # Image projection
        with init_empty_weights():
            image_proj = IPAdapterTimeImageProjection(
                embed_dim=embed_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                num_queries=num_queries,
                timestep_in_dim=timestep_in_dim,
            )

        # Always use memory-efficient loading
        device_map = {"": self.device}
        load_model_dict_into_meta(image_proj, updated_state_dict, device_map=device_map, dtype=self.dtype)

        return image_proj

    def _load_ip_adapter_weights(self, state_dict: Dict) -> None:
        """Sets IP-Adapter attention processors, image projection, and loads state_dict.

        Args:
            state_dict (`Dict`):
                State dict with keys "ip_adapter", which contains parameters for attention processors, and
                "image_proj", which contains parameters for image projection net.
        """

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dict["ip_adapter"])
        self.set_attn_processor(attn_procs)

        self.image_proj = self._convert_ip_adapter_image_proj_to_diffusers(state_dict["image_proj"])
