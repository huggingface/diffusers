# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from .core import Conversion


def qwen_image_conversion(config):
    """Validate the published Qwen Image layout, which already uses Diffusers parameter names."""
    modules = [
        "img_in",
        "txt_in",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
        "norm_out.linear",
        "proj_out",
    ]
    keys = ["txt_norm.weight"]
    if config["use_additional_t_cond"]:
        keys.append("time_text_embed.addition_t_embedding.weight")
    for i in range(config["num_layers"]):
        prefix = f"transformer_blocks.{i}"
        modules.extend(
            f"{prefix}.{modality}_{name}"
            for modality in ("img", "txt")
            for name in ("mod.1", "mlp.net.0.proj", "mlp.net.2")
        )
        modules.extend(
            f"{prefix}.attn.{name}"
            for name in ("to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")
        )
        keys.extend(f"{prefix}.attn.{name}.weight" for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"))
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in keys})
