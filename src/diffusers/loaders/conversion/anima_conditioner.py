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


def anima_conditioner_conversion(config):
    keys = ["embed.weight", "norm.weight"]
    modules = ["out_proj"]
    if config["model_dim"] != config["target_dim"]:
        modules.append("in_proj")
    for i in range(config["num_layers"]):
        prefix = f"blocks.{i}"
        norms = ["norm_cross_attn", "norm_mlp"]
        attentions = ["cross_attn"]
        if config["use_self_attention"]:
            norms.append("norm_self_attn")
            attentions.append("self_attn")
        for norm in norms:
            keys.append(f"{prefix}.{norm}.weight")
            if config["use_layer_norm"]:
                keys.append(f"{prefix}.{norm}.bias")
        for attn in attentions:
            keys.extend(
                f"{prefix}.{attn}.{name}.weight"
                for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
            )
        modules.extend(f"{prefix}.mlp.{j}" for j in (0, 2))
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in keys})
