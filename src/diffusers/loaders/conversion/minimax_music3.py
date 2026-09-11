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


from .core import Conversion, Rule
from .transforms import Split


def minimax_music3_conversion(config):
    mapping = {
        "timestep_features.weight": "time_proj.weight",
        "preprocess_conv.weight": "preprocess_conv.weight",
        "postprocess_conv.weight": "postprocess_conv.weight",
        "transformer.project_in.weight": "proj_in.weight",
        "transformer.project_out.weight": "proj_out.weight",
    }
    modules = [("to_timestep_embed.0", "time_embed.linear_1"), ("to_timestep_embed.2", "time_embed.linear_2")]
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    rules = []
    for i in range(config["num_layers"]):
        old, new = f"transformer.layers.{i}", f"transformer_blocks.{i}"
        for a, b in (("pre_norm", "norm1"), ("ff_norm", "norm2")):
            mapping[f"{old}.{a}.gamma"] = f"{new}.{b}.weight"
            mapping[f"{old}.{a}.beta"] = f"{new}.{b}.bias"
        rules.append(
            Rule(
                (old + ".self_attn.to_qkv.weight",),
                tuple(f"{new}.attn.to_{part}.weight" for part in ("q", "k", "v")),
                Split((hidden,) * 3),
            )
        )
        mapping[old + ".self_attn.to_out.weight"] = new + ".attn.to_out.0.weight"
        modules.extend([(old + ".ff.ff.0.proj", new + ".ff_in"), (old + ".ff.ff.2", new + ".ff_out")])
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
