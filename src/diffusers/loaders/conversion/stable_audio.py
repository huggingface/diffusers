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
from .transforms import Reshape, Split


def stable_audio_conversion(config):
    mapping = {
        "preprocess_conv.weight": "preprocess_conv.weight",
        "postprocess_conv.weight": "postprocess_conv.weight",
        "transformer.project_in.weight": "proj_in.weight",
        "transformer.project_out.weight": "proj_out.weight",
    }
    modules = [("to_timestep_embed.0", "timestep_proj.0"), ("to_timestep_embed.2", "timestep_proj.2")]
    for source, target in (("to_global_embed", "global_proj"), ("to_cond_embed", "cross_attention_proj")):
        mapping.update({f"{source}.{i}.weight": f"{target}.{i}.weight" for i in (0, 2)})
    fourier = config["time_proj_dim"] // 2
    rules = [Rule(("timestep_features.weight",), ("time_proj.weight",), Reshape((fourier, 1), (fourier,)))]
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    kv = config["num_key_value_attention_heads"] * config["attention_head_dim"]
    for i in range(config["num_layers"]):
        old, new = f"transformer.layers.{i}", f"transformer_blocks.{i}"
        for a, b in (("pre_norm", "norm1"), ("cross_attend_norm", "norm2"), ("ff_norm", "norm3")):
            mapping[f"{old}.{a}.gamma"] = f"{new}.{b}.weight"
            mapping[f"{old}.{a}.beta"] = f"{new}.{b}.bias"
        rules.append(
            Rule(
                (old + ".self_attn.to_qkv.weight",),
                tuple(f"{new}.attn1.to_{part}.weight" for part in ("q", "k", "v")),
                Split((hidden,) * 3),
            )
        )
        rules.append(
            Rule(
                (old + ".cross_attn.to_kv.weight",),
                tuple(f"{new}.attn2.to_{part}.weight" for part in ("k", "v")),
                Split((kv,) * 2),
            )
        )
        mapping.update(
            {
                f"{old}.{a}.weight": f"{new}.{b}.weight"
                for a, b in (
                    ("self_attn.to_out", "attn1.to_out.0"),
                    ("cross_attn.to_q", "attn2.to_q"),
                    ("cross_attn.to_out", "attn2.to_out.0"),
                )
            }
        )
        modules.extend((f"{old}.ff.ff.{name}", f"{new}.ff.net.{name}") for name in ("0.proj", "2"))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
