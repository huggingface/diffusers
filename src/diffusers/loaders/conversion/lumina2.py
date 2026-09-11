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


def lumina2_conversion(config):
    modules = [
        ("x_embedder", "x_embedder"),
        ("t_embedder.mlp.0", "time_caption_embed.timestep_embedder.linear_1"),
        ("t_embedder.mlp.2", "time_caption_embed.timestep_embedder.linear_2"),
        ("cap_embedder.1", "time_caption_embed.caption_embedder.1"),
        ("final_layer.adaLN_modulation.1", "norm_out.linear_1"),
        ("final_layer.linear", "norm_out.linear_2"),
    ]
    mapping = {"cap_embedder.0.weight": "time_caption_embed.caption_embedder.0.weight"}
    rules = []
    hidden = config["hidden_size"]
    kv = hidden // config["num_attention_heads"] * config["num_kv_heads"]
    for group, count in (
        ("noise_refiner", config["num_refiner_layers"]),
        ("context_refiner", config["num_refiner_layers"]),
        ("layers", config["num_layers"]),
    ):
        for i in range(count):
            prefix = f"{group}.{i}"
            rules.append(
                Rule(
                    (prefix + ".attention.qkv.weight",),
                    tuple(f"{prefix}.attn.to_{part}.weight" for part in ("q", "k", "v")),
                    Split((hidden, kv, kv)),
                )
            )
            for old, new in (
                ("attention.q_norm", "attn.norm_q"),
                ("attention.k_norm", "attn.norm_k"),
                ("attention.out", "attn.to_out.0"),
                ("feed_forward.w1", "feed_forward.linear_1"),
                ("feed_forward.w2", "feed_forward.linear_2"),
                ("feed_forward.w3", "feed_forward.linear_3"),
                ("attention_norm1", "norm1" if group == "context_refiner" else "norm1.norm"),
                ("attention_norm2", "norm2"),
                ("ffn_norm1", "ffn_norm1"),
                ("ffn_norm2", "ffn_norm2"),
            ):
                mapping[f"{prefix}.{old}.weight"] = f"{prefix}.{new}.weight"
            if group != "context_refiner":
                modules.append((prefix + ".adaLN_modulation.1", prefix + ".norm1.linear"))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
