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
from .transforms import MergeEqual, Reverse, Split


def dit_conversion(config):
    modules = [
        ("x_embedder.proj", "pos_embed.proj"),
        ("final_layer.adaLN_modulation.1", "proj_out_1"),
        ("final_layer.linear", "proj_out_2"),
    ]
    mapping, rules = {}, []
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    embeddings = [
        ("t_embedder.mlp.0", "timestep_embedder.linear_1"),
        ("t_embedder.mlp.2", "timestep_embedder.linear_2"),
    ]
    for old, new in embeddings:
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (f"{old}.{p}",),
                    tuple(f"transformer_blocks.{i}.norm1.emb.{new}.{p}" for i in range(config["num_layers"])),
                    Reverse(MergeEqual(config["num_layers"])),
                )
            )
    rules.append(
        Rule(
            ("y_embedder.embedding_table.weight",),
            tuple(
                f"transformer_blocks.{i}.norm1.emb.class_embedder.embedding_table.weight"
                for i in range(config["num_layers"])
            ),
            Reverse(MergeEqual(config["num_layers"])),
        )
    )
    for i in range(config["num_layers"]):
        old, new = f"blocks.{i}", f"transformer_blocks.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("adaLN_modulation.1", "norm1.linear"),
                ("attn.proj", "attn1.to_out.0"),
                ("mlp.fc1", "ff.net.0.proj"),
                ("mlp.fc2", "ff.net.2"),
            )
        )
        for p in ("weight", "bias") if config["attention_bias"] else ("weight",):
            rules.append(
                Rule(
                    (f"{old}.attn.qkv.{p}",),
                    tuple(f"{new}.attn1.to_{part}.{p}" for part in ("q", "k", "v")),
                    Split((hidden,) * 3),
                )
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
