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
from .transforms import ReorderChunks, Split


def cogview3plus_conversion(config):
    modules = [
        ("mixins.patch_embed.proj", "patch_embed.proj"),
        ("mixins.patch_embed.text_proj", "patch_embed.text_proj"),
        ("mixins.final_layer.linear", "proj_out"),
    ]
    modules.extend(
        (f"{source}.{i}", f"time_condition_embed.{target}.linear_{j}")
        for source, target in (("time_embed", "timestep_embedder"), ("label_emb.0", "condition_embedder"))
        for i, j in ((0, 1), (2, 2))
    )
    rules = []
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    for i in range(config["num_layers"]):
        old, new = f"transformer.layers.{i}", f"transformer_blocks.{i}"
        modules.append((f"mixins.adaln.adaln_modules.{i}.1", new + ".norm1.linear"))
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("attention.dense", "attn1.to_out.0"),
                ("mlp.dense_h_to_4h", "ff.net.0.proj"),
                ("mlp.dense_4h_to_h", "ff.net.2"),
            )
        )
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (f"{old}.attention.query_key_value.{p}",),
                    tuple(f"{new}.attn1.to_{part}.{p}" for part in ("q", "k", "v")),
                    Split((hidden,) * 3),
                )
            )
    for p in ("weight", "bias"):
        rules.append(Rule((f"mixins.final_layer.adaln.1.{p}",), (f"norm_out.linear.{p}",), ReorderChunks((1, 0))))
    mapping = {f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")}
    return Conversion(mapping=mapping, rules=tuple(rules))
