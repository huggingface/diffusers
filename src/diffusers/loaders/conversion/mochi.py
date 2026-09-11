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


def mochi_conversion(config):
    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    mapping = {"pos_frequencies": "pos_frequencies"}
    modules = [
        ("x_embedder.proj", "patch_embed.proj"),
        ("t_embedder.mlp.0", "time_embed.timestep_embedder.linear_1"),
        ("t_embedder.mlp.2", "time_embed.timestep_embedder.linear_2"),
        ("t5_yproj", "time_embed.caption_proj"),
        ("final_layer.linear", "proj_out"),
    ]
    modules.extend(("t5_y_embedder." + name, "time_embed.pooler." + name) for name in ("to_q", "to_kv", "to_out"))
    rules = []
    for i in range(config["num_layers"]):
        old, new = f"blocks.{i}", f"transformer_blocks.{i}"
        modules.extend(
            [
                (old + ".mod_x", new + ".norm1.linear"),
                (
                    old + ".mod_y",
                    new + (".norm1_context.linear_1" if i == config["num_layers"] - 1 else ".norm1_context.linear"),
                ),
            ]
        )
        for branch, attention, output, norms, ff in (
            ("x", ("to_q", "to_k", "to_v"), "to_out.0", ("norm_q", "norm_k"), "ff"),
            (
                "y",
                ("add_q_proj", "add_k_proj", "add_v_proj"),
                "to_add_out",
                ("norm_added_q", "norm_added_k"),
                "ff_context",
            ),
        ):
            rules.append(
                Rule(
                    (f"{old}.attn.qkv_{branch}.weight",),
                    tuple(f"{new}.attn1.{name}.weight" for name in attention),
                    Split((hidden_size,) * 3),
                )
            )
            for source, target in zip(("q", "k"), norms):
                mapping[f"{old}.attn.{source}_norm_{branch}.weight"] = f"{new}.attn1.{target}.weight"
            if branch == "x" or i < config["num_layers"] - 1:
                modules.append((f"{old}.attn.proj_{branch}", f"{new}.attn1.{output}"))
                rules.append(
                    Rule((f"{old}.mlp_{branch}.w1.weight",), (f"{new}.{ff}.net.0.proj.weight",), ReorderChunks((1, 0)))
                )
                mapping[f"{old}.mlp_{branch}.w2.weight"] = f"{new}.{ff}.net.2.weight"
    for p in ("weight", "bias"):
        rules.append(Rule((f"final_layer.mod.{p}",), (f"norm_out.linear.{p}",), ReorderChunks((1, 0))))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
