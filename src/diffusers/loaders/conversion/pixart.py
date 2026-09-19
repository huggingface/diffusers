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


def pixart_conversion(config):
    modules = [
        ("x_embedder.proj", "pos_embed.proj"),
        ("y_embedder.y_proj.fc1", "caption_projection.linear_1"),
        ("y_embedder.y_proj.fc2", "caption_projection.linear_2"),
        ("t_block.1", "adaln_single.linear"),
        ("final_layer.linear", "proj_out"),
    ]
    if config["caption_channels"] is None:
        modules = [(old, new) for old, new in modules if not new.startswith("caption_projection.")]
    mapping = {"final_layer.scale_shift_table": "scale_shift_table"}
    rules = []
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    embeddings = [("t_embedder", "timestep_embedder")]
    additional = config.get("use_additional_conditions")
    if additional is None:
        additional = config["sample_size"] == 128
    if additional:
        embeddings.extend([("csize_embedder", "resolution_embedder"), ("ar_embedder", "aspect_ratio_embedder")])
    for old, new in embeddings:
        modules.extend((f"{old}.mlp.{i}", f"adaln_single.emb.{new}.linear_{j}") for i, j in ((0, 1), (2, 2)))
    for i in range(config["num_layers"]):
        old, new = f"blocks.{i}", f"transformer_blocks.{i}"
        mapping[old + ".scale_shift_table"] = new + ".scale_shift_table"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("attn.proj", "attn1.to_out.0"),
                ("cross_attn.q_linear", "attn2.to_q"),
                ("cross_attn.proj", "attn2.to_out.0"),
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
            rules.append(
                Rule(
                    (f"{old}.cross_attn.kv_linear.{p}",),
                    tuple(f"{new}.attn2.to_{part}.{p}" for part in ("k", "v")),
                    Split((hidden,) * 2),
                )
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
