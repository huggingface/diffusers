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


def sana_conversion(config):
    mapping = {"attention_y_norm.weight": "caption_norm.weight", "final_layer.scale_shift_table": "scale_shift_table"}
    modules = [
        ("x_embedder.proj", "patch_embed.proj"),
        ("t_block.1", "time_embed.linear"),
        ("y_embedder.y_proj.fc1", "caption_projection.linear_1"),
        ("y_embedder.y_proj.fc2", "caption_projection.linear_2"),
        ("final_layer.linear", "proj_out"),
    ]
    timed = "time_embed" if config.get("guidance_embeds", False) else "time_embed.emb"
    modules.extend((f"t_embedder.mlp.{i}", f"{timed}.timestep_embedder.linear_{j}") for i, j in ((0, 1), (2, 2)))
    if config.get("guidance_embeds", False):
        modules.extend(
            (f"cfg_embedder.mlp.{i}", f"time_embed.guidance_embedder.linear_{j}") for i, j in ((0, 1), (2, 2))
        )
    rules = []
    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    has_cross = config["cross_attention_dim"] is not None
    cross_size = config["num_cross_attention_heads"] * config["cross_attention_head_dim"] if has_cross else None
    for i in range(config["num_layers"]):
        old, new = f"blocks.{i}", f"transformer_blocks.{i}"
        mapping[old + ".scale_shift_table"] = new + ".scale_shift_table"
        for p in ("weight", "bias") if config["attention_bias"] else ("weight",):
            rules.append(
                Rule(
                    (f"{old}.attn.qkv.{p}",),
                    tuple(f"{new}.attn1.to_{part}.{p}" for part in ("q", "k", "v")),
                    Split((hidden_size,) * 3),
                )
            )
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("attn.proj", "attn1.to_out.0"),
                ("mlp.inverted_conv.conv", "ff.conv_inverted"),
                ("mlp.depth_conv.conv", "ff.conv_depth"),
            )
        )
        mapping[old + ".mlp.point_conv.conv.weight"] = new + ".ff.conv_point.weight"
        if has_cross:
            for p in ("weight", "bias"):
                rules.append(
                    Rule(
                        (f"{old}.cross_attn.kv_linear.{p}",),
                        tuple(f"{new}.attn2.to_{part}.{p}" for part in ("k", "v")),
                        Split((cross_size,) * 2),
                    )
                )
            modules.extend(
                [
                    (old + ".cross_attn.q_linear", new + ".attn2.to_q"),
                    (old + ".cross_attn.proj", new + ".attn2.to_out.0"),
                ]
            )
            if config["norm_elementwise_affine"]:
                modules.append((old + ".norm2", new + ".norm2"))
        if config.get("qk_norm") not in (None, "l2"):
            norm_parameters = (
                ("weight", "bias")
                if config["qk_norm"] in ("layer_norm", "fp32_layer_norm", "layer_norm_across_heads")
                else ("weight",)
            )
            for source, target in [("attn", "attn1")] + ([("cross_attn", "attn2")] if has_cross else []):
                mapping.update(
                    {
                        f"{old}.{source}.{part}_norm.{p}": f"{new}.{target}.norm_{part}.{p}"
                        for part in ("q", "k")
                        for p in norm_parameters
                    }
                )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
