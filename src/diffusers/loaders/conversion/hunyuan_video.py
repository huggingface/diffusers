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


def hunyuan_video_conversion(config):
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    modules = [
        ("img_in.proj", "x_embedder.proj"),
        ("txt_in.input_embedder", "context_embedder.proj_in"),
        ("final_layer.linear", "proj_out"),
    ]
    mapping, rules = {}, []
    for old, new in (
        ("time_in.mlp.0", "timestep_embedder.linear_1"),
        ("time_in.mlp.2", "timestep_embedder.linear_2"),
        ("vector_in.in_layer", "text_embedder.linear_1"),
        ("vector_in.out_layer", "text_embedder.linear_2"),
    ):
        modules.append((old, "time_text_embed." + new))
    if config["guidance_embeds"]:
        modules.extend(
            (f"guidance_in.mlp.{i}", f"time_text_embed.guidance_embedder.linear_{j}") for i, j in ((0, 1), (2, 2))
        )
    for old, new in (
        ("t_embedder.mlp.0", "timestep_embedder.linear_1"),
        ("t_embedder.mlp.2", "timestep_embedder.linear_2"),
        ("c_embedder.linear_1", "text_embedder.linear_1"),
        ("c_embedder.linear_2", "text_embedder.linear_2"),
    ):
        modules.append(("txt_in." + old, "context_embedder.time_text_embed." + new))
    for i in range(config["num_refiner_layers"]):
        old, new = f"txt_in.individual_token_refiner.blocks.{i}", f"context_embedder.token_refiner.refiner_blocks.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("norm1", "norm1"),
                ("norm2", "norm2"),
                ("self_attn_proj", "attn.to_out.0"),
                ("mlp.fc1", "ff.net.0.proj"),
                ("mlp.fc2", "ff.net.2"),
                ("adaLN_modulation.1", "norm_out.linear"),
            )
        )
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (f"{old}.self_attn_qkv.{p}",),
                    tuple(f"{new}.attn.to_{part}.{p}" for part in ("q", "k", "v")),
                    Split((hidden,) * 3),
                )
            )
    for i in range(config["num_layers"]):
        old, new = f"double_blocks.{i}", f"transformer_blocks.{i}"
        for source, norm, ff, projections, output, norms in (
            ("img", "norm1", "ff", ("to_q", "to_k", "to_v"), "to_out.0", ("norm_q", "norm_k")),
            (
                "txt",
                "norm1_context",
                "ff_context",
                ("add_q_proj", "add_k_proj", "add_v_proj"),
                "to_add_out",
                ("norm_added_q", "norm_added_k"),
            ),
        ):
            modules.extend(
                [
                    (f"{old}.{source}_mod.linear", f"{new}.{norm}.linear"),
                    (f"{old}.{source}_mlp.fc1", f"{new}.{ff}.net.0.proj"),
                    (f"{old}.{source}_mlp.fc2", f"{new}.{ff}.net.2"),
                    (f"{old}.{source}_attn_proj", f"{new}.attn.{output}"),
                ]
            )
            for p in ("weight", "bias"):
                rules.append(
                    Rule(
                        (f"{old}.{source}_attn_qkv.{p}",),
                        tuple(f"{new}.attn.{part}.{p}" for part in projections),
                        Split((hidden,) * 3),
                    )
                )
            if config["qk_norm"] is not None:
                for part, target in zip(("q", "k"), norms):
                    mapping[f"{old}.{source}_attn_{part}_norm.weight"] = f"{new}.attn.{target}.weight"
    for i in range(config["num_single_layers"]):
        old, new = f"single_blocks.{i}", f"single_transformer_blocks.{i}"
        modules.extend([(old + ".modulation.linear", new + ".norm.linear"), (old + ".linear2", new + ".proj_out")])
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (f"{old}.linear1.{p}",),
                    tuple(f"{new}.{part}.{p}" for part in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp")),
                    Split((hidden,) * 3 + (int(hidden * config["mlp_ratio"]),)),
                )
            )
        if config["qk_norm"] is not None:
            for part in ("q", "k"):
                mapping[f"{old}.{part}_norm.weight"] = f"{new}.attn.norm_{part}.weight"
    for p in ("weight", "bias"):
        rules.append(Rule((f"final_layer.adaLN_modulation.1.{p}",), (f"norm_out.linear.{p}",), ReorderChunks((1, 0))))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
