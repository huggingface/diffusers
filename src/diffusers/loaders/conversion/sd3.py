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


def sd3_conversion(config):
    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    mapping = {"pos_embed": "pos_embed.pos_embed"}
    modules = [
        ("x_embedder.proj", "pos_embed.proj"),
        ("context_embedder", "context_embedder"),
        ("final_layer.linear", "proj_out"),
    ]
    rules = []
    for old, new in (("t_embedder", "timestep_embedder"), ("y_embedder", "text_embedder")):
        modules.extend(
            [
                (old + ".mlp.0", "time_text_embed." + new + ".linear_1"),
                (old + ".mlp.2", "time_text_embed." + new + ".linear_2"),
            ]
        )
    for i in range(config["num_layers"]):
        old, new = f"joint_blocks.{i}", f"transformer_blocks.{i}"
        branches = [
            ("x_block.attn", "attn", ("to_q", "to_k", "to_v"), "to_out.0", ("norm_q", "norm_k")),
            (
                "context_block.attn",
                "attn",
                ("add_q_proj", "add_k_proj", "add_v_proj"),
                "to_add_out",
                ("norm_added_q", "norm_added_k"),
            ),
        ]
        if i in config.get("dual_attention_layers", ()):
            branches.append(("x_block.attn2", "attn2", ("to_q", "to_k", "to_v"), "to_out.0", ("norm_q", "norm_k")))
        for branch, attn, targets, output, norms in branches:
            for parameter in ("weight", "bias"):
                rules.append(
                    Rule(
                        (f"{old}.{branch}.qkv.{parameter}",),
                        tuple(f"{new}.{attn}.{name}.{parameter}" for name in targets),
                        Split((hidden_size,) * 3),
                    )
                )
            if branch != "context_block.attn" or i != config["num_layers"] - 1:
                modules.append((f"{old}.{branch}.proj", f"{new}.{attn}.{output}"))
            if config.get("qk_norm") is not None:
                norm_parameters = (
                    ("weight", "bias") if config["qk_norm"] in ("layer_norm", "fp32_layer_norm") else ("weight",)
                )
                for source, target in zip(("ln_q", "ln_k"), norms):
                    mapping.update(
                        {f"{old}.{branch}.{source}.{p}": f"{new}.{attn}.{target}.{p}" for p in norm_parameters}
                    )
        for branch, norm, ff in (("x_block", "norm1", "ff"), ("context_block", "norm1_context", "ff_context")):
            if branch == "context_block" and i == config["num_layers"] - 1:
                for p in ("weight", "bias"):
                    rules.append(
                        Rule(
                            (f"{old}.{branch}.adaLN_modulation.1.{p}",),
                            (f"{new}.{norm}.linear.{p}",),
                            ReorderChunks((1, 0)),
                        )
                    )
            else:
                modules.extend(
                    [
                        (f"{old}.{branch}.adaLN_modulation.1", f"{new}.{norm}.linear"),
                        (f"{old}.{branch}.mlp.fc1", f"{new}.{ff}.net.0.proj"),
                        (f"{old}.{branch}.mlp.fc2", f"{new}.{ff}.net.2"),
                    ]
                )
    for p in ("weight", "bias"):
        rules.append(Rule((f"final_layer.adaLN_modulation.1.{p}",), (f"norm_out.linear.{p}",), ReorderChunks((1, 0))))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
