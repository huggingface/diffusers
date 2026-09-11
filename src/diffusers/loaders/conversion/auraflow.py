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
from .transforms import ReorderChunks


def auraflow_conversion(config):
    mapping = {"register_tokens": "register_tokens", "positional_encoding": "pos_embed.pos_embed"}
    modules = [("cond_seq_linear", "context_embedder"), ("final_linear", "proj_out")]
    for old, new in (
        ("t_embedder.mlp.0", "time_step_proj.linear_1"),
        ("t_embedder.mlp.2", "time_step_proj.linear_2"),
        ("init_x_linear", "pos_embed.proj"),
    ):
        mapping.update({f"{old}.{p}": f"{new}.{p}" for p in ("weight", "bias")})
    for i in range(config["num_mmdit_layers"]):
        old, new = f"double_layers.{i}", f"joint_transformer_blocks.{i}"
        for source, target in (("mlpX", "ff"), ("mlpC", "ff_context")):
            modules.extend(
                (f"{old}.{source}.{a}", f"{new}.{target}.{b}")
                for a, b in (("c_fc1", "linear_1"), ("c_fc2", "linear_2"), ("c_proj", "out_projection"))
            )
        modules.extend(
            (f"{old}.{a}.1", f"{new}.{b}.linear") for a, b in (("modX", "norm1"), ("modC", "norm1_context"))
        )
        modules.extend(
            (f"{old}.attn.{a}", f"{new}.attn.{b}")
            for a, b in (
                ("w2q", "to_q"),
                ("w2k", "to_k"),
                ("w2v", "to_v"),
                ("w2o", "to_out.0"),
                ("w1q", "add_q_proj"),
                ("w1k", "add_k_proj"),
                ("w1v", "add_v_proj"),
                ("w1o", "to_add_out"),
            )
        )
    for i in range(config["num_single_dit_layers"]):
        old, new = f"single_layers.{i}", f"single_transformer_blocks.{i}"
        modules.extend(
            (f"{old}.mlp.{a}", f"{new}.ff.{b}")
            for a, b in (("c_fc1", "linear_1"), ("c_fc2", "linear_2"), ("c_proj", "out_projection"))
        )
        modules.append((old + ".modCX.1", new + ".norm1.linear"))
        modules.extend(
            (f"{old}.attn.w1{a}", f"{new}.attn.{b}")
            for a, b in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out.0"))
        )
    mapping.update({old + ".weight": new + ".weight" for old, new in modules})
    return Conversion(
        mapping=mapping, rules=(Rule(("modF.1.weight",), ("norm_out.linear.weight",), ReorderChunks((1, 0))),)
    )
