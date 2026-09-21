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


def flux_conversion(config):
    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    parameters = ("weight", "bias")
    modules = [("txt_in", "context_embedder"), ("img_in", "x_embedder"), ("final_layer.linear", "proj_out")]
    mapping, rules = ({}, [])
    embedding = "time_text_embed"
    embeddings = [("time_in", "timestep_embedder")]
    embeddings.append(("vector_in", "text_embedder"))
    if config.get("guidance_embeds", False):
        embeddings.append(("guidance_in", "guidance_embedder"))
    for old, new in embeddings:
        modules.extend(
            [(old + ".in_layer", f"{embedding}.{new}.linear_1"), (old + ".out_layer", f"{embedding}.{new}.linear_2")]
        )
    for parameter in parameters:
        rules.append(
            Rule(
                (f"final_layer.adaLN_modulation.1.{parameter}",),
                (f"norm_out.linear.{parameter}",),
                ReorderChunks((1, 0)),
            )
        )
    for i in range(config["num_layers"]):
        old, new = (f"double_blocks.{i}", f"transformer_blocks.{i}")
        modules.extend(
            [(old + ".img_mod.lin", new + ".norm1.linear"), (old + ".txt_mod.lin", new + ".norm1_context.linear")]
        )
        for modality, attention, mlp, norms in (
            ("img", ("to_q", "to_k", "to_v"), "ff", ("norm_q", "norm_k")),
            ("txt", ("add_q_proj", "add_k_proj", "add_v_proj"), "ff_context", ("norm_added_q", "norm_added_k")),
        ):
            for parameter in parameters:
                rules.append(
                    Rule(
                        (f"{old}.{modality}_attn.qkv.{parameter}",),
                        tuple((f"{new}.attn.{name}.{parameter}" for name in attention)),
                        Split((hidden_size,) * 3),
                    )
                )
            for source, target in zip(("query_norm", "key_norm"), norms):
                mapping[f"{old}.{modality}_attn.norm.{source}.scale"] = f"{new}.attn.{target}.weight"
            modules.extend(
                [
                    (f"{old}.{modality}_mlp.0", f"{new}.{mlp}." + "net.0.proj"),
                    (f"{old}.{modality}_mlp.2", f"{new}.{mlp}." + "net.2"),
                    (
                        f"{old}.{modality}_attn.proj",
                        f"{new}.attn." + ("to_out.0" if modality == "img" else "to_add_out"),
                    ),
                ]
            )
    for i in range(config["num_single_layers"]):
        old, new = (f"single_blocks.{i}", f"single_transformer_blocks.{i}")
        modules.append((old + ".modulation.lin", new + ".norm.linear"))
        for parameter in parameters:
            rules.append(
                Rule(
                    (f"{old}.linear1.{parameter}",),
                    tuple(
                        (f"{new}.{name}.{parameter}" for name in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"))
                    ),
                    Split((hidden_size,) * 3 + (4 * hidden_size,)),
                )
            )
        modules.append((old + ".linear2", new + ".proj_out"))
        for source, target in (("query_norm", "norm_q"), ("key_norm", "norm_k")):
            mapping[f"{old}.norm.{source}.scale"] = f"{new}.attn.{target}.weight"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in parameters})
    return Conversion(mapping=mapping, rules=tuple(rules))
