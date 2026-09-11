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


from .core import Conversion


def cosmos_conversion(config):
    version = config.get("original_format", "cosmos2")
    if version not in ("cosmos1", "cosmos2"):
        raise ValueError("original_format must be 'cosmos1' or 'cosmos2'.")
    first = version == "cosmos1"
    modules = [
        ("x_embedder.proj.1", "patch_embed.proj"),
        ("t_embedder.1.linear_1", "time_embed.t_embedder.linear_1"),
        ("t_embedder.1.linear_2", "time_embed.t_embedder.linear_2"),
        ("affline_norm" if first else "t_embedding_norm", "time_embed.norm"),
        ("final_layer.linear", "proj_out"),
    ]
    for j in (1, 2):
        source = f"final_layer.{'adaLN_modulation' if first else 'adaln_modulation'}.{j}"
        modules.append((source, f"norm_out.linear_{j}"))
    mapping = {}
    if config["extra_pos_embed_type"] == "learnable":
        for axis in ("t", "h", "w"):
            source = "extra_pos_embedder" if first else "learnable_pos_embed"
            mapping[f"{source}.pos_emb_{axis}"] = f"learnable_pos_embed.pos_emb_{axis}"
    for i in range(config["num_layers"]):
        old = f"blocks.block{i}" if first else f"blocks.{i}"
        new = f"transformer_blocks.{i}"
        for j, name in enumerate(("self_attn", "cross_attn", "mlp")):
            old_norm = f"{old}.blocks.{j}.adaLN_modulation" if first else f"{old}.adaln_modulation_{name}"
            for layer in (1, 2):
                modules.append((f"{old_norm}.{layer}", f"{new}.norm{j + 1}.linear_{layer}"))
        for j, name in enumerate(("self_attn", "cross_attn")):
            source = f"{old}.blocks.{j}.block.attn" if first else f"{old}.{name}"
            target = f"{new}.attn{j + 1}"
            pairs = (
                (
                    ("to_q.0", "to_q"),
                    ("to_q.1", "norm_q"),
                    ("to_k.0", "to_k"),
                    ("to_k.1", "norm_k"),
                    ("to_v.0", "to_v"),
                    ("to_out.0", "to_out.0"),
                )
                if first
                else (
                    ("q_proj", "to_q"),
                    ("q_norm", "norm_q"),
                    ("k_proj", "to_k"),
                    ("k_norm", "norm_k"),
                    ("v_proj", "to_v"),
                    ("output_proj", "to_out.0"),
                )
            )
            modules.extend((f"{source}.{a}", f"{target}.{b}") for a, b in pairs)
            if j == 1 and config["img_context_dim_in"]:
                modules.extend(
                    (f"{source}.{name}", f"{target}.{name}")
                    for name in ("q_img", "k_img", "v_img", "q_img_norm", "k_img_norm")
                )
        source = f"{old}.blocks.2.block" if first else f"{old}.mlp"
        modules.extend((f"{source}.layer{j}", f"{new}.ff.{name}") for j, name in ((1, "net.0.proj"), (2, "net.2")))
    mapping.update({old + ".weight": new + ".weight" for old, new in modules})
    for name, enabled in (
        ("crossattn_proj.0", config["use_crossattn_projection"]),
        ("img_context_proj.0", config["img_context_dim_in"]),
    ):
        if enabled:
            mapping.update({f"{name}.{p}": f"{name}.{p}" for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
