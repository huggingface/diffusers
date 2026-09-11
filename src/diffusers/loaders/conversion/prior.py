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


import torch

from .core import Conversion, Rule
from .transforms import Chain, Permute, Reshape, Split, WithConstants


def prior_conversion(config):
    original_format = config.get("original_format", "shap_e" if config["encoder_hid_proj_type"] is None else "unclip")
    if original_format not in ("shap_e", "unclip", "kandinsky"):
        raise ValueError("Prior original_format must be 'unclip', 'kandinsky', or 'shap_e'.")
    shape = original_format == "shap_e"
    prefix = "wrapped" if shape else "model"
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    heads, dim = config["num_attention_heads"], config["attention_head_dim"]
    clip_dim = config["clip_embed_dim"] or config["embedding_dim"]
    mapping, rules = {}, []
    modules = [
        (f"{prefix}.time_embed.{a}", f"time_embedding.linear_{b}")
        for a, b in (("c_fc" if shape else "0", 1), ("c_proj" if shape else "2", 2))
    ]
    modules.extend(
        [
            (prefix + (".input_proj" if shape else ".clip_img_proj"), "proj_in"),
            (prefix + (".ln_post" if shape else ".final_ln"), "norm_out"),
            (prefix + (".output_proj" if shape else ".out_proj"), "proj_to_clip_embeddings"),
        ]
    )
    projection = ".clip_embed.1" if config["embedding_proj_norm_type"] else ".clip_embed"
    modules.append((prefix + (projection if shape else ".text_emb_proj"), "embedding_proj"))
    if config["embedding_proj_norm_type"]:
        modules.append((prefix + ".clip_embed.0", "embedding_proj_norm"))
    if config["encoder_hid_proj_type"]:
        modules.append((prefix + ".text_enc_proj", "encoder_hidden_states_proj"))
    if config["norm_in_type"]:
        modules.append((prefix + ".ln_pre", "norm_in"))
    if config["added_emb_type"]:
        mapping[prefix + ".prd_emb"] = "prd_embedding"
    if shape:
        tokens = config["num_embeddings"] + config["additional_embeddings"]
        rules.append(
            Rule((prefix + ".pos_emb",), ("positional_embedding",), Reshape((tokens, hidden), (1, tokens, hidden)))
        )
    else:
        mapping[prefix + ".positional_embedding"] = "positional_embedding"
        rules.extend(
            Rule((f"clip_stats.{name}",), (f"clip_{name}",), Reshape((clip_dim,), (1, clip_dim)))
            for name in ("mean", "std")
        )
    for i in range(config["num_layers"]):
        old = f"{prefix}.{'backbone' if shape else 'transformer'}.resblocks.{i}"
        new = f"transformer_blocks.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("ln_1", "norm1"),
                ("ln_2", "norm3"),
                ("attn.c_proj", "attn1.to_out.0"),
                ("mlp.c_fc", "ff.net.0.proj"),
                ("mlp.c_proj", "ff.net.2"),
            )
        )
        for p in ("weight", "bias"):
            trailing = (hidden,) if p == "weight" else ()
            transform = Chain(
                (
                    Reshape((3 * hidden,) + trailing, (heads, 3, dim) + trailing),
                    Permute((1, 0, 2, 3) if trailing else (1, 0, 2)),
                    Reshape((3, heads, dim) + trailing, (3 * hidden,) + trailing),
                    Split((hidden,) * 3),
                )
            )
            rules.append(
                Rule(
                    (f"{old}.attn.c_qkv.{p}",),
                    tuple(f"{new}.attn1.to_{part}.{p}" for part in ("q", "k", "v")),
                    transform,
                )
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    if shape:
        anchor = prefix + ".input_proj.weight"
        del mapping[anchor]
        rules.append(
            Rule(
                (anchor,),
                ("proj_in.weight", "clip_mean", "clip_std"),
                WithConstants((torch.zeros(1, clip_dim), torch.zeros(1, clip_dim))),
            )
        )
    return Conversion(mapping=mapping, rules=tuple(rules))
