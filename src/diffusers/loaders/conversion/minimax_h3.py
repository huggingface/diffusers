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
from .transforms import Chain, Permute, ReorderChunks, Reshape, Split


def minimax_h3_conversion(config):
    mapping = {
        "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
        "final_layer.norm.weight": "norm_out.norm.weight",
    }
    modules = [
        ("video_patch_proj", "proj_in"),
        ("audio_patch_proj", "audio_proj_in"),
        ("condition_proj", "context_embedder"),
        ("time_embedder.proj_in", "time_embedder.linear_1"),
        ("time_embedder.proj_out", "time_embedder.linear_2"),
        ("final_layer.adaln_proj.linear", "norm_out.linear"),
        ("final_layer.video_out", "proj_out"),
        ("final_layer.audio_out", "audio_proj_out"),
    ]
    heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
    inner, hidden = heads * head_dim, config["hidden_size"]
    original_format = config.get("original_format", "minimax_h3")
    if original_format not in ("minimax_h3", "minimax_h3_shards"):
        raise ValueError("MiniMax H3 original_format must be 'minimax_h3' or 'minimax_h3_shards'.")
    split = Split((inner,) * 3)
    if original_format == "minimax_h3_shards":
        split = Chain(
            (
                Reshape((3 * inner, hidden), (heads, 3, head_dim, hidden)),
                Permute((1, 0, 2, 3)),
                Reshape((3, heads, head_dim, hidden), (3 * inner, hidden)),
                split,
            )
        )
    rules = []
    for old_group, new_group, count, modulated in (
        ("blocks", "transformer_blocks", config["num_layers"], True),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", config["num_refiner_layers"], False),
    ):
        for i in range(count):
            old, new = f"{old_group}.{i}", f"{new_group}.{i}"
            mapping.update(
                {
                    f"{old}.{a}.weight": f"{new}.{b}.weight"
                    for a, b in (
                        ("norm1", "norm1"),
                        ("norm2", "norm2"),
                        ("attn.q_norm", "attn.norm_q"),
                        ("attn.k_norm", "attn.norm_k"),
                        ("attn.out_proj", "attn.to_out.0"),
                        ("mlp.fc2", "ff.net.2"),
                    )
                }
            )
            rules.append(
                Rule(
                    (old + ".attn.qkv_proj.weight",),
                    tuple(f"{new}.attn.to_{part}.weight" for part in ("q", "k", "v")),
                    split,
                )
            )
            rules.append(Rule((old + ".mlp.fc1.weight",), (new + ".ff.net.0.proj.weight",), ReorderChunks((1, 0))))
            if modulated:
                modules.append((old + ".adaln_proj.linear", new + ".adaln_proj.linear"))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
