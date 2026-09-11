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
from .transforms import MergeEqual, Reverse, Split


def unidiffuser_conversion(config):
    if (
        config["norm_type"] != "layer_norm"
        or config["cross_attention_dim"] is not None
        or config["block_type"] != "unidiffuser"
    ):
        raise ValueError("Original UniDiffuser uses layer_norm, self attention, and unidiffuser blocks.")
    width = config["num_attention_heads"] * config["attention_head_dim"]
    mapping = {"pos_embed": "pos_embed"}
    modules = [
        ("clip_img_embed", "clip_img_in"),
        ("text_embed", "text_in"),
        ("decoder_pred", "vae_img_out"),
        ("norm", "transformer.norm_out"),
        ("clip_img_out", "clip_img_out"),
        ("text_out", "text_out"),
    ]
    rules = [
        Rule(
            (f"patch_embed.proj.{p}",),
            (f"vae_img_in.proj.{p}", f"transformer.pos_embed.proj.{p}"),
            Reverse(MergeEqual(2)),
        )
        for p in ("weight", "bias")
    ]
    if config["use_data_type_embedding"]:
        mapping.update(
            {
                "pos_embed_token": "data_type_pos_embed_token",
                "token_embedding.weight": "data_type_token_embedding.weight",
            }
        )
    if config["use_timestep_embedding"]:
        modules.extend(
            (f"time_{modality}_embed.{i}", f"timestep_{modality}_embed.linear_{j}")
            for modality in ("img", "text")
            for i, j in ((0, 1), (2, 2))
        )
    blocks = ["mid_block"] + [f"{side}_blocks.{i}" for side in ("in", "out") for i in range(config["num_layers"] // 2)]
    for old in blocks:
        new = f"transformer.transformer_{old}"
        if old.startswith("out_"):
            modules.extend([(f"{old}.skip_linear", f"{new}.skip.skip_linear"), (f"{old}.norm1", f"{new}.skip.norm")])
            new += ".block"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (("attn.proj", "attn1.to_out.0"), ("mlp.fc1", "ff.net.0.proj"), ("mlp.fc2", "ff.net.2"))
        )
        if config["norm_elementwise_affine"]:
            modules.extend([(f"{old}.norm2", f"{new}.norm1"), (f"{old}.norm3", f"{new}.norm3")])
        for p in ("weight", "bias") if config["attention_bias"] else ("weight",):
            rules.append(
                Rule(
                    (f"{old}.attn.qkv.{p}",),
                    tuple(f"{new}.attn1.to_{part}.{p}" for part in ("q", "k", "v")),
                    Split((width,) * 3),
                )
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)
