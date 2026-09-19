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


def wan_conversion(config):
    mapping = {"head.modulation": "scale_shift_table"}
    modules = [
        ("patch_embedding", "patch_embedding"),
        ("head.head", "proj_out"),
        ("time_projection.1", "condition_embedder.time_proj"),
    ]
    for source, target in (("time_embedding", "time_embedder"), ("text_embedding", "text_embedder")):
        modules.extend((f"{source}.{i}", f"condition_embedder.{target}.linear_{j}") for i, j in ((0, 1), (2, 2)))
    if config["image_dim"] is not None:
        modules.extend(
            (f"img_emb.proj.{i}", f"condition_embedder.image_embedder.{name}")
            for i, name in ((0, "norm1"), (1, "ff.net.0.proj"), (3, "ff.net.2"), (4, "norm2"))
        )
        if config["pos_embed_seq_len"] is not None:
            mapping["img_emb.emb_pos"] = "condition_embedder.image_embedder.pos_embed"
    for i in range(config["num_layers"]):
        prefix = f"blocks.{i}"
        mapping[prefix + ".modulation"] = prefix + ".scale_shift_table"
        modules.extend((f"{prefix}.ffn.{j}", f"{prefix}.ffn.{name}") for j, name in ((0, "net.0.proj"), (2, "net.2")))
        if config["cross_attn_norm"]:
            modules.append((prefix + ".norm3", prefix + ".norm2"))
        for source, target in (("self_attn", "attn1"), ("cross_attn", "attn2")):
            modules.extend(
                (f"{prefix}.{source}.{a}", f"{prefix}.{target}.{b}")
                for a, b in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out.0"))
            )
            if config["qk_norm"] is not None:
                mapping.update(
                    {
                        f"{prefix}.{source}.norm_{part}.weight": f"{prefix}.{target}.norm_{part}.weight"
                        for part in ("q", "k")
                    }
                )
            if source == "cross_attn" and config["added_kv_proj_dim"] is not None:
                modules.extend(
                    (f"{prefix}.{source}.{part}_img", f"{prefix}.{target}.add_{part}_proj") for part in ("k", "v")
                )
                mapping[f"{prefix}.{source}.norm_k_img.weight"] = f"{prefix}.{target}.norm_added_k.weight"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
