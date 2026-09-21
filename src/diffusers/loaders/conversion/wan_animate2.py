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


def wan_animate2_conversion(config):
    mapping = {"head.modulation": "head.modulation"}
    modules = [
        (name, name)
        for name in (
            "patch_embedding",
            "head.head",
            "time_projection.1",
            "time_embedding.0",
            "time_embedding.2",
            "text_embedding.0",
            "text_embedding.2",
        )
    ]
    if config["use_img_emb"]:
        modules.extend((f"img_emb.proj.{i}", f"img_emb.proj.{i}") for i in (0, 1, 3, 4))
    for i in range(config["num_layers"]):
        old, new = f"blocks.{i}.block", f"blocks.{i}"
        mapping[old + ".modulation"] = new + ".modulation"
        modules.extend((f"{old}.ffn.{j}", f"{new}.ffn.{j}") for j in (0, 2))
        if config["cross_attn_norm"]:
            modules.append((old + ".norm3", new + ".norm3"))
        for attn in ("self_attn", "cross_attn"):
            modules.extend(
                (f"{old}.{attn}.{a}", f"{new}.{attn}.{b}")
                for a, b in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out.0"))
            )
            mapping.update(
                {f"{old}.{attn}.norm_{part}.weight": f"{new}.{attn}.norm_{part}.weight" for part in ("q", "k")}
            )
            if attn == "cross_attn" and config["use_img_emb"]:
                modules.extend((f"{old}.{attn}.{part}_img", f"{new}.{attn}.add_{part}_proj") for part in ("k", "v"))
                mapping[f"{old}.{attn}.norm_k_img.weight"] = f"{new}.{attn}.norm_added_k.weight"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
