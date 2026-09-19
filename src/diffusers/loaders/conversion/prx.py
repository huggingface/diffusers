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


def prx_conversion(config):
    modules = [
        (name, name)
        for name in (
            "txt_in",
            "time_in.in_layer",
            "time_in.out_layer",
            "final_layer.linear",
            "final_layer.adaLN_modulation.1",
        )
    ]
    modules.extend(
        (name, name) for name in (("img_in.0", "img_in.1") if config["bottleneck_size"] is not None else ("img_in",))
    )
    if config["resolution_embeds"]:
        modules.extend(
            (f"resolution_embedder.mlp.{name}", f"resolution_embedder.mlp.{name}")
            for name in ("in_layer", "out_layer")
        )
    mapping = {}
    original_format = config.get("original_format", "prx")
    if original_format not in ("prx", "prx_weight_norm"):
        raise ValueError("PRX original_format must be 'prx' or 'prx_weight_norm'.")
    norm_parameter = "weight" if original_format == "prx_weight_norm" else "scale"
    for i in range(config["depth"]):
        prefix = f"blocks.{i}"
        modules.append((prefix + ".modulation.lin", prefix + ".modulation.lin"))
        for old, new in (
            ("img_qkv_proj", "attention.img_qkv_proj"),
            ("txt_kv_proj", "attention.txt_kv_proj"),
            ("attn_out", "attention.to_out.0"),
            ("gate_proj", "gate_proj"),
            ("up_proj", "up_proj"),
            ("down_proj", "down_proj"),
        ):
            mapping[f"{prefix}.{old}.weight"] = f"{prefix}.{new}.weight"
        mapping.update(
            {
                f"{prefix}.{old}.{norm_parameter}": f"{prefix}.attention.{new}.weight"
                for old, new in (
                    ("qk_norm.query_norm", "norm_q"),
                    ("qk_norm.key_norm", "norm_k"),
                    ("k_norm", "norm_added_k"),
                )
            }
        )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
