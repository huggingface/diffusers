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
from .transforms import Split


def z_image_conversion(config):
    mapping = {
        "x_pad_token": "x_pad_token",
        "cap_pad_token": "cap_pad_token",
        "cap_embedder.0.weight": "cap_embedder.0.weight",
    }
    modules = [
        ("t_embedder.mlp.0", "t_embedder.mlp.0"),
        ("t_embedder.mlp.2", "t_embedder.mlp.2"),
        ("cap_embedder.1", "cap_embedder.1"),
    ]
    patches = list(zip(config["all_patch_size"], config["all_f_patch_size"]))
    if patches != [(2, 1)]:
        raise ValueError("The original Z-Image single-patch checkpoint format requires patch sizes (2, 1).")
    modules.extend(
        [
            ("x_embedder", "all_x_embedder.2-1"),
            ("final_layer.linear", "all_final_layer.2-1.linear"),
            ("final_layer.adaLN_modulation.1", "all_final_layer.2-1.adaLN_modulation.1"),
        ]
    )
    groups = [
        ("noise_refiner", config["n_refiner_layers"]),
        ("context_refiner", config["n_refiner_layers"]),
        ("layers", config["n_layers"]),
    ]
    if config["siglip_feat_dim"] is not None:
        mapping.update(
            {"siglip_pad_token": "siglip_pad_token", "siglip_embedder.0.weight": "siglip_embedder.0.weight"}
        )
        modules.append(("siglip_embedder.1", "siglip_embedder.1"))
        groups.append(("siglip_refiner", config["n_refiner_layers"]))
    rules = []
    for group, count in groups:
        for i in range(count):
            prefix = f"{group}.{i}"
            for name in (
                "feed_forward.w1",
                "feed_forward.w2",
                "feed_forward.w3",
                "attention_norm1",
                "attention_norm2",
                "ffn_norm1",
                "ffn_norm2",
            ):
                mapping[f"{prefix}.{name}.weight"] = f"{prefix}.{name}.weight"
            mapping[prefix + ".attention.out.weight"] = prefix + ".attention.to_out.0.weight"
            if config["qk_norm"]:
                for part in ("q", "k"):
                    mapping[f"{prefix}.attention.{part}_norm.weight"] = f"{prefix}.attention.norm_{part}.weight"
            rules.append(
                Rule(
                    (prefix + ".attention.qkv.weight",),
                    tuple(f"{prefix}.attention.to_{part}.weight" for part in ("q", "k", "v")),
                    Split((config["dim"],) * 3),
                )
            )
            if group in ("noise_refiner", "layers"):
                modules.append((prefix + ".adaLN_modulation.0", prefix + ".adaLN_modulation.0"))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
