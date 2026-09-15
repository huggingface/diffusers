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


def joy_image_conversion(config):
    names = [
        "img_in",
        "proj_out",
        "condition_embedder.time_embedder.linear_1",
        "condition_embedder.time_embedder.linear_2",
        "condition_embedder.time_proj",
        "condition_embedder.text_embedder.linear_1",
        "condition_embedder.text_embedder.linear_2",
    ]
    mapping = {}
    for i in range(config["num_layers"]):
        prefix = f"double_blocks.{i}"
        for modality in ("img", "txt"):
            mapping[f"{prefix}.{modality}_mod.modulate_table"] = f"{prefix}.{modality}_mod.modulate_table"
            names.extend(f"{prefix}.{modality}_mlp.net.{leaf}" for leaf in ("0.proj", "2"))
            for leaf in ("qkv", "proj"):
                for p in ("weight", "bias"):
                    mapping[f"{prefix}.{modality}_attn_{leaf}.{p}"] = f"{prefix}.attn.{modality}_attn_{leaf}.{p}"
            for part in ("q", "k"):
                mapping[f"{prefix}.{modality}_attn_{part}_norm.weight"] = (
                    f"{prefix}.attn.{modality}_attn_{part}_norm.weight"
                )
    mapping.update({f"{name}.{p}": f"{name}.{p}" for name in names for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
