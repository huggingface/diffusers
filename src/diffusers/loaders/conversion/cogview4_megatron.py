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

from .cogview3plus import cogview3plus_conversion
from .core import Conversion, Rule
from .transforms import Chain, Permute, Reshape


def cogview4_megatron_conversion(config):
    base = cogview3plus_conversion(config)
    replacements = {
        "mixins.patch_embed.proj": "encoder_expand_linear",
        "mixins.patch_embed.text_proj": "text_projector",
        "mixins.final_layer.linear": "output_projector",
        "time_embed.": "time_embedding.time_embed.",
        "label_emb.0.": "label_embedding.label_embed.",
        "mixins.final_layer.adaln.1": "adaln_final",
    }
    for i in range(config["num_layers"]):
        replacements[f"mixins.adaln.adaln_modules.{i}.1"] = f"decoder.layers.{i}.adaln"
        for old, new in (
            ("attention.query_key_value", "self_attention.linear_qkv"),
            ("attention.dense", "self_attention.linear_proj"),
            ("mlp.dense_h_to_4h", "mlp.linear_fc1"),
            ("mlp.dense_4h_to_h", "mlp.linear_fc2"),
        ):
            replacements[f"transformer.layers.{i}.{old}"] = f"decoder.layers.{i}.{new}"

    def rename(key):
        for old, new in replacements.items():
            if key.startswith(old):
                return new + key[len(old) :]
        raise ValueError(f"Unknown CogView Megatron source key {key}.")

    mapping = {rename(old): new for old, new in base.mapping.items()}
    rules = []
    heads, dim = config["num_attention_heads"], config["attention_head_dim"]
    width = heads * dim
    for rule in base.rules:
        transform = rule.transform
        if ".query_key_value." in rule.original[0]:
            trailing = (width,) if rule.original[0].endswith("weight") else ()
            transform = Chain(
                (
                    Reshape((3 * width,) + trailing, (heads, 3, dim) + trailing),
                    Permute((1, 0, 2, 3) if trailing else (1, 0, 2)),
                    Reshape((3, heads, dim) + trailing, (3 * width,) + trailing),
                    transform,
                )
            )
        rules.append(Rule(tuple(rename(key) for key in rule.original), rule.diffusers, transform))
    return Conversion(mapping=mapping, rules=rules)
