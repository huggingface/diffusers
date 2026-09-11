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
from .transforms import Reshape, WithConstants


def blip_qformer_conversion(config):
    qformer, vision = config["qformer_config"], config["vision_config"]
    mapping = {
        "blip.query_tokens": "query_tokens",
        "blip.Qformer.bert.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "blip.Qformer.bert.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "blip.visual_encoder.conv1.weight": "visual_encoder.embeddings.patch_embedding.weight",
    }
    modules = [
        ("blip.Qformer.bert.embeddings.LayerNorm", "embeddings.LayerNorm"),
        ("blip.visual_encoder.ln_pre", "visual_encoder.pre_layernorm"),
        ("blip.ln_vision", "visual_encoder.post_layernorm"),
    ]
    modules.extend(("proj_layer." + name, "proj_layer." + name) for name in ("dense1", "dense2", "LayerNorm"))
    for i in range(qformer["num_hidden_layers"]):
        old, new = f"blip.Qformer.bert.encoder.layer.{i}", f"encoder.layer.{i}"
        attentions = ["attention"]
        if i % qformer["cross_attention_frequency"] == 0:
            attentions.append("crossattention")
        for attn in attentions:
            modules.extend(
                (f"{old}.{attn}.self.{part}", f"{new}.{attn}.attention.{part}") for part in ("query", "key", "value")
            )
            modules.extend(
                (f"{old}.{attn}.output.{name}", f"{new}.{attn}.output.{name}") for name in ("dense", "LayerNorm")
            )
        modules.extend(
            (f"{old}.{name}", f"{new}.{name}")
            for name in (
                "intermediate.dense",
                "intermediate_query.dense",
                "output.dense",
                "output.LayerNorm",
                "output_query.dense",
                "output_query.LayerNorm",
            )
        )
    for i in range(vision["num_hidden_layers"]):
        old, new = f"blip.visual_encoder.transformer.resblocks.{i}", f"visual_encoder.encoder.layers.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("ln_1", "layer_norm1"),
                ("ln_2", "layer_norm2"),
                ("attn.out_proj", "self_attn.projection"),
                ("mlp.c_fc", "mlp.fc1"),
                ("mlp.c_proj", "mlp.fc2"),
            )
        )
        mapping.update({f"{old}.attn.in_proj_{p}": f"{new}.self_attn.qkv.{p}" for p in ("weight", "bias")})
    hidden = vision["hidden_size"]
    positions = (vision["image_size"] // vision["patch_size"]) ** 2 + 1
    rules = (
        Rule(
            ("blip.visual_encoder.class_embedding",),
            ("visual_encoder.embeddings.class_embedding",),
            Reshape((hidden,), (1, 1, hidden)),
        ),
        Rule(
            ("blip.visual_encoder.positional_embedding",),
            ("visual_encoder.embeddings.position_embedding",),
            Reshape((positions, hidden), (1, positions, hidden)),
        ),
    )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    position_ids = torch.arange(qformer["max_position_embeddings"], device="cpu").expand((1, -1))
    source = "blip.Qformer.bert.embeddings.word_embeddings.weight"
    target = mapping.pop(source)
    rules += (Rule((source,), (target, "embeddings.position_ids"), WithConstants((position_ids,))),)
    return Conversion(mapping=mapping, rules=rules)
