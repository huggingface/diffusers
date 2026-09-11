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
from .transforms import Permute, Split, WithConstants


def clip_conversion(config):
    original_format = config.get(
        "original_format", "openclip" if config["_class_name"] == "CLIPTextModelWithProjection" else "clip"
    )
    if original_format == "openclip":
        return openclip_conversion(config)
    if original_format != "clip":
        raise ValueError("CLIP original_format must be 'clip' or 'openclip'.")
    modules = ["text_model.final_layer_norm"]
    keys = ["text_model.embeddings.token_embedding.weight", "text_model.embeddings.position_embedding.weight"]
    for i in range(config["num_hidden_layers"]):
        prefix = f"text_model.encoder.layers.{i}"
        modules.extend(
            f"{prefix}.{name}"
            for name in (
                "layer_norm1",
                "layer_norm2",
                "mlp.fc1",
                "mlp.fc2",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
            )
        )
    if config["_class_name"] == "CLIPTextModelWithProjection":
        keys.append("text_projection.weight")
    if config["_class_name"] == "ContextCLIPTextModel":
        keys.append("text_model.embeddings.position_ids")
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    prefix = "" if _uses_flat_text_model(config) else "text_model."
    return Conversion(
        mapping={
            key: prefix + key.removeprefix("text_model.") if key.startswith("text_model.") else key for key in keys
        }
    )


def openclip_conversion(config):
    prefix = "" if _uses_flat_text_model(config) else "text_model."
    mapping = {
        "token_embedding.weight": prefix + "embeddings.token_embedding.weight",
        "positional_embedding": prefix + "embeddings.position_embedding.weight",
    }
    modules, rules = [("ln_final", prefix + "final_layer_norm")], []
    for i in range(config["num_hidden_layers"]):
        old, new = f"transformer.resblocks.{i}", f"{prefix}encoder.layers.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("ln_1", "layer_norm1"),
                ("ln_2", "layer_norm2"),
                ("mlp.c_fc", "mlp.fc1"),
                ("mlp.c_proj", "mlp.fc2"),
                ("attn.out_proj", "self_attn.out_proj"),
            )
        )
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (f"{old}.attn.in_proj_{p}",),
                    tuple(f"{new}.self_attn.{part}_proj.{p}" for part in ("q", "k", "v")),
                    Split((config["hidden_size"],) * 3),
                )
            )
    if config["_class_name"] == "CLIPTextModelWithProjection":
        rules.append(Rule(("text_projection",), ("text_projection.weight",), Permute((1, 0))))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    if config["_class_name"] == "ContextCLIPTextModel":
        target = mapping.pop("token_embedding.weight")
        rules.append(
            Rule(
                ("token_embedding.weight",),
                (target, prefix + "embeddings.position_ids"),
                WithConstants((torch.arange(config["max_position_embeddings"]).unsqueeze(0),)),
            )
        )
    return Conversion(mapping=mapping, rules=tuple(rules))


def _uses_flat_text_model(config):
    version = (config.get("transformers_version") or "4").split(".", 1)[0]
    return config["_class_name"] == "CLIPTextModel" and int(version) >= 5
