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

"""Reversible definitions for encoders and small components embedded in legacy pipeline converters."""

from .clip_vision import clip_vision_conversion
from .core import Conversion, Rule
from .transforms import Reshape, Split


def ldm_bert_conversion(config):
    mapping = {
        "transformer.token_emb.weight": "model.embed_tokens.weight",
        "transformer.pos_emb.emb.weight": "model.embed_positions.weight",
    }
    modules = [("transformer.norm", "model.layer_norm"), ("transformer.to_logits", "to_logits")]
    for i in range(config["encoder_layers"]):
        source, target = "transformer.attn_layers.layers", f"model.layers.{i}"
        modules.extend(
            [
                (f"{source}.{2 * i}.0", target + ".self_attn_layer_norm"),
                (f"{source}.{2 * i + 1}.0", target + ".final_layer_norm"),
                (f"{source}.{2 * i}.1.to_out", target + ".self_attn.out_proj"),
                (f"{source}.{2 * i + 1}.1.net.0.0", target + ".fc1"),
                (f"{source}.{2 * i + 1}.1.net.2", target + ".fc2"),
            ]
        )
        for part in ("q", "k", "v"):
            mapping[f"{source}.{2 * i}.1.to_{part}.weight"] = f"{target}.self_attn.{part}_proj.weight"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)


def paint_by_example_conversion(config):
    vision = clip_vision_conversion({**config, "_class_name": "CLIPVisionModel", "original_format": "clip"})
    mapping = {"cond_stage_model.transformer." + old: "model." + new for old, new in vision.mapping.items()}
    mapping["learnable_vector"] = "uncond_vector"
    modules = [("cond_stage_model.final_ln", "final_layer_norm"), ("proj_out", "proj_out")]
    rules = []
    for i in range((config["num_hidden_layers"] + 1) // 5):
        old, new = f"cond_stage_model.mapper.resblocks.{i}", f"mapper.blocks.{i}"
        modules.extend(
            (old + "." + a, new + "." + b)
            for a, b in (
                ("attn.c_proj", "attn1.to_out.0"),
                ("ln_1", "norm1"),
                ("ln_2", "norm3"),
                ("mlp.c_fc", "ff.net.0.proj"),
                ("mlp.c_proj", "ff.net.2"),
            )
        )
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (old + ".attn.c_qkv." + p,),
                    tuple(new + f".attn1.to_{part}." + p for part in ("q", "k", "v")),
                    Split((config["hidden_size"],) * 3),
                )
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)


def if_safety_checker_conversion(config):
    cfg = {**config["vision_config"], "_class_name": "CLIPVisionModelWithProjection", "original_format": "clip"}
    base = clip_vision_conversion(cfg)
    mapping = {"vision_model." + old: "vision_model." + new for old, new in base.mapping.items()}
    rules = []
    for name in ("p_head", "w_head"):
        rules.append(
            Rule(
                (name + ".weights",),
                (name + ".weight",),
                Reshape((cfg["projection_dim"],), (1, cfg["projection_dim"])),
            )
        )
        rules.append(Rule((name + ".biases",), (name + ".bias",), Reshape((), (1,))))
    return Conversion(mapping=mapping, rules=rules)


def learned_classifier_free_conversion(config):
    return Conversion(mapping={"transformer.empty_text_embed": "embeddings"})
