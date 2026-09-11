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
from .transforms import ReorderChunks


def stable_audio3_conversion(config):
    mapping = {
        "conditioner.conditioners.prompt.padding_embedding": "prompt_padding_embedding",
        "model.model.transformer.rotary_pos_emb.inv_freq": "rotary_pos_emb.inv_freq",
    }
    modules = []
    for name in ("to_timestep_embed.0", "to_timestep_embed.2"):
        modules.append(("model.model." + name, name))
    for name in (
        "to_cond_embed.0",
        "to_cond_embed.2",
        "to_global_embed.0",
        "to_global_embed.2",
        "preprocess_conv",
        "postprocess_conv",
    ):
        mapping[f"model.model.{name}.weight"] = name + ".weight"
    for i in (0, 2):
        modules.append((f"model.model.transformer.global_cond_embedder.{i}", f"global_cond_embedder.{i}"))
    for name in ("in", "out"):
        mapping[f"model.model.transformer.project_{name}.weight"] = f"proj_{name}.weight"
    if config["num_memory_tokens"] > 0:
        mapping["model.model.transformer.memory_tokens"] = "memory_tokens"
    rules = []
    for i in range(config["depth"]):
        old, new = f"model.model.transformer.layers.{i}", f"transformer_blocks.{i}"
        mapping[old + ".to_scale_shift_gate"] = new + ".to_scale_shift_gate"
        for name in (
            "pre_norm",
            "cross_attend_norm",
            "ff_norm",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "cross_attn.q_norm",
            "cross_attn.k_norm",
        ):
            mapping[f"{old}.{name}.gamma"] = f"{new}.{name}.gamma"
        for name in ("self_attn.to_out", "cross_attn.to_q", "cross_attn.to_kv", "cross_attn.to_out"):
            mapping[f"{old}.{name}.weight"] = f"{new}.{name}.weight"
        source, target = old + ".self_attn.to_qkv.weight", new + ".self_attn.to_qkv.weight"
        if config["use_differential_attention"]:
            rules.append(Rule((source,), (target,), ReorderChunks((0, 3, 1, 4, 2))))
        else:
            mapping[source] = target
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("ff.ff.0.proj", "ff.proj_in"),
                ("ff.ff.2", "ff.proj_out"),
                ("to_local_embed.0", "to_local_embed.0"),
                ("to_local_embed.2", "to_local_embed.2"),
            )
        )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
