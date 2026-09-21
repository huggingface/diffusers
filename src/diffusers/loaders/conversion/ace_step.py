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


def ace_step_conversion(config):
    mapping = {"scale_shift_table": "scale_shift_table", "norm_out.weight": "norm_out.weight"}
    modules = [
        ("proj_in.1", "proj_in_conv"),
        ("proj_out.1", "proj_out_conv"),
        ("condition_embedder", "condition_embedder"),
    ]
    modules.extend(
        (f"{prefix}.{name}", f"{prefix}.{name}")
        for prefix in ("time_embed", "time_embed_r")
        for name in ("linear_1", "linear_2", "time_proj")
    )
    for i in range(config["num_hidden_layers"]):
        prefix = f"layers.{i}"
        mapping[prefix + ".scale_shift_table"] = prefix + ".scale_shift_table"
        mapping.update(_ace_step_attention_mapping(prefix + ".self_attn", config["attention_bias"]))
        mapping.update(_ace_step_attention_mapping(prefix + ".cross_attn", config["attention_bias"]))
        for name in ("self_attn_norm", "cross_attn_norm", "mlp_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            mapping[f"{prefix}.{name}.weight"] = f"{prefix}.{name}.weight"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)


def _ace_step_attention_mapping(prefix, bias):
    mapping = {}
    for old, new in (("q_proj", "to_q"), ("k_proj", "to_k"), ("v_proj", "to_v"), ("o_proj", "to_out.0")):
        mapping.update(
            {f"{prefix}.{old}.{p}": f"{prefix}.{new}.{p}" for p in (("weight", "bias") if bias else ("weight",))}
        )
    mapping.update({f"{prefix}.{part}_norm.weight": f"{prefix}.norm_{part}.weight" for part in ("q", "k")})
    return mapping


def _ace_step_encoder_mapping(config, count):
    mapping = {
        "embed_tokens.weight": "embed_tokens.weight",
        "embed_tokens.bias": "embed_tokens.bias",
        "norm.weight": "norm.weight",
    }
    for i in range(count):
        prefix = f"layers.{i}"
        mapping.update(_ace_step_attention_mapping(prefix + ".self_attn", config["attention_bias"]))
        mapping.update(
            {
                f"{prefix}.{name}.weight": f"{prefix}.{name}.weight"
                for name in (
                    "input_layernorm",
                    "post_attention_layernorm",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                )
            }
        )
    return mapping
