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


def qwen3_conversion(config):
    causal = config["_class_name"] == "Qwen3ForCausalLM"
    prefix = "model." if causal else ""
    keys = [prefix + "embed_tokens.weight", prefix + "norm.weight"]
    if causal:
        keys.append("lm_head.weight")
    for i in range(config["num_hidden_layers"]):
        block = f"{prefix}layers.{i}"
        keys.extend(
            f"{block}.{name}.weight"
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "self_attn.q_norm",
                "self_attn.k_norm",
            )
        )
        if config["attention_bias"]:
            keys.extend(f"{block}.self_attn.{name}.bias" for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    return Conversion(mapping={key: key for key in keys})
