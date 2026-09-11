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


def omnigen_conversion(config):
    mapping = {
        "pos_embed": "patch_embedding.pos_embed",
        "llm.embed_tokens.weight": "embed_tokens.weight",
        "llm.norm.weight": "norm.weight",
    }
    modules = [
        ("x_embedder.proj", "patch_embedding.output_image_proj"),
        ("input_x_embedder.proj", "patch_embedding.input_image_proj"),
        ("final_layer.adaLN_modulation.1", "norm_out.linear"),
        ("final_layer.linear", "proj_out"),
    ]
    modules.extend(
        (f"{prefix}.mlp.{i}", f"{prefix}.linear_{j}")
        for prefix in ("time_token", "t_embedder")
        for i, j in ((0, 1), (2, 2))
    )
    hidden = config["hidden_size"]
    kv = hidden // config["num_attention_heads"] * config["num_key_value_heads"]
    rules = []
    for i in range(config["num_layers"]):
        old, new = f"llm.layers.{i}", f"layers.{i}"
        rules.append(
            Rule(
                (old + ".self_attn.qkv_proj.weight",),
                tuple(f"{new}.self_attn.to_{part}.weight" for part in ("q", "k", "v")),
                Split((hidden, kv, kv)),
            )
        )
        mapping[old + ".self_attn.o_proj.weight"] = new + ".self_attn.to_out.0.weight"
        mapping.update(
            {
                f"{old}.{name}.weight": f"{new}.{name}.weight"
                for name in ("input_layernorm", "post_attention_layernorm", "mlp.gate_up_proj", "mlp.down_proj")
            }
        )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
