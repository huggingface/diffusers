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


def ernie_image_conversion(config):
    modules = [
        "x_embedder.proj",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
        "adaLN_modulation.1",
        "final_norm.linear",
        "final_linear",
    ]
    keys = []
    if config["text_in_dim"] != config["hidden_size"]:
        keys.append("text_proj.weight")
    for i in range(config["num_layers"]):
        prefix = f"layers.{i}"
        names = [
            "adaLN_sa_ln",
            "adaLN_mlp_ln",
            "self_attention.to_q",
            "self_attention.to_k",
            "self_attention.to_v",
            "self_attention.to_out.0",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.linear_fc2",
        ]
        if config["qk_layernorm"]:
            names.extend(["self_attention.norm_q", "self_attention.norm_k"])
        keys.extend(f"{prefix}.{name}.weight" for name in names)
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in keys})
