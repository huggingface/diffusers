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


def unidiffuser_text_conversion(config):
    mapping = {}
    if config["prefix_hidden_dim"] is not None:
        mapping.update(
            {f"{name}.{p}": f"{name}.{p}" for name in ("encode_prefix", "decode_prefix") for p in ("weight", "bias")}
        )
    keys = ["transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"]
    modules = ["transformer.ln_f"]
    for i in range(config["n_layer"]):
        modules.extend(
            f"transformer.h.{i}.{name}"
            for name in ("ln_1", "ln_2", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        )
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    mapping.update({f"gpt.{key}": f"transformer.{key}" for key in keys})
    return Conversion(mapping=mapping)
