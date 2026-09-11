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


def t5_conversion(config):
    keys = ["shared.weight", "encoder.embed_tokens.weight", "encoder.final_layer_norm.weight"]
    for i in range(config["num_layers"]):
        prefix = f"encoder.block.{i}"
        keys.extend(f"{prefix}.layer.0.SelfAttention.{part}.weight" for part in ("q", "k", "v", "o"))
        keys.extend(f"{prefix}.layer.{j}.layer_norm.weight" for j in (0, 1))
        if i == 0:
            keys.append(prefix + ".layer.0.SelfAttention.relative_attention_bias.weight")
        names = ("wi_0", "wi_1", "wo") if config["feed_forward_proj"].startswith("gated-") else ("wi", "wo")
        keys.extend(f"{prefix}.layer.1.DenseReluDense.{name}.weight" for name in names)
    return Conversion(mapping={key: key for key in keys})
