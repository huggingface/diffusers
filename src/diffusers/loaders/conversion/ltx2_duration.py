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


def ltx2_duration_conversion(config):
    names = ["video_input_proj", "audio_input_proj", "mlp_hidden", "mlp_out"]
    mapping = {f"{name}.{p}": f"{name}.{p}" for name in names for p in ("weight", "bias")}
    for name in ("video_modality_emb", "audio_modality_emb", "attention_pooler.query_tokens"):
        mapping[name] = name
    rules = []
    for p in ("weight", "bias"):
        mapping[f"attention_pooler.cross_attn.out_proj.{p}"] = f"attention_pooler.to_out.{p}"
        rules.append(
            Rule(
                (f"attention_pooler.cross_attn.in_proj_{p}",),
                tuple(f"attention_pooler.to_{kind}.{p}" for kind in ("q", "k", "v")),
                Split((config["pooler_hidden_dim"],) * 3),
            )
        )
    return Conversion(mapping=mapping, rules=rules)
