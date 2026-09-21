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


def clap_text_conversion(config):
    mapping = {
        f"text_branch.embeddings.{name}": f"text_model.embeddings.{name}"
        for name in (
            "position_ids",
            "token_type_ids",
            "word_embeddings.weight",
            "position_embeddings.weight",
            "token_type_embeddings.weight",
        )
    }
    modules = [
        ("text_branch.embeddings.LayerNorm", "text_model.embeddings.LayerNorm"),
        ("text_branch.pooler.dense", "text_model.pooler.dense"),
        ("text_projection.0", "text_projection.linear1"),
        ("text_projection.2", "text_projection.linear2"),
    ]
    for i in range(config["num_hidden_layers"]):
        modules.extend(
            (f"text_branch.encoder.layer.{i}.{name}", f"text_model.encoder.layer.{i}.{name}")
            for name in (
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "attention.output.LayerNorm",
                "intermediate.dense",
                "output.dense",
                "output.LayerNorm",
            )
        )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
