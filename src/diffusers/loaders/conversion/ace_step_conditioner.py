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


from .ace_step import _ace_step_encoder_mapping
from .core import Conversion, Rule
from .transforms import Permute


def ace_step_conditioner_conversion(config):
    mapping = {
        "encoder.text_projector.weight": "text_projector.weight",
        "null_condition_emb": "null_condition_emb",
        "encoder.timbre_encoder.special_token": "timbre_encoder.special_token",
    }
    for prefix, count in (
        ("lyric_encoder", config["num_lyric_encoder_hidden_layers"]),
        ("timbre_encoder", config["num_timbre_encoder_hidden_layers"]),
    ):
        mapping.update(
            {
                f"encoder.{prefix}.{old}": f"{prefix}.{new}"
                for old, new in _ace_step_encoder_mapping(config, count).items()
            }
        )
    return Conversion(mapping=mapping, rules=(Rule(("silence_latent",), ("silence_latent",), Permute((0, 2, 1))),))
