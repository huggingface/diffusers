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
from .core import Conversion


def ace_step_detokenizer_conversion(config):
    mapping = _ace_step_encoder_mapping(config, config["num_attention_pooler_hidden_layers"])
    mapping["special_tokens"] = "special_tokens"
    mapping.update({f"proj_out.{p}": f"proj_out.{p}" for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
