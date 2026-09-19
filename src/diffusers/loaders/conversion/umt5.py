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
from .t5 import t5_conversion


def umt5_conversion(config):
    mapping = dict(t5_conversion(config).mapping)
    for i in range(1, config["num_layers"]):
        key = f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
        mapping[key] = key
    return Conversion(mapping=mapping)
