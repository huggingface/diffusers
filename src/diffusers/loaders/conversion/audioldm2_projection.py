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
from .transforms import Chain, Reshape, Split


def audioldm2_projection_conversion(config):
    modules = [("input_sequence_embed_linear.0", "projection"), ("input_sequence_embed_linear.1", "projection_1")]
    mapping = {f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")}
    dim = config["langauge_model_dim"]
    transform = Chain((Reshape((2, dim), (2 * dim,)), Split((dim,) * 2)))
    rules = tuple(
        Rule((f"{old}_of_sequence_tokens.weight",), (f"{new}_embed", f"{new}_embed_1"), transform)
        for old, new in (("start", "sos"), ("end", "eos"))
    )
    if config["use_learned_position_embedding"] is not None:
        mapping["learnable_positional_embedding"] = "learnable_positional_embedding"
    return Conversion(mapping=mapping, rules=rules)
