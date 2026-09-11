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
from .transforms import Permute


def spectrogram_notes_conversion(config):
    mapping, rules = _spectrogram_encoder_rules(config)
    mapping["token_embedder.embedding"] = "token_embedder.weight"
    return Conversion(mapping=mapping, rules=rules)


def _spectrogram_encoder_rules(config):
    mapping = {"Embed_0.embedding": "position_encoding.weight", "encoder_norm.scale": "layer_norm.weight"}
    modules = []
    for i in range(config["num_layers"]):
        old, new = f"layers_{i}", f"encoders.{i}"
        mapping[old + ".pre_attention_layer_norm.scale"] = new + ".layer.0.layer_norm.weight"
        mapping[old + ".pre_mlp_layer_norm.scale"] = new + ".layer.1.layer_norm.weight"
        modules.extend(
            (f"{old}.attention.{a}", f"{new}.layer.0.SelfAttention.{b}")
            for a, b in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "o"))
        )
        names = ("wi_0", "wi_1", "wo") if config["feed_forward_proj"].startswith("gated-") else ("wi", "wo")
        modules.extend((f"{old}.mlp.{name}", f"{new}.layer.1.DenseReluDense.{name}") for name in names)
    rules = tuple(Rule((old + ".kernel",), (new + ".weight",), Permute((1, 0))) for old, new in modules)
    return mapping, rules
