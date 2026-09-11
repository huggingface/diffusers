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


def t5_film_conversion(config):
    mapping = {"Embed_0.embedding": "position_encoding.weight", "decoder_norm.scale": "decoder_norm.weight"}
    modules = [
        ("time_emb_dense0", "conditioning_emb.0"),
        ("time_emb_dense1", "conditioning_emb.2"),
        ("continuous_inputs_projection", "continuous_inputs_projection"),
        ("spec_out_dense", "spec_out"),
    ]
    for i in range(config["num_layers"]):
        old, new = f"layers_{i}", f"decoders.{i}"
        for j, norm in enumerate(
            ("pre_self_attention_layer_norm", "pre_cross_attention_layer_norm", "pre_mlp_layer_norm")
        ):
            mapping[f"{old}.{norm}.scale"] = f"{new}.layer.{j}.layer_norm.weight"
        for j, attn in enumerate(("self_attention", "MultiHeadDotProductAttention_0")):
            modules.extend(
                (f"{old}.{attn}.{a}", f"{new}.layer.{j}.attention.{b}")
                for a, b in (("query", "to_q"), ("key", "to_k"), ("value", "to_v"), ("out", "to_out.0"))
            )
        modules.extend(
            [
                (old + ".FiLMLayer_0.DenseGeneral_0", new + ".layer.0.FiLMLayer.scale_bias"),
                (old + ".FiLMLayer_1.DenseGeneral_0", new + ".layer.2.film.scale_bias"),
            ]
        )
        modules.extend(
            (f"{old}.mlp.{name}", f"{new}.layer.2.DenseReluDense.{name}") for name in ("wi_0", "wi_1", "wo")
        )
    rules = tuple(Rule((old + ".kernel",), (new + ".weight",), Permute((1, 0))) for old, new in modules)
    return Conversion(mapping=mapping, rules=rules)
