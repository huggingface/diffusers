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


def minimax_music3_rvq_conversion(config):
    mapping = {"model.audio_extra_embedding.weight": "audio_embeddings.weight"}
    modules = [("model.audio_decoder." + name, name) for name in ("projection", "pos_embedding", "norm")]
    modules.extend(
        (f"model.audio_decoder.audio_heads.{i}", f"audio_heads.{i}") for i in range(config["num_codebooks"] - 1)
    )
    for i in range(config["num_layers"]):
        old, new = f"model.audio_decoder.layers.{i}", f"layers.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("input_layernorm", "input_layernorm"),
                ("post_attention_layernorm", "post_attention_layernorm"),
                ("self_attn.q_proj", "attn.to_q"),
                ("self_attn.k_proj", "attn.to_k"),
                ("self_attn.v_proj", "attn.to_v"),
                ("self_attn.o_proj", "attn.to_out"),
                ("mlp.gate_proj", "gate_proj"),
                ("mlp.up_proj", "up_proj"),
                ("mlp.down_proj", "down_proj"),
            )
        )
    mapping.update({old + ".weight": new + ".weight" for old, new in modules})
    return Conversion(mapping=mapping)
