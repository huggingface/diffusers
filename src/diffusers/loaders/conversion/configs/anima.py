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

"""Original configuration helpers and model presets for the anima assembly recipe."""

from typing import Any

import torch
from transformers import Qwen3Config


def infer_text_conditioner_config(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    model_dim = state_dict["blocks.0.self_attn.q_proj.weight"].shape[0]
    source_dim = state_dict["blocks.0.cross_attn.k_proj.weight"].shape[1]
    target_vocab_size, target_dim = state_dict["embed.weight"].shape
    attention_head_dim = state_dict["blocks.0.self_attn.q_norm.weight"].shape[0]
    num_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("blocks."))

    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "model_dim": model_dim,
        "num_layers": num_layers,
        "num_attention_heads": model_dim // attention_head_dim,
        "target_vocab_size": target_vocab_size,
    }


def infer_qwen3_config(state_dict: dict[str, torch.Tensor]) -> Qwen3Config:
    vocab_size, hidden_size = state_dict["embed_tokens.weight"].shape
    intermediate_size = state_dict["layers.0.mlp.gate_proj.weight"].shape[0]
    num_hidden_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("layers."))
    head_dim = state_dict["layers.0.self_attn.q_norm.weight"].shape[0]
    num_attention_heads = state_dict["layers.0.self_attn.q_proj.weight"].shape[0] // head_dim
    num_key_value_heads = state_dict["layers.0.self_attn.k_proj.weight"].shape[0] // head_dim

    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        head_dim=head_dim,
        attention_bias=False,
        tie_word_embeddings=False,
    )


__all__ = ["infer_qwen3_config", "infer_text_conditioner_config"]
