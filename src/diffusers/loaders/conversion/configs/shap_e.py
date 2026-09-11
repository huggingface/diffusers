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

"""Original configuration helpers and model presets for the shap_e assembly recipe."""

PRIOR_CONFIG = {
    "num_attention_heads": 16,
    "attention_head_dim": 1024 // 16,
    "num_layers": 24,
    "embedding_dim": 1024,
    "num_embeddings": 1024,
    "additional_embeddings": 0,
    "time_embed_act_fn": "gelu",
    "norm_in_type": "layer",
    "encoder_hid_proj_type": None,
    "added_emb_type": None,
    "time_embed_dim": 1024 * 4,
    "embedding_proj_dim": 768,
    "clip_embed_dim": 1024 * 2,
}

PRIOR_IMAGE_CONFIG = {
    "num_attention_heads": 8,
    "attention_head_dim": 1024 // 8,
    "num_layers": 24,
    "embedding_dim": 1024,
    "num_embeddings": 1024,
    "additional_embeddings": 0,
    "time_embed_act_fn": "gelu",
    "norm_in_type": "layer",
    "embedding_proj_norm_type": "layer",
    "encoder_hid_proj_type": None,
    "added_emb_type": None,
    "time_embed_dim": 1024 * 4,
    "embedding_proj_dim": 1024,
    "clip_embed_dim": 1024 * 2,
}

RENDERER_CONFIG = {}

__all__ = ["PRIOR_CONFIG", "PRIOR_IMAGE_CONFIG", "RENDERER_CONFIG"]
