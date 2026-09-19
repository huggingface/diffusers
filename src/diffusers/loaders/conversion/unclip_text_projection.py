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


def unclip_text_projection_conversion(config):
    modules = [
        ("text_seq_proj.0", "encoder_hidden_states_proj"),
        ("text_seq_proj.1", "text_encoder_hidden_states_norm"),
        ("clip_tok_proj", "clip_extra_context_tokens_proj"),
        ("text_feat_proj", "embedding_proj"),
        ("clip_emb", "clip_image_embeddings_project_to_time_embeddings"),
    ]
    mapping = {"cf_param": "learned_classifier_free_guidance_embeddings"}
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
