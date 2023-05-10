# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin


class KandinskyTextProjModel(ModelMixin, ConfigMixin):
    """
    Utility class for Kandingsky text embeddings. Used to combine the image and text embeddings into a format usable by
    the unet diffusion model.
    """

    @register_to_config
    def __init__(
        self,
        *,
        clip_extra_context_tokens: int = 10,
        clip_text_encoder_hidden_states_dim: int = 1024,
        clip_embeddings_dim: int = 768,
        time_embed_dim: int = 1536,
        cross_attention_dim: int = 768,
    ):
        super().__init__()

        # parameters for additional clip time embeddings
        self.embedding_proj = nn.Linear(clip_embeddings_dim, time_embed_dim)
        self.embedding_norm = nn.LayerNorm(time_embed_dim)
        self.clip_image_embeddings_project_to_time_embeddings = nn.Linear(clip_embeddings_dim, time_embed_dim)

        # parameters for encoder hidden states
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.clip_extra_context_tokens_proj = nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.encoder_hidden_states_proj = nn.Linear(clip_text_encoder_hidden_states_dim, cross_attention_dim)

    def forward(self, *, image_embeddings, prompt_embeds, text_encoder_hidden_states):
        # The image embeddings batch size and the text embeddings batch size are equal
        assert image_embeddings.shape[0] == prompt_embeds.shape[0] == text_encoder_hidden_states.shape[0]

        batch_size = prompt_embeds.shape[0]

        # project text and image embeddings to add to the existing timestep embedding
        time_projected_prompt_embeds = self.embedding_proj(prompt_embeds)
        time_projected_prompt_embeds = self.embedding_norm(time_projected_prompt_embeds)
        time_projected_image_embeddings = self.clip_image_embeddings_project_to_time_embeddings(image_embeddings)
        additive_clip_time_embeddings = time_projected_image_embeddings + time_projected_prompt_embeds

        # extra tokens of context that are concatenated to the sequence of outputs from the GLIDE text encoder"
        clip_extra_context_tokens = self.clip_extra_context_tokens_proj(image_embeddings)
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(batch_size, self.clip_extra_context_tokens, -1)

        text_encoder_hidden_states = self.encoder_hidden_states_proj(text_encoder_hidden_states)
        text_encoder_hidden_states = torch.cat([clip_extra_context_tokens, text_encoder_hidden_states], dim=1)

        return text_encoder_hidden_states, additive_clip_time_embeddings
