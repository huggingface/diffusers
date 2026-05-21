# Copyright 2025 The HuggingFace Team. All rights reserved.
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


from dataclasses import dataclass

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ..modeling_utils import ModelMixin


def add_special_tokens(hidden_states, attention_mask, sos_token, eos_token):
    batch_size = hidden_states.shape[0]

    if attention_mask is not None:
        # Add two more steps to attn mask
        new_attn_mask_step = attention_mask.new_ones((batch_size, 1))
        attention_mask = torch.concat([new_attn_mask_step, attention_mask, new_attn_mask_step], dim=-1)

    # Add the SOS / EOS tokens at the start / end of the sequence respectively
    sos_token = sos_token.expand(batch_size, 1, -1)
    eos_token = eos_token.expand(batch_size, 1, -1)
    hidden_states = torch.concat([sos_token, hidden_states, eos_token], dim=1)
    return hidden_states, attention_mask


@dataclass
class AudioLDM2ProjectionModelOutput(BaseOutput):
    """
    Args:
    Class for AudioLDM2 projection layer's outputs.
        hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states obtained by linearly projecting the hidden-states for each of the text
             encoders and subsequently concatenating them together.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices, formed by concatenating the attention masks
             for the two text encoders together. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    """

    hidden_states: torch.Tensor
    attention_mask: torch.LongTensor | None = None


class AudioLDM2ProjectionModel(ModelMixin, ConfigMixin):
    """
    A simple linear projection model to map two text embeddings to a shared latent space. It also inserts learned
    embedding vectors at the start and end of each text embedding sequence respectively. Each variable appended with
    `_1` refers to that corresponding to the second text encoder. Otherwise, it is from the first.

    Args:
        text_encoder_dim (`int`):
            Dimensionality of the text embeddings from the first text encoder (CLAP).
        text_encoder_1_dim (`int`):
            Dimensionality of the text embeddings from the second text encoder (T5 or VITS).
        langauge_model_dim (`int`):
            Dimensionality of the text embeddings from the language model (GPT2).
    """

    @register_to_config
    def __init__(
        self,
        text_encoder_dim,
        text_encoder_1_dim,
        langauge_model_dim,
        use_learned_position_embedding=None,
        max_seq_length=None,
    ):
        super().__init__()
        # additional projection layers for each text encoder
        self.projection = nn.Linear(text_encoder_dim, langauge_model_dim)
        self.projection_1 = nn.Linear(text_encoder_1_dim, langauge_model_dim)

        # learnable SOS / EOS token embeddings for each text encoder
        self.sos_embed = nn.Parameter(torch.ones(langauge_model_dim))
        self.eos_embed = nn.Parameter(torch.ones(langauge_model_dim))

        self.sos_embed_1 = nn.Parameter(torch.ones(langauge_model_dim))
        self.eos_embed_1 = nn.Parameter(torch.ones(langauge_model_dim))

        self.use_learned_position_embedding = use_learned_position_embedding

        # learable positional embedding for vits encoder
        if self.use_learned_position_embedding is not None:
            self.learnable_positional_embedding = torch.nn.Parameter(
                torch.zeros((1, text_encoder_1_dim, max_seq_length))
            )

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        hidden_states_1: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        attention_mask_1: torch.LongTensor | None = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`, *optional*):
                Hidden states from the first text encoder of shape `(batch_size, sequence_length, text_encoder_dim)`.
            hidden_states_1 (`torch.Tensor`, *optional*):
                Hidden states from the second text encoder of shape `(batch_size, sequence_length_1,
                text_encoder_1_dim)`.
            attention_mask (`torch.LongTensor`, *optional*):
                Attention mask of shape `(batch_size, sequence_length)` for `hidden_states`.
            attention_mask_1 (`torch.LongTensor`, *optional*):
                Attention mask of shape `(batch_size, sequence_length_1)` for `hidden_states_1`.
        """
        hidden_states = self.projection(hidden_states)
        hidden_states, attention_mask = add_special_tokens(
            hidden_states, attention_mask, sos_token=self.sos_embed, eos_token=self.eos_embed
        )

        # Add positional embedding for Vits hidden state
        if self.use_learned_position_embedding is not None:
            hidden_states_1 = (hidden_states_1.permute(0, 2, 1) + self.learnable_positional_embedding).permute(0, 2, 1)

        hidden_states_1 = self.projection_1(hidden_states_1)
        hidden_states_1, attention_mask_1 = add_special_tokens(
            hidden_states_1, attention_mask_1, sos_token=self.sos_embed_1, eos_token=self.eos_embed_1
        )

        # concatenate clap and t5 text encoding
        hidden_states = torch.cat([hidden_states, hidden_states_1], dim=1)

        # concatenate attention masks
        if attention_mask is None and attention_mask_1 is not None:
            attention_mask = attention_mask_1.new_ones((hidden_states[:2]))
        elif attention_mask is not None and attention_mask_1 is None:
            attention_mask_1 = attention_mask.new_ones((hidden_states_1[:2]))

        if attention_mask is not None and attention_mask_1 is not None:
            attention_mask = torch.cat([attention_mask, attention_mask_1], dim=-1)
        else:
            attention_mask = None

        return AudioLDM2ProjectionModelOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
