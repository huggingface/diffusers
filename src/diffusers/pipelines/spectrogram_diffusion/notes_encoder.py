# Copyright 2022 The Music Spectrogram Diffusion Authors.
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
import torch.nn as nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.t5.modeling_t5 import T5Block, T5Config, T5LayerNorm

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin


class SpectrogramNotesEncoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(
        self,
        max_length: int,
        vocab_size: int,
        d_model: int,
        dropout_rate: float,
        num_layers: int,
        num_heads: int,
        d_kv: int,
        d_ff: int,
        feed_forward_proj: str,
        is_decoder: bool = False,
    ):
        super().__init__()

        self.token_embedder = nn.Embedding(vocab_size, d_model)

        self.position_encoding = nn.Embedding(max_length, d_model)
        self.position_encoding.weight.requires_grad = False

        self.dropout_pre = nn.Dropout(p=dropout_rate)

        t5config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            feed_forward_proj=feed_forward_proj,
            is_decoder=is_decoder,
            is_encoder_decoder=False,
        )

        self.encoders = nn.ModuleList()
        for lyr_num in range(num_layers):
            lyr = T5Block(t5config)
            self.encoders.append(lyr)

        self.layer_norm = T5LayerNorm(d_model)
        self.dropout_post = nn.Dropout(p=dropout_rate)

    def forward(self, encoder_input_tokens, encoder_inputs_mask):
        x = self.token_embedder(encoder_input_tokens)

        seq_length = encoder_input_tokens.shape[1]
        inputs_positions = torch.arange(seq_length, device=encoder_input_tokens.device)
        x += self.position_encoding(inputs_positions)

        x = self.dropout_pre(x)

        # inverted the attention mask
        input_shape = encoder_input_tokens.size()
        extended_attention_mask = self.get_extended_attention_mask(encoder_inputs_mask, input_shape)

        for lyr in self.encoders:
            x = lyr(x, extended_attention_mask)[0]
        x = self.layer_norm(x)

        return self.dropout_post(x), encoder_inputs_mask
