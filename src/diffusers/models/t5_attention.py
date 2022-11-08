# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5Config,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
)

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from .embeddings import get_timestep_embedding
from .film import FiLMLayer


class T5LayerSelfAttentionCond(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.FiLMLayer = FiLMLayer(in_features=config.d_model * 4, out_features=config.d_model)
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        conditioning_emb=None,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # pre_self_attention_layer_norm
        normed_hidden_states = self.layer_norm(hidden_states)

        if conditioning_emb is not None:
            normed_hidden_states = self.FiLMLayer(normed_hidden_states, conditioning_emb)

        # Self-attention block
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer = nn.ModuleList()

        # cond self attention: layer 0
        self.layer.append(T5LayerSelfAttentionCond(config, has_relative_attention_bias=has_relative_attention_bias))

        # cross attention: layer 1
        self.layer.append(T5LayerCrossAttention(config))

        # pre_mlp_layer_norm: layer 2
        self.layer.append(T5LayerNorm(hidden_size=config.d_model))

        # FiLM layer: 3
        self.layer.append(FiLMLayer(in_features=config.d_model * 4, out_features=config.d_model))

        # MLP + dropout: last layer
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        conditioning_emb=None,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            conditioning_emb=conditioning_emb,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply LayerNorm
        hidden_states = self.layer[2](hidden_states)

        # FiLM
        if conditioning_emb is not None:
            hidden_states = self.layer[3](hidden_states, conditioning_emb)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class TokenEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, config: T5Config):
        super().__init__()

        self.token_embedder = nn.Embedding(config.vocab_size, config.d_model)

        self.position_encoding = nn.Embedding(config.max_length, config.d_model)
        self.position_encoding.weight.requires_grad = False

        self.dropout_pre = nn.Dropout(p=config.dropout_rate)

        config.is_decoder = False
        config.is_encoder_decoder = False
        self.encoders = nn.ModuleList()
        for lyr_num in range(config.num_layers):
            lyr = T5Block(config)
            self.encoders.append(lyr)

        self.layer_norm = T5LayerNorm(hidden_size=config.d_model)
        self.dropout_post = nn.Dropout(p=config.dropout_rate)

    def forward(self, encoder_input_tokens, encoder_inputs_mask):
        x = self.token_embedder(encoder_input_tokens)

        seq_length = encoder_input_tokens.shape[1]
        inputs_positions = torch.arange(seq_length, device=encoder_input_tokens.device)
        x += self.position_encoding(inputs_positions)

        x = self.dropout_pre(x)

        for lyr in self.encoders:
            x = lyr(x, encoder_inputs_mask)[0]
        x = self.layer_norm(x)

        return self.dropout_post(x), encoder_inputs_mask


class ContinuousEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, config):
        super().__init__()

        self.input_proj = nn.Linear(config.input_dims, config.d_model, bias=False)

        self.position_encoding = nn.Embedding(config.targets_context_length, config.d_model)
        self.position_encoding.weight.requires_grad = False

        self.dropout_pre = nn.Dropout(p=config.dropout_rate)

        config.is_decoder = False
        config.is_encoder_decoder = False
        self.encoders = nn.ModuleList()
        for lyr_num in range(config.num_layers):
            lyr = T5Block(config)
            self.encoders.append(lyr)

        self.layer_norm = T5LayerNorm(hidden_size=config.d_model)
        self.dropout_post = nn.Dropout(p=config.dropout_rate)

    def forward(self, encoder_inputs, encoder_inputs_mask):
        x = self.input_proj(encoder_inputs)

        # terminal relative positional encodings
        max_positions = encoder_inputs.shape[1]
        input_positions = torch.arange(max_positions, device=encoder_inputs.device)

        seq_lens = encoder_inputs_mask.sum(-1)
        input_positions = torch.roll(input_positions.unsqueeze(0), tuple(seq_lens.tolist()), dims=0)
        x += self.position_encoding(input_positions)

        x = self.dropout_pre(x)

        for lyr in self.encoders:
            x = lyr(x, encoder_inputs_mask)[0]
        x = self.layer_norm(x)

        return self.dropout_post(x), encoder_inputs_mask


class Decoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, config: T5Config):
        super().__init__()

        self.conditioning_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model * 4, bias=False),
            nn.SiLU(),
        )

        self.position_encoding = nn.Embedding(config.targets_length, config.d_model)
        self.position_encoding.weight.requires_grad = False

        self.continuous_inputs_projection = nn.Linear(config.input_dims, config.d_model)

        self.dropout = nn.Dropout(p=config.dropout_rate)

        config.is_decoder = True
        config.is_encoder_decoder = False
        self.decoders = nn.ModuleList()
        for lyr_num in range(config.num_decoder_layers):
            # FiLM conditional T5 decoder
            lyr = DecoderLayer(config)
            self.decoders.append(lyr)

        self.decoder_norm = T5LayerNorm(config.d_model)

        self.post_dropout = nn.Dropout(p=config.dropout_rate)
        self.spec_out = nn.Linear(config.d_model, config.input_dims, bias=False)

        self.max_decoder_noise_time = config.max_decoder_noise_time
        self.emb_dim = config.d_model

    def encoder_decoder_mask(self, query_input, key_input, pairwise_fn=torch.mul):
        mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
        return mask.unsqueeze(-3)

    def forward(self, encodings_and_masks, decoder_input_tokens, decoder_noise_time):
        batch, _, _ = decoder_input_tokens.shape
        assert decoder_noise_time.shape == (batch,)

        # decoder_noise_time is in [0, 1), so rescale to expected timing range.
        conditioning_emb = get_timestep_embedding(
            decoder_noise_time * self.max_decoder_noise_time,
            embedding_dim=self.emb_dim,
            max_period=self.max_decoder_noise_time,
        )

        conditioning_emb = self.conditioning_emb(conditioning_emb)

        assert conditioning_emb.shape == (batch, self.emb_dim * 4)

        seq_length = decoder_input_tokens.shape[1]

        # If we want to use relative positions for audio context, we can just offset
        # this sequence by the length of encodings_and_masks.
        decoder_positions = torch.broadcast_to(
            torch.arange(seq_length, device=decoder_input_tokens.device),
            (batch, seq_length),
        )

        position_encodings = self.position_encoding(decoder_positions)

        # decoder: No padding present.
        decoder_mask = torch.ones(decoder_input_tokens.shape[:2], device=decoder_input_tokens.device)

        # Translate encoding masks to encoder-decoder masks.
        encodings_and_encdec_masks = [(x, self.encoder_decoder_mask(decoder_mask, y)) for x, y in encodings_and_masks]

        inputs = self.continuous_inputs_projection(decoder_input_tokens)

        inputs += position_encodings

        y = self.dropout(inputs)

        # cross attend style: concat encodings
        encoded = torch.cat([x[0] for x in encodings_and_encdec_masks], dim=1)
        encoder_decoder_mask = torch.cat([x[1] for x in encodings_and_encdec_masks], dim=-1)
        for lyr in self.decoders:
            y = lyr(
                y,
                conditioning_emb=conditioning_emb,
                encoder_hidden_states=encoded,
                encoder_attention_mask=encoder_decoder_mask,
            )[0]

        y = self.decoder_norm(y)
        y = self.post_dropout(y)

        spec_out = self.spec_out(y)
        return spec_out


class ContinuousContextTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, config: T5Config):
        super().__init__()

        self.token_encoder = TokenEncoder(config=config)
        self.continuous_encoder = ContinuousEncoder(config=config)
        self.decoder = Decoder(config=config)

    def encode(self, input_tokens, continuous_inputs, continuous_mask):
        tokens_mask = input_tokens > 0
        tokens_encoded, tokens_mask = self.token_encoder(
            encoder_input_tokens=input_tokens,
            encoder_inputs_mask=tokens_mask,
        )

        continuous_encoded, continuous_mask = self.continuous_encoder(
            encoder_inputs=continuous_inputs,
            encoder_inputs_mask=continuous_mask,
        )

        return [(tokens_encoded, tokens_mask), (continuous_encoded, continuous_mask)]

    def decode(self, encodings_and_masks, input_tokens, noise_time):
        logits = self.decoder(
            encodings_and_masks=encodings_and_masks,
            decoder_input_tokens=input_tokens,
            decoder_noise_time=noise_time,
        )
        return logits

    def forward(
        self,
        encoder_input_tokens,
        encoder_continuous_inputs,
        encoder_continuous_mask,
        decoder_input_tokens,
        decoder_noise_time,
    ):
        encodings_and_masks = self.encode(
            input_tokens=encoder_input_tokens,
            continuous_inputs=encoder_continuous_inputs,
            continuous_mask=encoder_continuous_mask,
        )

        return self.decode(
            encodings_and_masks=encodings_and_masks,
            input_tokens=decoder_input_tokens,
            noise_time=decoder_noise_time,
        )
