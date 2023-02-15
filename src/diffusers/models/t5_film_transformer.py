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
from torch import nn
import math
from ..configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin
from .embeddings import get_timestep_embedding
from .attention import FiLMLayer


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class T5LayerCrossAttention(nn.Module):
    def __init__(self, d_model, d_kv, num_heads, dropout_rate, layer_norm_epsilon):
        super().__init__()
        self.EncDecAttention = T5Attention(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,)
        return outputs


class T5Attention(nn.Module):
    def __init__(self, d_model, d_kv, num_heads, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            else:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            return hidden_states
        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(hidden_states, self.k, key_value_states)
        value_states = project(hidden_states, self.v, key_value_states)

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        if mask is not None:
            scores += mask
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        return (attn_output,)


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = NewGELUActivation()

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if hidden_states.dtype != self.wo.weight.dtype and self.wo.weight.dtype != torch.int8:
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFFCond(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate, layer_norm_epsilon):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)
        self.film = FiLMLayer(in_features=d_model * 4, out_features=d_model)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, conditioning_emb=None):
        forwarded_states = self.layer_norm(hidden_states)
        if conditioning_emb is not None:
            forwarded_states = self.film(forwarded_states, conditioning_emb)

        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5LayerSelfAttentionCond(nn.Module):
    def __init__(self, d_model, d_kv, num_heads, dropout_rate):
        super().__init__()
        self.layer_norm = T5LayerNorm(d_model)
        self.FiLMLayer = FiLMLayer(in_features=d_model * 4, out_features=d_model)
        self.SelfAttention = T5Attention(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        conditioning_emb=None,
        attention_mask=None,
    ):
        # pre_self_attention_layer_norm
        normed_hidden_states = self.layer_norm(hidden_states)

        if conditioning_emb is not None:
            normed_hidden_states = self.FiLMLayer(normed_hidden_states, conditioning_emb)

        # Self-attention block
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,)
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_kv, num_heads, d_ff, dropout_rate, layer_norm_epsilon=1e-6):
        super().__init__()
        self.layer = nn.ModuleList()

        # cond self attention: layer 0
        self.layer.append(T5LayerSelfAttentionCond(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate))

        # cross attention: layer 1
        self.layer.append(T5LayerCrossAttention(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate, layer_norm_epsilon=layer_norm_epsilon))

        # Film Cond MLP + dropout: last layer
        self.layer.append(T5LayerFFCond(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, layer_norm_epsilon=layer_norm_epsilon))

    def forward(
        self,
        hidden_states,
        conditioning_emb=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            conditioning_emb=conditioning_emb,
            attention_mask=attention_mask,
        )
        hidden_states = self_attention_outputs[0]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            encoder_extended_attention_mask = torch.where(encoder_attention_mask > 0, 0, -1e10)

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_extended_attention_mask,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Film Conditional Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, conditioning_emb)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,)


class T5FilmDecoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_dims: int = 128,
        targets_length: int = 256,
        max_decoder_noise_time: float = 2000.0,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_kv: int = 64,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        feed_forward_proj: str = "gated-gelu",
    ):
        super().__init__()

        self.conditioning_emb = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 4, bias=False),
            nn.SiLU(),
        )

        self.position_encoding = nn.Embedding(targets_length, d_model)
        self.position_encoding.weight.requires_grad = False

        self.continuous_inputs_projection = nn.Linear(input_dims, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.decoders = nn.ModuleList()
        for lyr_num in range(num_layers):
            # FiLM conditional T5 decoder
            lyr = DecoderLayer(d_model=d_model, d_kv=d_kv, num_heads=num_heads, d_ff=d_ff, dropout_rate=dropout_rate)
            self.decoders.append(lyr)

        self.decoder_norm = T5LayerNorm(d_model)

        self.post_dropout = nn.Dropout(p=dropout_rate)
        self.spec_out = nn.Linear(d_model, input_dims, bias=False)

    def encoder_decoder_mask(self, query_input, key_input, pairwise_fn=torch.mul):
        mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
        return mask.unsqueeze(-3)

    def forward(self, encodings_and_masks, decoder_input_tokens, decoder_noise_time):
        batch, _, _ = decoder_input_tokens.shape
        assert decoder_noise_time.shape == (batch,)

        # decoder_noise_time is in [0, 1), so rescale to expected timing range.
        time_steps = get_timestep_embedding(
            decoder_noise_time * self.config.max_decoder_noise_time,
            embedding_dim=self.config.d_model,
            max_period=self.config.max_decoder_noise_time,
        ).to(dtype=self.dtype)

        conditioning_emb = self.conditioning_emb(time_steps).unsqueeze(1)

        assert conditioning_emb.shape == (batch, 1, self.config.d_model * 4)

        seq_length = decoder_input_tokens.shape[1]

        # If we want to use relative positions for audio context, we can just offset
        # this sequence by the length of encodings_and_masks.
        decoder_positions = torch.broadcast_to(
            torch.arange(seq_length, device=decoder_input_tokens.device),
            (batch, seq_length),
        )

        position_encodings = self.position_encoding(decoder_positions)

        inputs = self.continuous_inputs_projection(decoder_input_tokens)
        inputs += position_encodings
        y = self.dropout(inputs)

        # decoder: No padding present.
        decoder_mask = torch.ones(decoder_input_tokens.shape[:2], device=decoder_input_tokens.device)

        # Translate encoding masks to encoder-decoder masks.
        encodings_and_encdec_masks = [(x, self.encoder_decoder_mask(decoder_mask, y)) for x, y in encodings_and_masks]

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
