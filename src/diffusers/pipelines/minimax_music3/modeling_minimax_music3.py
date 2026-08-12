# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.attention import AttentionModuleMixin
from ...models.attention_dispatch import dispatch_attention_fn
from ...models.modeling_utils import ModelMixin
from ...models.normalization import RMSNorm


class MiniMaxMusic3ConditionEncoder(ModelMixin, ConfigMixin):
    r"""
    Projects the per-frame hidden states of the autoregressive stage onto the Flow-VAE latent timeline.

    Each generated frame carries `num_condition_layers` hidden states of size `condition_hidden_dim` (one from the
    language model and one per residual codebook step). They are mixed with learned softmax weights, projected, and
    resampled from the language-model frame rate to the latent frame rate with nearest-neighbor interpolation.
    """

    @register_to_config
    def __init__(
        self,
        condition_hidden_dim: int = 4096,
        num_condition_layers: int = 8,
        out_dim: int = 2048,
        input_sampling_rate: int = 24000,
        input_hop_length: int = 960,
        output_sampling_rate: int = 44100,
        output_hop_length: int = 512,
    ):
        super().__init__()
        self.layer_weight_logits = nn.Parameter(torch.zeros(num_condition_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, frames, num_condition_layers * condition_hidden_dim)`):
                Concatenated per-frame hidden states from the autoregressive stage.

        Returns:
            `torch.Tensor` of shape `(batch, latent_length, out_dim)`: the latent-aligned conditioning sequence.
        """
        batch_size, num_frames, _ = hidden_states.shape
        num_layers = self.config.num_condition_layers
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, num_layers, self.config.condition_hidden_dim, num_frames)
        layer_weights = torch.softmax(self.layer_weight_logits, dim=0).to(hidden_states.dtype)
        hidden_states = torch.einsum("blht,l->bht", hidden_states, layer_weights)
        hidden_states = self.layer_scale.to(hidden_states.dtype) * hidden_states
        hidden_states = self.proj(hidden_states)
        latent_length = max(
            1,
            int(
                num_frames
                * self.config.output_sampling_rate
                / self.config.input_sampling_rate
                * self.config.input_hop_length
                / self.config.output_hop_length
            ),
        )
        hidden_states = F.interpolate(hidden_states, size=latent_length, mode="nearest")
        return hidden_states.transpose(1, 2)


class MiniMaxMusic3DepthAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "MiniMaxMusic3DepthAttention", hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.view(batch_size, seq_len, attn.heads, attn.head_dim)
        key = key.view(batch_size, seq_len, attn.heads, attn.head_dim)
        value = value.view(batch_size, seq_len, attn.heads, attn.head_dim)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            is_causal=True,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return attn.to_out(hidden_states)


class MiniMaxMusic3DepthAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxMusic3DepthAttnProcessor
    _available_processors = [MiniMaxMusic3DepthAttnProcessor]

    def __init__(self, dim: int, heads: int, processor: Optional[MiniMaxMusic3DepthAttnProcessor] = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.set_processor(processor or MiniMaxMusic3DepthAttnProcessor())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class MiniMaxMusic3DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MiniMaxMusic3DepthAttention(dim, heads)
        self.post_attention_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.input_layernorm(hidden_states))
        norm_states = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states))


class MiniMaxMusic3RVQDepthDecoder(ModelMixin, ConfigMixin):
    r"""
    The local language model of MiniMax Music 3. Within each audio frame it autoregressively predicts the seven
    residual RVQ codebooks (c1..c7) from the global language model's hidden state and the frame's semantic code, and
    exposes the per-step hidden states that condition the flow-matching transformer.

    It also owns the embedding table for the residual codebooks, which the pipeline uses to embed complete frames for
    the global language model's feedback loop.
    """

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_position_embeddings: int = 16,
    ):
        super().__init__()
        self.audio_embeddings = nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList(
            [
                MiniMaxMusic3DepthDecoderBlock(hidden_size, num_attention_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.audio_heads = nn.ModuleList(
            [nn.Linear(hidden_size, audio_vocab_size, bias=False) for _ in range(num_codebooks - 1)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds (`torch.Tensor` of shape `(batch, steps, hidden_size)`):
                Projected depth-sequence embeddings: the global hidden state followed by the embedded codes sampled so
                far, each passed through `projection`.

        Returns:
            `torch.Tensor` of shape `(batch, steps, hidden_size)`: normalized hidden states; the last step feeds the
            next codebook head.
        """
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class MiniMaxMusic3VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.snake1 = MiniMaxMusic3Snake1d(dim)
        self.conv1 = weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = MiniMaxMusic3Snake1d(dim)
        self.conv2 = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        return hidden_states + residual


class MiniMaxMusic3VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = MiniMaxMusic3Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
            )
        )
        self.res_unit1 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=1)
        self.res_unit2 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=3)
        self.res_unit3 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class MiniMaxMusic3Vocoder(ModelMixin, ConfigMixin):
    r"""
    The Flow-VAE waveform decoder of MiniMax Music 3 (a DAC-style decoder). It decodes flow-matched latents of shape
    `(batch, latent_channels, length)` into stereo waveforms at `sampling_rate`; the two audio channels are decoded as
    two folded `latent_channels // 2` streams.
    """

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 128,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: tuple = (8, 8, 4, 2),
        sampling_rate: int = 44100,
    ):
        super().__init__()
        self.dec_in_proj = nn.Conv1d(latent_channels // 2, decoder_input_dim, kernel_size=1)
        self.conv_in = weight_norm(nn.Conv1d(decoder_input_dim, decoder_hidden_dim, kernel_size=7, padding=3))
        blocks = []
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_dim = decoder_hidden_dim // (2**index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(MiniMaxMusic3VocoderBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = MiniMaxMusic3Snake1d(output_dim)
        self.conv_out = weight_norm(nn.Conv1d(output_dim, 1, kernel_size=7, padding=3))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            latents (`torch.Tensor` of shape `(batch, latent_channels, length)`):
                Flow-matched Flow-VAE latents.

        Returns:
            `torch.Tensor` of shape `(batch, 2, samples)`: the stereo waveform in `[-1, 1]`.
        """
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.config.latent_channels // 2, length)
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = torch.tanh(self.conv_out(self.snake_out(hidden_states)))
        return waveform.reshape(batch_size, 2, -1)
