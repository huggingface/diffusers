# Copyright 2024 Lumina, Hunyuan DiT, The HuggingFace Team. All rights reserved.
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
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import FeedForward
from ..embeddings import PatchEmbed, PixArtAlphaTextProjection, apply_rotary_emb
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle, FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SmolDiTAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        cross_attention_dim,
        dim_head,
        num_heads,
        kv_heads,
    ):
        super().__init__()

        self.inner_dim = dim_head * num_heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=False)

        self.to_out = nn.Linear(self.inner_dim, query_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[torch.Tensor] = None,
        key_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape

        # Projections.
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // self.num_heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim
        query = query.view(batch_size, -1, self.num_heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply RoPE if needed
        if query_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Apply proportional attention if true
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * self.scale
            else:
                softmax_scale = self.scale

        # perform Grouped-qurey Attention (GQA)
        n_rep = self.num_heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        attention_mask = attention_mask.expand(-1, self.num_heads, sequence_length, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for self.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).to(dtype)

        return hidden_states


class SmolDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        ff_inner_dim: int,
        cross_attention_dim: int = 1024,
        activation_fn="gelu-approximate",
    ):
        super().__init__()
        from .hunyuan_transformer_2d import AdaLayerNormShift

        # 1. Self-Attn
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=True, eps=1e-6)
        self.attn1 = SmolDiTAttention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            num_heads=num_attention_heads,
            kv_heads=num_kv_heads,
        )

        # 2. Cross-Attn
        self.norm2 = FP32LayerNorm(dim, eps=1e-6, norm_elementwise_affine=True)
        self.attn2 = SmolDiTAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            num_heads=num_attention_heads,
            kv_heads=num_kv_heads,
        )

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            activation_fn=activation_fn,
            inner_dim=ff_inner_dim,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states, temb)
        attn_output = self.attn1(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # FFN Layer
        hidden_states = hidden_states + self.ff(hidden_states)

        return hidden_states


class SmolDiT2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        num_attention_heads: int = 16,
        num_kv_heads: int = 8,
        attention_head_dim: int = 88,
        in_channels: int = 4,
        out_channels: int = 4,
        activation_fn: str = "gelu-approximate",
        hidden_size=1152,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 1024,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, use_additional_conditions=False)

        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim,
            hidden_size=cross_attention_dim * 4,
            out_features=cross_attention_dim,
            act_fn="silu_fp32",
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
        )

        # SmolDiT Blocks
        self.blocks = nn.ModuleList(
            [
                SmolDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                )
                for layer in range(num_layers)
            ]
        )

        self.out_channels = out_channels
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels, bias=True)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        image_rotary_emb=None,
        return_dict=True,
    ):
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)

        # timestep
        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep,
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        # text projection
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        encoder_hidden_states = self.text_embedder(encoder_hidden_states.view(-1, encoder_hidden_states.shape[-1]))
        encoder_hidden_states = encoder_hidden_states.view(batch_size, sequence_length, -1)

        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=timestep,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )  # (N, L, D)

        # final layer
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        # (N, L, patch_size ** 2 * out_channels)

        # unpatchify: (N, out_channels, H, W)
        patch_size = self.pos_embed.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
