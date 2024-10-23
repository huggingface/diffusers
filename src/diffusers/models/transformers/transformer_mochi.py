# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
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

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..embeddings import PatchEmbed, MochiAttentionPool, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import MochiRMSNormZero, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class MochiTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        caption_dim: int,
        activation_fn: str = "swiglu",
        update_captions: bool = True,
    ) -> None:
        super().__init__()

        self.update_captions = update_captions
        
        self.norm1 = MochiRMSNormZero(dim, 4 * dim)

        if update_captions:
            self.norm_context1 = MochiRMSNormZero(dim, 4 * caption_dim)
        else:
            self.norm_context1 = RMSNorm(caption_dim, eps=1e-5, elementwise_affine=False)
        
        self.attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            out_dim=4 * dim,
            qk_norm="rms_norm",
            eps=1e-5,
            elementwise_affine=False,
        )
        self.attn_context = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            out_dim=4 * caption_dim if update_captions else caption_dim,
            qk_norm="rms_norm",
            eps=1e-5,
            elementwise_affine=False,
        )

        self.ff = FeedForward(dim, mult=4, activation_fn=activation_fn)
        self.ff_context = FeedForward(caption_dim, mult=4, activation_fn=activation_fn)
    
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor, image_rotary_emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

        if self.update_captions:
            norm_encoder_hidden_states, enc_gate_msa, enc_scale_mlp, enc_gate_mlp = self.norm_context1(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states = self.norm_context1(encoder_hidden_states)
        
        attn_hidden_states = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
        )
        attn_encoder_hidden_states = self.attn_context(
            hidden_states=norm_encoder_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=None,
        )


@maybe_allow_in_graph
class MochiTransformer3D(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        caption_dim=1536,
        mlp_ratio_x=4.0,
        mlp_ratio_y=4.0,
        in_channels=12,
        qk_norm=True,
        qkv_bias=False,
        out_bias=True,
        timestep_mlp_bias=True,
        timestep_scale=1000.0,
        text_embed_dim=4096,
        activation_fn: str = "swiglu",
        max_sequence_length=256,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
        )

        self.caption_embedder = MochiAttentionPool(num_attention_heads=8, embed_dim=text_embed_dim, output_dim=inner_dim)
        self.caption_proj = nn.Linear(text_embed_dim, caption_dim)

        self.pos_frequencies = nn.Parameter(
            torch.empty(3, num_attention_heads, attention_head_dim // 2)
        )

        self.transformer_blocks = nn.ModuleList([
            MochiTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                caption_dim=caption_dim,
                activation_fn=activation_fn,
                update_captions=i < num_layers - 1,
            )
            for i in range(num_layers)
        ])
