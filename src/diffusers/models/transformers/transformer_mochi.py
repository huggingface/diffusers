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
from ..attention import Attention, FeedForward, JointAttnProcessor2_0
from ..embeddings import PatchEmbed, MochiCombinedTimestepCaptionEmbedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, MochiRMSNormZero, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class MochiTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        pooled_projection_dim: int,
        qk_norm: str = "rms_norm",
        activation_fn: str = "swiglu",
        context_pre_only: bool = True,
    ) -> None:
        super().__init__()

        self.context_pre_only = context_pre_only
        
        self.norm1 = MochiRMSNormZero(dim, 4 * dim)

        if context_pre_only:
            self.norm1_context = MochiRMSNormZero(dim, 4 * pooled_projection_dim)
        else:
            self.norm1_context = RMSNorm(pooled_projection_dim, eps=1e-6, elementwise_affine=False)
        
        self.attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            out_dim=4 * dim,
            qk_norm=qk_norm,
            eps=1e-6,
            elementwise_affine=False,
            processor=JointAttnProcessor2_0(),
        )

        self.norm2 = RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2_context = RMSNorm(pooled_projection_dim, eps=1e-6, elementwise_affine=False)
        
        self.norm3 = RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm3_context = RMSNorm(pooled_projection_dim, eps=1e-56, elementwise_affine=False)

        self.ff = FeedForward(dim, mult=4, activation_fn=activation_fn)
        self.ff_context = FeedForward(pooled_projection_dim, mult=4, activation_fn=activation_fn)

        self.norm4 = RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm4_context = RMSNorm(pooled_projection_dim, eps=1e-56, elementwise_affine=False)
    
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor, image_rotary_emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

        if self.context_pre_only:
            norm_encoder_hidden_states, enc_gate_msa, enc_scale_mlp, enc_gate_mlp = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        
        attn_hidden_states, context_attn_hidden_states = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + self.norm2(attn_hidden_states) * torch.tanh(gate_msa).unsqueeze(1)
        hidden_states = self.norm3(hidden_states) * (1 + scale_mlp.unsqueeze(1))
        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + self.norm2_context(context_attn_hidden_states) * torch.tanh(enc_gate_msa).unsqueeze(1)
            encoder_hidden_states = encoder_hidden_states + self.norm3_context(encoder_hidden_states) * (1 + enc_scale_mlp.unsqueeze(1))
        
        ff_output = self.ff(hidden_states)
        context_ff_output = self.ff_context(encoder_hidden_states)
        
        hidden_states = hidden_states + ff_output * torch.tanh(gate_mlp).unsqueeze(1)
        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + context_ff_output * torch.tanh(enc_gate_mlp).unsqueeze(0)
        
        return hidden_states, encoder_hidden_states


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
        pooled_projection_dim: int = 1536,
        in_channels=12,
        out_channels: Optional[int] = None,
        qk_norm: str = "rms_norm",
        timestep_mlp_bias=True,
        timestep_scale=1000.0,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 256,
        activation_fn: str = "swiglu",
        max_sequence_length: int = 256,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.time_embed = MochiCombinedTimestepCaptionEmbedding(
            embedding_dim=text_embed_dim,
            pooled_projection_dim=pooled_projection_dim,
            time_embed_dim=time_embed_dim,
            num_attention_heads=8,
        )
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
        )

        self.pos_frequencies = nn.Parameter(
            torch.empty(3, num_attention_heads, attention_head_dim // 2)
        )

        self.transformer_blocks = nn.ModuleList([
            MochiTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                pooled_projection_dim=pooled_projection_dim,
                qk_norm=qk_norm,
                activation_fn=activation_fn,
                context_pre_only=i < num_layers - 1,
            )
            for i in range(num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6, norm_type="layer_norm")
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size

        post_patch_height = height // p
        post_patch_width = width // p

        temb, caption_proj = self.time_embed(timestep, encoder_hidden_states, encoder_attention_mask)

        hidden_states = self.patch_embed(hidden_states)

        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        
        # TODO(aryan): do something with self.pos_frequencies

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_height, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
