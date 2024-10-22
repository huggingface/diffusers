# Copyright 2024 The RhymesAI and The HuggingFace Team.
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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_processor import (
    AllegroAttnProcessor2_0,
    Attention,
)
from ..embeddings import PixArtAlphaTextProjection
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AllegroAdaLayerNormSingle


logger = logging.get_logger(__name__)


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        use_abs_pos=False,
    ):
        super().__init__()
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size

    def forward(self, latent):
        b, _, _, _, _ = latent.shape
        video_latent = None

        latent = rearrange(latent, "b c t h w -> (b t) c h w")

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        latent = rearrange(latent, "(b t) n c -> b (t n) c", b=b)
        video_latent = latent

        return video_latent


@maybe_allow_in_graph
class AllegroTransformerBlock(nn.Module):
    r"""
    TODO(aryan): docs
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            processor=AllegroAttnProcessor2_0(),
        )

        # 2. Cross Attention
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            processor=AllegroAttnProcessor2_0(),
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed Forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Scale-shift
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + temb.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = hidden_states

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                image_rotary_emb=None,
            )
            hidden_states = attn_output + hidden_states

        # 2. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        # TODO(aryan): maybe following line is not required
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class AllegroTransformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_temporal: int = 1,
        num_attention_heads: int = 24,
        attention_head_dim: int = 96,
        in_channels: int = 4,
        out_channels: int = 4,
        num_layers: int = 32,
        dropout: float = 0.0,
        cross_attention_dim: int = 2304,
        attention_bias: bool = True,
        sample_height: int = 90,
        sample_width: int = 160,
        sample_frames: int = 22,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        interpolation_scale_h: float = 2.0,
        interpolation_scale_w: float = 2.0,
        interpolation_scale_t: float = 2.2,
    ):
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim

        interpolation_scale_t = (
            interpolation_scale_t
            if interpolation_scale_t is not None
            else ((sample_frames - 1) // 16 + 1)
            if sample_frames % 2 == 1
            else sample_frames // 16
        )
        interpolation_scale_h = interpolation_scale_h if interpolation_scale_h is not None else sample_height / 30
        interpolation_scale_w = interpolation_scale_w if interpolation_scale_w is not None else sample_width / 40

        # 1. Patch embedding
        self.pos_embed = PatchEmbed2D(
            height=sample_height,
            width=sample_width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            # pos_embed_type=None,
        )

        # 2. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                AllegroTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Output projection & norm
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

        # 4. Timestep embeddings
        self.adaln_single = AllegroAdaLayerNormSingle(self.inner_dim, use_additional_conditions=False)

        # 5. Caption projection
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=self.inner_dim)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t = self.config.patch_size_temporal
        p = self.config.patch_size

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)        attention_mask_vid, attention_mask_img = None, None
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame+use_image_num, h, w -> a video with images
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(hidden_states.dtype)
            attention_mask = attention_mask[:, :num_frames]  # [batch_size, num_frames, height, width]

            if attention_mask.numel() > 0:
                attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, num_frames, height, width]
                attention_mask = F.max_pool3d(attention_mask, kernel_size=(p_t, p, p), stride=(p_t, p, p))
                attention_mask = attention_mask.flatten(1).view(batch_size, 1, -1)

            attention_mask = (
                (1 - attention_mask.bool().to(hidden_states.dtype)) * -10000.0 if attention_mask.numel() > 0 else None
            )

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:
            # b, 1+use_image_num, l -> a video with images
            # b, 1, l -> only images
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            encoder_attention_mask = (
                rearrange(encoder_attention_mask, "b 1 l -> (b 1) 1 l") if encoder_attention_mask.numel() > 0 else None
            )

        # 1. Input
        post_patch_num_frames = num_frames // self.config.patch_size_temporal
        post_patch_height = height // self.config.patch_size
        post_patch_width = width // self.config.patch_size

        timestep, embedded_timestep = self.adaln_single(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        hidden_states = self.pos_embed(
            hidden_states
        )  # TODO(aryan): remove dtype conversion here and move to pipeline if needed

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, encoder_hidden_states.shape[-1])

        for i, block in enumerate(self.transformer_blocks):
            # TODO(aryan): Implement gradient checkpointing
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        # 3. Output
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)

        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p, p, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
