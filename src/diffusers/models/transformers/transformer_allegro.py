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

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_torch_version, logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_processor import AllegroAttnProcessor2_0, Attention
from ..embeddings import PatchEmbed, PixArtAlphaTextProjection
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle


logger = logging.get_logger(__name__)


@maybe_allow_in_graph
class AllegroTransformerBlock(nn.Module):
    r"""
    Transformer block used in [Allegro](https://github.com/rhymes-ai/Allegro) model.

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        cross_attention_dim (`int`, defaults to `2304`):
            The dimension of the cross attention features.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        only_cross_attention (`bool`, defaults to `False`):
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
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
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
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
            cross_attention_dim=None,
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
            processor=AllegroAttnProcessor2_0(),
        )

        # 3. Feed Forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
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
    A 3D Transformer model for video-like data.

    Args:
        patch_size (`int`, defaults to `2`):
            The size of spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of temporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `96`):
            The number of channels in each head.
        in_channels (`int`, defaults to `4`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `4`):
            The number of channels in the output.
        num_layers (`int`, defaults to `32`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        cross_attention_dim (`int`, defaults to `2304`):
            The dimension of the cross attention features.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_height (`int`, defaults to `90`):
            The height of the input latents.
        sample_width (`int`, defaults to `160`):
            The width of the input latents.
        sample_frames (`int`, defaults to `22`):
            The number of frames in the input latents.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value to use in normalization layers.
        caption_channels (`int`, defaults to `4096`):
            Number of channels to use for projecting the caption embeddings.
        interpolation_scale_h (`float`, defaults to `2.0`):
            Scaling factor to apply in 3D positional embeddings across height dimension.
        interpolation_scale_w (`float`, defaults to `2.0`):
            Scaling factor to apply in 3D positional embeddings across width dimension.
        interpolation_scale_t (`float`, defaults to `2.2`):
            Scaling factor to apply in 3D positional embeddings across time dimension.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: int = 1,
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
        self.pos_embed = PatchEmbed(
            height=sample_height,
            width=sample_width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_type=None,
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
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, use_additional_conditions=False)

        # 5. Caption projection
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=self.inner_dim)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t = self.config.patch_size_t
        p = self.config.patch_size

        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

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
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Timestep embeddings
        timestep, embedded_timestep = self.adaln_single(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        # 2. Patch embeddings
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.pos_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, encoder_hidden_states.shape[-1])

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # TODO(aryan): Implement gradient checkpointing
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    attention_mask,
                    encoder_attention_mask,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=timestep,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

        # 4. Output normalization & projection
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)

        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p, p, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
