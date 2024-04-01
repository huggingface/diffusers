# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward, _chunked_feed_forward
from ..attention_processor import Attention
from ..embeddings import PatchEmbed3D, PixArtAlphaTextProjection, get_1d_sincos_pos_embed_from_grid
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer3DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


@maybe_allow_in_graph
class Transformer3DBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        num_embeds_ada_norm: Optional[int] = None,
        norm_eps: float = 1e-6,
        num_temporal_patches: int = 16,
        num_spatial_patches: int = 256,
    ):
        super().__init__()
        # We keep these boolean flags for backward-compatibility.

        self.use_ada_layer_norm_single = True
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.num_temporal_patches = num_temporal_patches
        self.num_spatial_patches = num_spatial_patches

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Spatial Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=True,
        )

        # 2. Temporal Self-Attn
        self.attn_temporal = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=True,
        )

        # 2. Cross-Attn
        # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
        # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
        # the second cross attention block.
        self.norm2 = nn.LayerNorm(dim, norm_eps, elementwise_affine=False)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=True,
        )

        # 3. Feed-forward
        self.ff = FeedForward(dim, activation_fn="gelu-approximate")

        # 4. Scale-shift for PixArt-Alpha.
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        temporal_pos_embed: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 1. Spatial Self-Attention
        # reshape (batch, num_temporal_patches*num_spatial_patches, dim) -> (batch * num_temporal_patches, num_spatial_patches, dim)
        norm_hidden_states = norm_hidden_states.view(
            batch_size, self.num_temporal_patches, self.num_spatial_patches, -1
        )
        norm_hidden_states = norm_hidden_states.view(
            batch_size * self.num_temporal_patches, self.num_spatial_patches, -1
        )

        attn_output = self.attn1(norm_hidden_states)

        # reshape (batch * num_temporal_patches, num_spatial_patches, dim) -> (batch, num_temporal_patches*num_spatial_patches, dim)
        attn_output = attn_output.view(batch_size, self.num_temporal_patches, self.num_spatial_patches, -1)
        attn_output = attn_output.view(batch_size, self.num_temporal_patches * self.num_spatial_patches, -1)

        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2. Temporal Self-Attention
        # reshape (batch, num_temporal_patches*num_spatial_patches, dim) -> (batch * num_spatial_patches, num_temporal_patches, dim)
        temporal_hidden_states = hidden_states.view(
            batch_size, self.num_temporal_patches, self.num_spatial_patches, -1
        ).transpose(1, 2)
        temporal_hidden_states = temporal_hidden_states.view(
            batch_size * self.num_spatial_patches, self.num_temporal_patches, -1
        )

        if temporal_pos_embed is not None:
            temporal_hidden_states = temporal_hidden_states + temporal_pos_embed

        attn_output = self.attn_temporal(temporal_hidden_states)

        # reshape (batch * num_spatial_patches, num_temporal_patches, dim) -> (batch, num_temporal_patches*num_spatial_patches, dim)
        attn_output = (
            attn_output.view(batch_size, self.num_spatial_patches, self.num_temporal_patches, -1)
            .transpose(1, 2)
            .contiguous()
        )
        attn_output = attn_output.view(batch_size, self.num_temporal_patches * self.num_spatial_patches, -1)

        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Cross-Attention
        # For PixArt norm2 isn't applied here:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
        norm_hidden_states = hidden_states
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin):
    """
    A 3D Transformer model for image-like data.

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
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        sample_size: Tuple[int] = (2, 4, 4),
        patch_size: Tuple[int] = (1, 2, 2),
        in_channels: int = 4,
        out_channels: int = 8,
        num_layers: int = 1,
        cross_attention_dim: int = 256,
        num_embeds_ada_norm: int = 1000,
        norm_eps: float = 1e-6,
        caption_channels: int = 256,
        interpolation_scale: float = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 2. Define input layers
        self.height = sample_size[1]
        self.width = sample_size[2]
        self.num_patches = np.prod([sample_size[i] // patch_size[i] for i in range(3)])
        self.num_temporal_patches = sample_size[0] // patch_size[0]
        self.num_spatial_patches = self.num_patches // self.num_temporal_patches

        self.patch_size = patch_size
        interpolation_scale = (
            interpolation_scale if interpolation_scale is not None else max(self.config.sample_size[1] // 64, 1)
        )
        self.pos_embed = PatchEmbed3D(
            height=self.height,
            width=self.width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Transformer3DBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    norm_eps=norm_eps,
                    num_temporal_patches=self.num_temporal_patches,
                    num_spatial_patches=self.num_spatial_patches,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.proj_out = nn.Linear(inner_dim, np.prod(patch_size) * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = AdaLayerNormSingle(inner_dim)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        temporal_pos_embed = get_1d_sincos_pos_embed(inner_dim, self.num_temporal_patches)
        temporal_pos_embed = torch.from_numpy(temporal_pos_embed).float().unsqueeze(0).requires_grad_(False)
        self.register_buffer("pos_embed_temporal", temporal_pos_embed)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        height, width = hidden_states.shape[-2] // self.patch_size[1], hidden_states.shape[-1] // self.patch_size[2]
        hidden_states = self.pos_embed(hidden_states)

        batch_size = hidden_states.shape[0]
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        # 2. Blocks
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                temporal_pos_embed=self.pos_embed_temporal if i == 0 else None,
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
            shape=(
                -1,
                self.num_temporal_patches,
                height,
                width,
                self.patch_size[1],
                self.patch_size[2],
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nthwpqc->ncthpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                self.num_temporal_patches,
                height * self.patch_size[1],
                width * self.patch_size[2],
            )
        )

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
