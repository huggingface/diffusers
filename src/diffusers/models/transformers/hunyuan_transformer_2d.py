# Copyright 2024 HunyuanDiT Authors and The HuggingFace Team. All rights reserved.
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
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_processor import Attention, HunyuanAttnProcessor2_0
from ..embeddings import PixArtAlphaTextProjection, PatchEmbed, TimestepEmbedding, Timesteps, get_timestep_embedding, HunyuanCombinedTimestepTextSizeStyleEmbedding
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous
from ..modeling_outputs import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps
        ).to(origin_dtype)


class AdaLayerNormShift(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        shift = self.linear(self.silu(emb.to(torch.float32)).to(emb.dtype))
        x = self.norm(x) + shift.unsqueeze(dim=1)
        return x



@maybe_allow_in_graph
class HunyuanDiTBlock(nn.Module):
    r"""
    Parameters:
    HunyuanDiT Transformer block. Allow skip connection and QKNorm
        dim (`int`): The number of channels in the input and output. num_attention_heads (`int`): The number of heads
        to use for multi-head attention. attention_head_dim (`int`): The number of channels in each head. dropout
        (`float`, *optional*, defaults to 0.0): The dropout probability to use. cross_attention_dim (`int`,
        *optional*): The size of the encoder_hidden_states vector for cross attention. activation_fn (`str`,
        *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward. num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: int = 1024,
        dropout=0.0,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=HunyuanAttnProcessor2_0(),
        )

        # 2. Cross-Attn
        self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=HunyuanAttnProcessor2_0(),
        )
        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
        skip=None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([hidden_states, skip], dim=-1)
            cat = self.skip_norm(cat)
            hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states, temb)  ### checked: self.norm1 is correct
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

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states


class HunyuanDiT2DModel(ModelMixin, ConfigMixin):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters ---------- args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        sample_size=32,
        hidden_size=1152,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        cross_attention_dim: int = 1024,
        norm_type: str = "layer_norm",
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
    ):
        super().__init__()
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim

        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim_t5,
            hidden_size=cross_attention_dim_t5 * 4,
            out_features=cross_attention_dim,
            act_fn="silu_fp32",
        )

        self.text_embedding_padding = nn.Parameter(
            torch.randn(text_len + text_len_t5, cross_attention_dim, dtype=torch.float32)
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
        )

        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            seq_len=text_len_t5,
            cross_attention_dim=cross_attention_dim_t5,
        )

        # HunyuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunyuanDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    activation_fn=activation_fn,
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    skip=layer > num_layers // 2,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        return_dict=True,
    ):
        """
        Forward pass of the encoder.

        Args:
        hidden_states: torch.Tensor (B, D, H, W)
        timestep: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        
        temb = self.time_extra_emb(timestep, encoder_hidden_states_t5, image_meta_size, style, hidden_dtype=timestep.dtype)  # [B, D]

        # text projection
        batch_size, sequence_length, _ = encoder_hidden_states_t5.shape 
        encoder_hidden_states_t5 = self.text_embedder(encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1]))
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, sequence_length, -1)
        
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, encoder_hidden_states_t5], dim=1
        ) 
        text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
        text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()

        encoder_hidden_states = torch.where(
            text_embedding_mask, encoder_hidden_states, self.text_embedding_padding
        )

        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.config.num_layers // 2:
                skip = skips.pop()
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                )  # (N, L, D)
            else:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )  # (N, L, D)

            if layer < (self.config.num_layers // 2 - 1):
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
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
