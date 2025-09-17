# Copyright 2025 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
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
from ..attention import LuminaFeedForward
from ..attention_processor import Attention, LuminaAttnProcessor2_0
from ..embeddings import (
    LuminaCombinedTimestepCaptionEmbedding,
    LuminaPatchEmbed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import LuminaLayerNormContinuous, LuminaRMSNormZero, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LuminaNextDiTBlock(nn.Module):
    """
    A LuminaNextDiTBlock for LuminaNextDiT2DModel.

    Parameters:
        dim (`int`): Embedding dimension of the input features.
        num_attention_heads (`int`): Number of attention heads.
        num_kv_heads (`int`):
            Number of attention heads in key and value features (if using GQA), or set to None for the same as query.
        multiple_of (`int`): The number of multiple of ffn layer.
        ffn_dim_multiplier (`float`): The multiplier factor of ffn layer dimension.
        norm_eps (`float`): The eps for norm layer.
        qk_norm (`bool`): normalization for query and key.
        cross_attention_dim (`int`): Cross attention embedding dimension of the input text prompt hidden_states.
        norm_elementwise_affine (`bool`, *optional*, defaults to True),
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        cross_attention_dim: int,
        norm_elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads

        self.gate = nn.Parameter(torch.zeros([num_attention_heads]))

        # Self-attention
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="layer_norm_across_heads" if qk_norm else None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.attn1.to_out = nn.Identity()

        # Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            qk_norm="layer_norm_across_heads" if qk_norm else None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )

        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=int(4 * 2 * dim / 3),
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.norm1 = LuminaRMSNormZero(
            embedding_dim=dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=norm_elementwise_affine,
        )
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm1_context = RMSNorm(cross_attention_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        temb: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Parameters:
            hidden_states (`torch.Tensor`): The input of hidden_states for LuminaNextDiTBlock.
            attention_mask (`torch.Tensor): The input of hidden_states corresponse attention mask.
            image_rotary_emb (`torch.Tensor`): Precomputed cosine and sine frequencies.
            encoder_hidden_states: (`torch.Tensor`): The hidden_states of text prompt are processed by Gemma encoder.
            encoder_mask (`torch.Tensor`): The hidden_states of text prompt attention mask.
            temb (`torch.Tensor`): Timestep embedding with text prompt embedding.
            cross_attention_kwargs (`Dict[str, Any]`): kwargs for cross attention.
        """
        residual = hidden_states

        # Self-attention
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        self_attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=image_rotary_emb,
            **cross_attention_kwargs,
        )

        # Cross-attention
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        cross_attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=encoder_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=None,
            **cross_attention_kwargs,
        )
        cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
        mixed_attn_output = self_attn_output + cross_attn_output
        mixed_attn_output = mixed_attn_output.flatten(-2)
        # linear proj
        hidden_states = self.attn2.to_out[0](mixed_attn_output)

        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(hidden_states)

        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))

        hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)

        return hidden_states


class LuminaNextDiT2DModel(ModelMixin, ConfigMixin):
    """
    LuminaNextDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        in_channels (`int`, *optional*, defaults to 4):
            The number of input channels for the model. Typically, this matches the number of channels in the input
            images.
        hidden_size (`int`, *optional*, defaults to 4096):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        num_layers (`int`, *optional*, default to 32):
            The number of layers in the model. This defines the depth of the neural network.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in each attention layer. This parameter specifies how many separate attention
            mechanisms are used.
        num_kv_heads (`int`, *optional*, defaults to 8):
            The number of key-value heads in the attention mechanism, if different from the number of attention heads.
            If None, it defaults to num_attention_heads.
        multiple_of (`int`, *optional*, defaults to 256):
            A factor that the hidden size should be a multiple of. This can help optimize certain hardware
            configurations.
        ffn_dim_multiplier (`float`, *optional*):
            A multiplier for the dimensionality of the feed-forward network. If None, it uses a default value based on
            the model configuration.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            A small value added to the denominator for numerical stability in normalization layers.
        learn_sigma (`bool`, *optional*, defaults to True):
            Whether the model should learn the sigma parameter, which might be related to uncertainty or variance in
            predictions.
        qk_norm (`bool`, *optional*, defaults to True):
            Indicates if the queries and keys in the attention mechanism should be normalized.
        cross_attention_dim (`int`, *optional*, defaults to 2048):
            The dimensionality of the text embeddings. This parameter defines the size of the text representations used
            in the model.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            A scaling factor applied to certain parameters or layers in the model. This can be used for adjusting the
            overall scale of the model's operations.
    """

    _skip_layerwise_casting_patterns = ["patch_embedder", "norm", "ffn_norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: Optional[int] = 2,
        in_channels: Optional[int] = 4,
        hidden_size: Optional[int] = 2304,
        num_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_kv_heads: Optional[int] = None,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: Optional[float] = 1e-5,
        learn_sigma: Optional[bool] = True,
        qk_norm: Optional[bool] = True,
        cross_attention_dim: Optional[int] = 2048,
        scaling_factor: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling_factor = scaling_factor

        self.patch_embedder = LuminaPatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True
        )

        self.pad_token = nn.Parameter(torch.empty(hidden_size))

        self.time_caption_embed = LuminaCombinedTimestepCaptionEmbedding(
            hidden_size=min(hidden_size, 1024), cross_attention_dim=cross_attention_dim
        )

        self.layers = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )
        # self.final_layer = LuminaFinalLayer(hidden_size, patch_size, self.out_channels)

        assert (hidden_size // num_attention_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict=True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        """
        Forward pass of LuminaNextDiT.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
        """
        hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(hidden_states, image_rotary_emb)
        image_rotary_emb = image_rotary_emb.to(hidden_states.device)

        temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)

        encoder_mask = encoder_mask.bool()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                mask,
                image_rotary_emb,
                encoder_hidden_states,
                encoder_mask,
                temb=temb,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)

        # unpatchify
        height_tokens = width_tokens = self.patch_size
        height, width = img_size[0]
        batch_size = hidden_states.size(0)
        sequence_length = (height // height_tokens) * (width // width_tokens)
        hidden_states = hidden_states[:, :sequence_length].view(
            batch_size, height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels
        )
        output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
