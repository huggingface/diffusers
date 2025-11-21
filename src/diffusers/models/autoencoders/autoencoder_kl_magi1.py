# Copyright 2025 The Sand AI Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def resize_pos_embed(posemb, src_shape, target_shape):
    posemb = posemb.reshape(1, src_shape[0], src_shape[1], src_shape[2], -1)
    posemb = posemb.permute(0, 4, 1, 2, 3)
    posemb = nn.functional.interpolate(posemb, size=target_shape, mode="trilinear", align_corners=False)
    posemb = posemb.permute(0, 2, 3, 4, 1)
    posemb = posemb.reshape(1, target_shape[0] * target_shape[1] * target_shape[2], -1)
    return posemb


# Copied from diffusers.models.transformers.transformer_wan._get_qkv_projections
def _get_qkv_projections(attn: "Magi1VAEAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


class Magi1VAELayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(Magi1VAELayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)

        x_normalized = (x - mean) / (std + self.eps)

        return x_normalized


class Magi1VAEAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Magi1VAEAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: "Magi1VAEAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, time_height_width, channels = hidden_states.size()

        query, key, value = _get_qkv_projections(attn, hidden_states, None)

        qkv = torch.cat((query, key, value), dim=2)
        qkv = qkv.reshape(batch_size, time_height_width, 3, attn.heads, channels // attn.heads)
        qkv = attn.qkv_norm(qkv)
        query, key, value = qkv.chunk(3, dim=2)

        # Remove the extra dimension from chunking
        # Shape: (batch_size, time_height_width, num_heads, head_dim)
        query = query.squeeze(2)
        key = key.squeeze(2)
        value = value.squeeze(2)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class Magi1VAEAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Magi1VAEAttnProcessor
    _available_processors = [Magi1VAEAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.qkv_norm = Magi1VAELayerNorm(dim // heads, elementwise_affine=False)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.set_processor(processor)

    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.fuse_projections
    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.unfuse_projections
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask, rotary_emb, **kwargs)


class Magi1VAETransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 4 * 1024,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = Magi1VAEAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=Magi1VAEAttnProcessor(),
        )

        self.norm = nn.LayerNorm(dim)
        self.proj_out = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu")

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states + self.attn(hidden_states)
        hidden_states = hidden_states + self.proj_out(self.norm(hidden_states))
        return hidden_states


class Magi1Encoder3d(nn.Module):
    def __init__(
        self,
        inner_dim=128,
        z_dim=4,
        patch_size: Tuple[int] = (1, 2, 2),
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_attention_heads: int = 40,
        ffn_dim: int = 4 * 1024,
        num_layers: int = 24,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.height = height
        self.width = width
        self.num_frames = num_frames

        # 1. Patch & position embedding
        self.patch_embedding = nn.Conv3d(3, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

        self.cls_token_nums = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, inner_dim))
        # `generator` as a parameter?
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        p_t, p_h, p_w = patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        num_patches = post_patch_num_frames * post_patch_height * post_patch_width

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.cls_token_nums, inner_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Magi1VAETransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    ffn_dim,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # output blocks
        self.norm_out = nn.LayerNorm(inner_dim)
        self.linear_out = nn.Linear(inner_dim, z_dim * 2)

        # `generator` as a parameter?
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.gradient_checkpointing = False

    def forward(self, x):
        B = x.shape[0]
        # B C T H W -> B C T/pT H/pH W//pW
        x = self.patch_embedding(x)
        latentT, latentH, latentW = x.shape[2], x.shape[3], x.shape[4]
        # B C T/pT H/pH W//pW -> B (T/pT H/pH W//pW) C
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if latentT != self.patch_size[0] or latentH != self.patch_size[1] or latentW != self.patch_size[2]:
            pos_embed = resize_pos_embed(
                self.pos_embed[:, 1:, :],
                src_shape=(
                    self.num_frames // self.patch_size[0],
                    self.height // self.patch_size[1],
                    self.width // self.patch_size[2],
                ),
                target_shape=(latentT, latentH, latentW),
            )
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :], pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        ## transformer blocks
        for block in self.blocks:
            x = block(x)

        ## head
        x = self.norm_out(x)
        x = x[:, 1:]  # remove cls_token
        x = self.linear_out(x)

        # B L C - > B , lT, lH, lW, zC (where zC is now z_dim * 2)
        x = x.reshape(B, latentT, latentH, latentW, self.z_dim * 2)

        # B , lT, lH, lW, zC -> B, zC, lT, lH, lW
        x = x.permute(0, 4, 1, 2, 3)

        return x


class Magi1Decoder3d(nn.Module):
    def __init__(
        self,
        inner_dim=1024,
        z_dim=16,
        patch_size: Tuple[int] = (4, 8, 8),
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_attention_heads: int = 16,
        ffn_dim: int = 4 * 1024,
        num_layers: int = 24,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.height = height
        self.width = width
        self.num_frames = num_frames

        # init block
        self.proj_in = nn.Linear(z_dim, inner_dim)

        self.cls_token_nums = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, inner_dim))
        # `generator` as a parameter?
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        p_t, p_h, p_w = patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        num_patches = post_patch_num_frames * post_patch_height * post_patch_width

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.cls_token_nums, inner_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Magi1VAETransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    ffn_dim,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # output blocks
        self.norm_out = nn.LayerNorm(inner_dim)
        self.unpatch_channels = inner_dim // (patch_size[0] * patch_size[1] * patch_size[2])
        self.conv_out = nn.Conv3d(self.unpatch_channels, 3, 3, padding=1)

        # `generator` as a parameter?
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.gradient_checkpointing = False

    def forward(self, x):
        B, C, latentT, latentH, latentW = x.shape
        x = x.permute(0, 2, 3, 4, 1)

        x = x.reshape(B, -1, C)

        x = self.proj_in(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if latentT != self.patch_size[0] or latentH != self.patch_size[1] or latentW != self.patch_size[2]:
            pos_embed = resize_pos_embed(
                self.pos_embed[:, 1:, :],
                src_shape=(
                    self.num_frames // self.patch_size[0],
                    self.height // self.patch_size[1],
                    self.width // self.patch_size[2],
                ),
                target_shape=(latentT, latentH, latentW),
            )
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :], pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        ## transformer blocks
        for block in self.blocks:
            x = block(x)

        ## head
        x = self.norm_out(x)
        x = x[:, 1:]  # remove cls_token

        x = x.reshape(
            B,
            latentT,
            latentH,
            latentW,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            self.unpatch_channels,
        )
        # Rearrange from (B, lT, lH, lW, pT, pH, pW, C) to (B, C, lT*pT, lH*pH, lW*pW)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (B, C, lT, pT, lH, pH, lW, pW)
        x = x.reshape(
            B,
            self.unpatch_channels,
            latentT * self.patch_size[0],
            latentH * self.patch_size[1],
            latentW * self.patch_size[2],
        )

        x = self.conv_out(x)
        return x


class AutoencoderKLMagi1(ModelMixin, ConfigMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Magi1](https://arxiv.org/abs/2505.13211).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False
    _skip_layerwise_casting_patterns = ["patch_embedding", "norm"]
    _no_split_modules = ["Magi1VAETransformerBlock"]
    # _keep_in_fp32_modules = ["qkv_norm", "norm1", "norm2"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (4, 8, 8),
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        z_dim: int = 16,
        height: int = 256,
        width: int = 256,
        num_frames: int = 16,
        ffn_dim: int = 4 * 1024,
        num_layers: int = 24,
        eps: float = 1e-6,
        temporal_compression_ratio: Optional[int] = None,
        spatial_compression_ratio: Optional[int] = None,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.z_dim = z_dim

        self.encoder = Magi1Encoder3d(
            inner_dim,
            z_dim,
            patch_size,
            num_frames,
            height,
            width,
            num_attention_heads,
            ffn_dim,
            num_layers,
            eps,
        )

        self.decoder = Magi1Decoder3d(
            inner_dim,
            z_dim,
            patch_size,
            num_frames,
            height,
            width,
            num_attention_heads,
            ffn_dim,
            num_layers,
            eps,
        )

        # Use provided compression ratios if given, otherwise compute from patch_size
        self.spatial_compression_ratio = (
            spatial_compression_ratio if spatial_compression_ratio is not None else (patch_size[1] or patch_size[2])
        )
        self.temporal_compression_ratio = (
            temporal_compression_ratio if temporal_compression_ratio is not None else patch_size[0]
        )

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal tile length for temporal tiling to be used
        self.tile_sample_min_length = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # The minimal distance between two temporal tiles
        self.tile_sample_stride_length = 16

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_length: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_length: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_length = tile_sample_min_length or self.tile_sample_min_length
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_length = tile_sample_stride_length or self.tile_sample_stride_length

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor):
        _, _, num_frames, height, width = x.shape

        if self.use_tiling and (
            width > self.tile_sample_min_width
            or height > self.tile_sample_min_height
            or num_frames > self.tile_sample_min_length
        ):
            return self.tiled_encode(x)

        out = self.encoder(x)

        return out

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_length = self.tile_sample_min_length // self.temporal_compression_ratio

        if self.use_tiling and (
            width > tile_latent_min_width or height > tile_latent_min_height or num_frames > tile_latent_min_length
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        out = self.decoder(z)

        if not return_dict:
            return (out,)

        return DecoderOutput(sample=out)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, power: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            w_a, w_b = 1 - y / blend_extent, y / blend_extent
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (w_a**power) + b[:, :, :, y, :] * (w_b**power)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, power: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            w_a, w_b = 1.0 - x / blend_extent, x / blend_extent
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (w_a**power) + b[:, :, :, :, x] * (w_b**power)
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, power: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for t in range(blend_extent):
            w_a, w_b = 1.0 - t / blend_extent, t / blend_extent
            b[:, :, t, :, :] = a[:, :, -blend_extent + t, :, :] * (w_a**power) + b[:, :, t, :, :] * (w_b**power)
        return b

    def _encode_tile(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single tile.
        """
        N, C, T, H, W = x.shape

        if T == 1 and self.temporal_compression_ratio > 1:
            x = x.expand(-1, -1, self.temporal_compression_ratio, -1, -1)
            x = self.encoder(x)
            # After temporal expansion and encoding, select only the first temporal frame.
            x = x[:, :, :1]
            return x
        else:
            x = self.encoder(x)
            return x

    def _decode_tile(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a single tile.
        """
        N, C, T, H, W = x.shape

        x = self.decoder(x)
        return x[:, :, :1, :, :] if T == 1 else x

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        B, C, num_frames, height, width = x.shape

        # Latent sizes after compression
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio
        latent_length = num_frames // self.temporal_compression_ratio

        # Tile latent sizes / strides
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_length = self.tile_sample_min_length // self.temporal_compression_ratio

        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_latent_stride_length = self.tile_sample_stride_length // self.temporal_compression_ratio

        # Overlap (blend) sizes in latent space
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width
        blend_length = tile_latent_min_length - tile_latent_stride_length

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        times = []
        for t in range(0, num_frames, self.tile_sample_stride_length):
            rows = []
            for i in range(0, height, self.tile_sample_stride_height):
                row = []
                for j in range(0, width, self.tile_sample_stride_width):
                    tile = x[
                        :,
                        :,
                        t : t + self.tile_sample_min_length,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                    h_tile = self._encode_tile(tile)
                    # Original implementation samples here and blends the latents.
                    # Instead we're keeping moments (mu and logvar) and blend them.
                    row.append(h_tile)
                rows.append(row)
            times.append(rows)

        # Calculate global blending order because blending is not commutative here.
        nT = len(times)
        nH = len(times[0]) if nT else 0
        nW = len(times[0][0]) if nH else 0
        idx_numel = []
        for tt in range(nT):
            for ii in range(nH):
                for jj in range(nW):
                    idx_numel.append(((tt, ii, jj), times[tt][ii][jj].numel()))
        global_order = [idx for (idx, _) in sorted(idx_numel, key=lambda kv: kv[1], reverse=True)]

        result_grid = [[[None for _ in range(nW)] for _ in range(nH)] for _ in range(nT)]
        for t_idx, i_idx, j_idx in global_order:
            rows = times[t_idx]
            row = rows[i_idx]
            h = row[j_idx]

            # Separate the mu and the logvar because mu needs to be blended linearly
            # but logvar needs to be blended quadratically to obtain numerical equivalence
            # so that the overall distribution is preserved
            mu, logvar = h[:, : self.z_dim], h[:, self.z_dim :]
            var = logvar.exp()

            # Blend the prev tile, the above tile and the left tile
            # to the current tile and add the current tile to the result grid
            if t_idx > 0:
                h_tile = times[t_idx - 1][i_idx][j_idx]
                mu_prev, logvar_prev = h_tile[:, : self.z_dim], h_tile[:, self.z_dim :]
                var_prev = logvar_prev.exp()
                mu = self.blend_t(mu_prev, mu, blend_length, power=1)
                var = self.blend_t(var_prev, var, blend_length, power=2)

            if i_idx > 0:
                h_tile = rows[i_idx - 1][j_idx]
                mu_up, logvar_up = h_tile[:, : self.z_dim], h_tile[:, self.z_dim :]
                var_up = logvar_up.exp()
                mu = self.blend_v(mu_up, mu, blend_height, power=1)
                var = self.blend_v(var_up, var, blend_height, power=2)

            if j_idx > 0:
                h_tile = row[j_idx - 1]
                mu_left, logvar_left = h_tile[:, : self.z_dim], h_tile[:, self.z_dim :]
                var_left = logvar_left.exp()
                mu = self.blend_h(mu_left, mu, blend_width, power=1)
                var = self.blend_h(var_left, var, blend_width, power=2)

            logvar = var.clamp_min(1e-12).log()
            h_blended = torch.cat([mu, logvar], dim=1)
            h_core = h_blended[
                :,
                :,
                :tile_latent_stride_length,
                :tile_latent_stride_height,
                :tile_latent_stride_width,
            ]
            result_grid[t_idx][i_idx][j_idx] = h_core

        # Stitch blended tiles to spatially correct places
        result_times = []
        for t_idx in range(nT):
            result_rows = []
            for i_idx in range(nH):
                result_row = []
                for j_idx in range(nW):
                    result_row.append(result_grid[t_idx][i_idx][j_idx])
                result_rows.append(torch.cat(result_row, dim=4))
            result_times.append(torch.cat(result_rows, dim=3))

        h_full = torch.cat(result_times, dim=2)[:, :, :latent_length, :latent_height, :latent_width]
        return h_full

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        B, C, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio
        sample_length = num_frames * self.temporal_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_length = self.tile_sample_min_length // self.temporal_compression_ratio

        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_latent_stride_length = self.tile_sample_stride_length // self.temporal_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width
        blend_length = self.tile_sample_min_length - self.tile_sample_stride_length

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        times = []
        for t in range(0, num_frames, tile_latent_stride_length):
            rows = []
            for i in range(0, height, tile_latent_stride_height):
                row = []
                for j in range(0, width, tile_latent_stride_width):
                    tile = z[
                        :,
                        :,
                        t : t + tile_latent_min_length,
                        i : i + tile_latent_min_height,
                        j : j + tile_latent_min_width,
                    ]
                    decoded = self._decode_tile(tile)
                    row.append(decoded)
                rows.append(row)
            times.append(rows)

        result_times = []
        for t in range(len(times)):
            result_rows = []
            for i in range(len(times[t])):
                result_row = []
                for j in range(len(times[t][i])):
                    # Clone the current decoded tile to ensure blending uses an unmodified copy of the tile.
                    tile = times[t][i][j].clone()

                    if t > 0:
                        tile = self.blend_t(times[t - 1][i][j], tile, blend_length, power=1)
                    if i > 0:
                        tile = self.blend_v(times[t][i - 1][j], tile, blend_height, power=1)
                    if j > 0:
                        tile = self.blend_h(times[t][i][j - 1], tile, blend_width, power=1)

                    result_row.append(
                        tile[
                            :,
                            :,
                            : self.tile_sample_stride_length,
                            : self.tile_sample_stride_height,
                            : self.tile_sample_stride_width,
                        ]
                    )

                result_rows.append(torch.cat(result_row, dim=4))
            result_times.append(torch.cat(result_rows, dim=3))
        dec = torch.cat(result_times, dim=2)[:, :, :sample_length, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec
