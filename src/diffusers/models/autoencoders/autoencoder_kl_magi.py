# Copyright 2025 SandAI and The HuggingFace Team. All rights reserved.
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

from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


class MagiVAEManualLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + self.eps)
        return x_normalized


class MagiVAEAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states):
        batch_size, sequence_length, dim = hidden_states.shape
        qkv = attn.qkv(hidden_states).reshape(batch_size, sequence_length, 3, attn.num_heads, dim // attn.num_heads)
        qkv = attn.qkv_norm(qkv)
        query, key, value = qkv.unbind(dim=2)
        hidden_states = dispatch_attention_fn(
            query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        return attn.proj(hidden_states.flatten(2, 3))


class MagiVAEAttention(nn.Module, AttentionModuleMixin):
    _supports_qkv_fusion = False
    _default_processor_cls = MagiVAEAttnProcessor
    _available_processors = [MagiVAEAttnProcessor]

    def __init__(self, dim, num_heads, eps):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_norm = MagiVAEManualLayerNorm(eps)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.set_processor(MagiVAEAttnProcessor())

    def forward(self, hidden_states):
        return self.processor(self, hidden_states)


class MagiVAEMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MagiVAEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, eps):
        super().__init__()
        self.attn = MagiVAEAttention(dim, num_heads, eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MagiVAEMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(hidden_states)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class MagiVAEPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, patch_length):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(patch_length, patch_size, patch_size),
            stride=(patch_length, patch_size, patch_size),
        )

    def forward(self, x):
        """Project a video onto a five-dimensional patch feature grid."""
        x = self.proj(x)
        return x


def resize_pos_embed(posemb, src_shape, target_shape):
    posemb = posemb.reshape(1, src_shape[0], src_shape[1], src_shape[2], -1)
    posemb = posemb.permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=target_shape, mode="trilinear", align_corners=False)
    posemb = posemb.permute(0, 2, 3, 4, 1)
    posemb = posemb.reshape(1, target_shape[0] * target_shape[1] * target_shape[2], -1)
    return posemb


class MagiVAEPositionEmbedding(nn.Module):
    def __init__(self, dim, latent_shape):
        super().__init__()
        self.latent_shape = latent_shape
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, latent_shape[0] * latent_shape[1] * latent_shape[2] + 1, dim))

    def forward(self, hidden_states, latent_shape):
        cls_tokens = self.cls_token.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        if latent_shape != self.latent_shape:
            pos_embed = resize_pos_embed(self.pos_embed[:, 1:], self.latent_shape, latent_shape)
            pos_embed = torch.cat((self.pos_embed[:, :1], pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed
        return hidden_states + pos_embed


class MagiVAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, dim, depth, num_heads, mlp_ratio, patch_shape, latent_shape, eps):
        super().__init__()
        self.patch_embed = MagiVAEPatchEmbed(in_channels, dim, patch_shape[1], patch_shape[0])
        self.position_embedding = MagiVAEPositionEmbedding(dim, latent_shape)
        self.blocks = nn.ModuleList([MagiVAEBlock(dim, num_heads, mlp_ratio, eps) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.last_layer = nn.Linear(dim, latent_channels * 2)
        self.gradient_checkpointing = False

    def forward(self, sample):
        hidden_states = self.patch_embed(sample)
        batch_size, _, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.position_embedding(hidden_states, (num_frames, height, width))
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states)
        hidden_states = self.last_layer(self.norm(hidden_states))[:, 1:]
        return hidden_states.reshape(batch_size, num_frames, height, width, -1).permute(0, 4, 1, 2, 3)


class MagiVAEDecoder(nn.Module):
    def __init__(
        self, out_channels, latent_channels, dim, depth, num_heads, mlp_ratio, patch_shape, latent_shape, eps
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.unpatch_channels = dim // (patch_shape[0] * patch_shape[1] * patch_shape[2])
        self.proj_in = nn.Linear(latent_channels, dim)
        self.position_embedding = MagiVAEPositionEmbedding(dim, latent_shape)
        self.blocks = nn.ModuleList([MagiVAEBlock(dim, num_heads, mlp_ratio, eps) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.last_layer = nn.Conv3d(self.unpatch_channels, out_channels, kernel_size=3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, latent):
        batch_size, _, num_frames, height, width = latent.shape
        hidden_states = latent.permute(0, 2, 3, 4, 1).flatten(1, 3)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.position_embedding(hidden_states, (num_frames, height, width))
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)[:, 1:]
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, height, width, *self.patch_shape, self.unpatch_channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch_size,
            self.unpatch_channels,
            num_frames * self.patch_shape[0],
            height * self.patch_shape[1],
            width * self.patch_shape[2],
        )
        return self.last_layer(hidden_states)


class AutoencoderKLMagi(ModelMixin, AttentionMixin, ConfigMixin):
    r"""
    Transformer VAE used by MAGI-1.

    The encoder returns a diagonal Gaussian posterior over video latents. Both encoder and decoder use full attention
    over the input patch sequence. The learned position embeddings are interpolated for different video sizes.

    Parameters:
        in_channels (`int`, defaults to 3):
            Number of input video channels.
        out_channels (`int`, defaults to 3):
            Number of reconstructed video channels.
        latent_channels (`int`, defaults to 16):
            Number of latent channels.
        embed_dim (`int`, defaults to 1024):
            Transformer hidden size. Must be divisible by the attention head count and the patch volume.
        num_layers (`int`, defaults to 24):
            Number of transformer blocks in each of the encoder and decoder.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads.
        mlp_ratio (`float`, defaults to 4.0):
            Feed-forward hidden size relative to the transformer hidden size.
        patch_size (`int`, defaults to 8):
            Spatial patch size and compression ratio.
        patch_length (`int`, defaults to 4):
            Temporal patch size and compression ratio.
        sample_size (`int`, defaults to 256):
            Spatial size used to define the learned position embedding grid.
        sample_frames (`int`, defaults to 16):
            Frame count used to define the learned position embedding grid.
        norm_eps (`float`, defaults to 1e-5):
            Epsilon for normalization. Attention divides by the standard deviation plus epsilon.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MagiVAEBlock", "MagiVAEPositionEmbedding"]
    _repeated_blocks = ["MagiVAEBlock"]
    _skip_layerwise_casting_patterns = ["patch_embed", "pos_embed", "cls_token", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        embed_dim: int = 1024,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        patch_length: int = 4,
        sample_size: int = 256,
        sample_frames: int = 16,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if embed_dim % num_attention_heads:
            raise ValueError("embed_dim must be divisible by num_attention_heads.")
        if embed_dim % (patch_length * patch_size * patch_size):
            raise ValueError("embed_dim must be divisible by the spatiotemporal patch volume.")
        if sample_frames % patch_length or sample_size % patch_size:
            raise ValueError("The reference sample dimensions must be divisible by the patch dimensions.")
        patch_shape = (patch_length, patch_size, patch_size)
        latent_shape = (sample_frames // patch_length, sample_size // patch_size, sample_size // patch_size)
        args = (
            latent_channels,
            embed_dim,
            num_layers,
            num_attention_heads,
            mlp_ratio,
            patch_shape,
            latent_shape,
            norm_eps,
        )
        self.encoder = MagiVAEEncoder(in_channels, *args)
        self.decoder = MagiVAEDecoder(out_channels, *args)
        self.spatial_compression_ratio = patch_size
        self.temporal_compression_ratio = patch_length
        self.use_tiling = False
        self.use_slicing = False
        self.tile_sample_min_length = sample_frames
        self.tile_sample_min_height = sample_size
        self.tile_sample_min_width = sample_size
        self.temporal_tile_overlap_factor = 0.0
        self.spatial_tile_overlap_factor = 0.25
        self.allow_spatial_tiling = True

    def enable_slicing(self):
        """Encode and decode one batch item at a time."""
        self.use_slicing = True

    def disable_slicing(self):
        """Encode and decode the full batch together."""
        self.use_slicing = False

    def enable_tiling(
        self,
        tile_sample_min_length: int | None = None,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        temporal_tile_overlap_factor: float = 0.0,
        spatial_tile_overlap_factor: float = 0.25,
        allow_spatial_tiling: bool = True,
    ):
        """
        Enable single-device temporal and optional spatial tiling.

        Tile sizes are measured in input video pixels and frames. Overlaps must give integral latent strides. The
        default spatial overlap is 25%; temporal tiles do not overlap by default. Tiling changes the attention context
        and is not numerically equivalent to whole-video encoding or decoding.
        """
        lengths = (
            self.config.sample_frames if tile_sample_min_length is None else tile_sample_min_length,
            self.config.sample_size if tile_sample_min_height is None else tile_sample_min_height,
            self.config.sample_size if tile_sample_min_width is None else tile_sample_min_width,
        )
        overlaps = (temporal_tile_overlap_factor, spatial_tile_overlap_factor, spatial_tile_overlap_factor)
        factors = (self.config.patch_length, self.config.patch_size, self.config.patch_size)
        for axis, (length, overlap, factor) in enumerate(zip(lengths, overlaps, factors)):
            if not isinstance(length, int) or isinstance(length, bool) or length <= 0 or length % factor:
                raise ValueError("Tile dimensions must be positive integer multiples of the patch dimensions.")
            if not 0 <= overlap < 1:
                raise ValueError("Tile overlap factors must be in [0, 1).")
            if axis > 0 and not allow_spatial_tiling:
                continue
            latent_overlap = length // factor * overlap
            if not float(latent_overlap).is_integer():
                raise ValueError("Tile overlaps must align with the latent grid.")
        self.tile_sample_min_length, self.tile_sample_min_height, self.tile_sample_min_width = lengths
        self.temporal_tile_overlap_factor = temporal_tile_overlap_factor
        self.spatial_tile_overlap_factor = spatial_tile_overlap_factor
        self.allow_spatial_tiling = allow_spatial_tiling
        self.use_tiling = True

    def disable_tiling(self):
        """Restore whole-input encoding and decoding."""
        self.use_tiling = False

    def _tile_parameters(self, shape):
        sample_sizes = (self.tile_sample_min_length, self.tile_sample_min_height, self.tile_sample_min_width)
        factors = (self.config.patch_length, self.config.patch_size, self.config.patch_size)
        latent_sizes = tuple(size // factor for size, factor in zip(sample_sizes, factors))
        overlaps = (
            self.temporal_tile_overlap_factor,
            self.spatial_tile_overlap_factor,
            self.spatial_tile_overlap_factor,
        )
        if not self.allow_spatial_tiling:
            latent_sizes = (latent_sizes[0], shape[1], shape[2])
            overlaps = (overlaps[0], 0.0, 0.0)
        blend_extents = tuple(int(size * overlap) for size, overlap in zip(latent_sizes, overlaps))
        strides = tuple(size - blend for size, blend in zip(latent_sizes, blend_extents))
        return latent_sizes, strides, blend_extents

    @staticmethod
    def _blend(previous, current, extent, dim, upcast=False):
        extent = min(previous.shape[dim], current.shape[dim], extent)
        for index in range(extent):
            before = previous.select(dim, previous.shape[dim] - extent + index)
            after = current.select(dim, index)
            if upcast:
                # The compiled reference decoder blends low-precision tiles with FP32 intermediates.
                blend_dtype = torch.promote_types(current.dtype, torch.float32)
                before = before.to(blend_dtype)
                after = after.to(blend_dtype)
            current.select(dim, index).copy_(before * (1 - index / extent) + after * (index / extent))
        return current

    def _encode(self, x):
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.config.patch_length, -1, -1)
        if x.shape[2] % self.config.patch_length:
            raise ValueError("Each temporal tile must contain complete patches or a single frame.")
        return self.encoder(x)

    def _tiled_encode(self, x):
        factors = (self.config.patch_length, self.config.patch_size, self.config.patch_size)
        latent_shape = tuple((size + factor - 1) // factor for size, factor in zip(x.shape[2:], factors))
        sizes, strides, blends = self._tile_parameters(latent_shape)
        positions = [range(0, size, stride) for size, stride in zip(latent_shape, strides)]
        tiles = {}
        for frame, height, width in product(*positions):
            tile = x[
                :,
                :,
                frame * factors[0] : (frame + sizes[0]) * factors[0],
                height * factors[1] : (height + sizes[1]) * factors[1],
                width * factors[2] : (width + sizes[2]) * factors[2],
            ]
            tile = self._encode(tile)
            for axis, position in enumerate((frame, height, width)):
                if position > 0 and blends[axis] > 0:
                    previous = [frame, height, width]
                    previous[axis] -= strides[axis]
                    tile = self._blend(tiles[tuple(previous)], tile.clone(), blends[axis], axis + 2)
            tiles[(frame, height, width)] = tile
        return torch.cat(
            [
                torch.cat(
                    [
                        torch.cat(
                            [
                                tiles[(frame, height, width)][:, :, : strides[0], : strides[1], : strides[2]]
                                for width in positions[2]
                            ],
                            dim=4,
                        )
                        for height in positions[1]
                    ],
                    dim=3,
                )
                for frame in positions[0]
            ],
            dim=2,
        )

    def _tiled_decode(self, z, preserve_frames):
        sizes, strides, blends = self._tile_parameters(z.shape[2:])
        factors = (self.config.patch_length, self.config.patch_size, self.config.patch_size)
        limits = tuple(stride * factor for stride, factor in zip(strides, factors))
        positions = [range(0, size, stride) for size, stride in zip(z.shape[2:], strides)]
        tiles = {}
        results = {}
        for frame, height, width in product(*positions):
            latent = z[:, :, frame : frame + sizes[0], height : height + sizes[1], width : width + sizes[2]]
            decoded = self.decoder(latent)
            if latent.shape[2] == 1 and not preserve_frames:
                decoded = decoded[:, :, :1]
            tiles[(frame, height, width)] = decoded
            tile = decoded.clone()
            for axis, position in enumerate((frame, height, width)):
                if position > 0 and blends[axis] > 0:
                    previous = [frame, height, width]
                    previous[axis] -= strides[axis]
                    tile = self._blend(
                        tiles[tuple(previous)], tile, blends[axis] * factors[axis], axis + 2, upcast=True
                    )
            results[(frame, height, width)] = tile[:, :, : limits[0], : limits[1], : limits[2]]
        return torch.cat(
            [
                torch.cat(
                    [
                        torch.cat([results[(frame, height, width)] for width in positions[2]], dim=4)
                        for height in positions[1]
                    ],
                    dim=3,
                )
                for frame in positions[0]
            ],
            dim=2,
        )

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        """
        Encode an image or video into a posterior without sampling it.

        Args:
            x (`torch.Tensor`):
                Input of shape `(batch, channels, frames, height, width)`, normalized to [-1, 1]. Spatial dimensions
                must be divisible by the spatial patch size. A single frame is repeated to fill one temporal patch.
                Tiling additionally supports a final one-frame temporal tile.
            return_dict (`bool`, defaults to `True`):
                Return an `AutoencoderKLOutput` instead of a tuple.

        Returns:
            `AutoencoderKLOutput` or `tuple`:
                The posterior. With tiling, its mean and log-variance are blended independently. Use `.mode()` to
                reproduce the official deterministic inference encoder.
        """
        if x.ndim != 5 or x.shape[1] != self.config.in_channels or any(size <= 0 for size in x.shape):
            raise ValueError(
                "Expected a nonempty video tensor with shape (batch, in_channels, frames, height, width)."
            )
        if any(size % self.config.patch_size for size in x.shape[3:]):
            raise ValueError("Spatial dimensions must be divisible by the patch size.")
        remainder = x.shape[2] % self.config.patch_length
        if x.shape[2] != 1 and remainder and not (self.use_tiling and remainder == 1):
            raise ValueError("Frame count must be divisible by patch_length, or end in a single frame when tiling.")
        encode = self._tiled_encode if self.use_tiling else self._encode
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([encode(sample) for sample in x.split(1)], dim=0)
        else:
            moments = encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if self.use_tiling:
            # The reference concatenates means, without interleaved log-variance storage.
            posterior.mean = posterior.mean.clone()
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True, num_frames: int | None = None
    ) -> DecoderOutput | tuple:
        """
        Decode latents using the official MAGI image/video convention.

        Args:
            z (`torch.Tensor`):
                Latents of shape `(batch, latent_channels, frames, height, width)`.
            return_dict (`bool`, defaults to `True`):
                Return a `DecoderOutput` instead of a tuple.
            num_frames (`int`, optional):
                Requested output length. If omitted, each one-position latent tile returns only its first decoded
                frame, matching the reference. Set this to the original video length to retain all frames.

        Returns:
            `DecoderOutput` or `tuple`:
                The reconstructed image or video.
        """
        if z.ndim != 5 or z.shape[1] != self.config.latent_channels or any(size <= 0 for size in z.shape):
            raise ValueError("Expected nonempty latents with shape (batch, latent_channels, frames, height, width).")
        if num_frames is not None and (
            not isinstance(num_frames, int)
            or isinstance(num_frames, bool)
            or not (z.shape[2] - 1) * self.config.patch_length < num_frames <= z.shape[2] * self.config.patch_length
        ):
            raise ValueError("num_frames must fit the number of latent temporal patches.")
        batches = z.split(1) if self.use_slicing else (z,)
        outputs = []
        for latent in batches:
            if self.use_tiling:
                decoded = self._tiled_decode(latent, preserve_frames=num_frames is not None)
            else:
                decoded = self.decoder(latent)
                if latent.shape[2] == 1 and num_frames is None:
                    decoded = decoded[:, :, :1]
            outputs.append(decoded)
        decoded = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        if num_frames is not None:
            decoded = decoded[:, :, :num_frames]
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | tuple:
        """
        Reconstruct an image or video, preserving its frame count.

        Args:
            sample (`torch.Tensor`):
                Input of shape `(batch, channels, frames, height, width)`, normalized to [-1, 1].
            sample_posterior (`bool`, defaults to `False`):
                Sample the posterior instead of using its mode. Sampling follows Diffusers generator and dtype
                conventions; the official inference pipeline uses the mode.
            return_dict (`bool`, defaults to `True`):
                Return a `DecoderOutput` instead of a tuple.
            generator (`torch.Generator`, optional):
                Random generator used when sampling the posterior.

        Returns:
            `DecoderOutput` or `tuple`:
                The reconstruction with the original frame count.
        """
        posterior = self.encode(sample).latent_dist
        latent = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latent, return_dict=return_dict, num_frames=sample.shape[2])
