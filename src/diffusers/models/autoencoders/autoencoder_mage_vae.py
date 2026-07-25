# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
MageVAE: DConvEncoder + DConvDenoiser (with CoD Decoder) autoencoder.

Encodes images to 128-channel latents at 16x spatial downsampling using a one-step
diffusion encoder, and decodes latents back to images using a DConv denoiser conditioned
on a CoD (Cascaded-of-Decoders) decoder.

Latent shape: [B, 128, H/16, W/16].
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _mage_vae_modulate(x, shift, scale):
    if x.dim() == 4:
        batch_size, channels = x.shape[:2]
        return x * (1 + scale.view(batch_size, channels, 1, 1)) + shift.view(batch_size, channels, 1, 1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------
class MageVAELayerNorm2d(nn.LayerNorm):
    """Channel-last LayerNorm for NCHW tensors."""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class MageVAERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class MageVAETimestepEmbedder(nn.Module):
    """Timestep MLP (max_period=10000, freq_size=256)."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        embedding = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# DConv blocks
# ---------------------------------------------------------------------------
class MageVAEDiCoBlock(nn.Module):
    """DConv block with adaLN modulation, used in encoder and decoder."""

    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )

        ffn_channels = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn_channels, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels, hidden_size, 1, bias=True)

        self.norm1 = MageVAELayerNorm2d(hidden_size, affine=False)
        self.norm2 = MageVAELayerNorm2d(hidden_size, affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, hidden_states, conditioning):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
            6, dim=1
        )
        residual = hidden_states
        hidden_states = _mage_vae_modulate(self.norm1(residual), shift_msa, scale_msa)
        hidden_states = F.gelu(self.conv2(self.conv1(hidden_states)))
        hidden_states = hidden_states * self.ca(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = residual + gate_msa[..., None, None] * hidden_states

        hidden_states = hidden_states + gate_mlp[..., None, None] * self.conv5(
            F.gelu(self.conv4(_mage_vae_modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)))
        )
        return hidden_states


class MageVAEEncoderDiCoBlock(nn.Module):
    """DConv block without adaLN modulation, for the encoder head pathway."""

    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )

        ffn_channels = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn_channels, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels, hidden_size, 1, bias=True)

        self.norm1 = MageVAELayerNorm2d(hidden_size, affine=True)
        self.norm2 = MageVAELayerNorm2d(hidden_size, affine=True)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(residual)
        hidden_states = F.gelu(self.conv2(self.conv1(hidden_states)))
        hidden_states = hidden_states * self.ca(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states + self.conv5(F.gelu(self.conv4(self.norm2(hidden_states))))


# ---------------------------------------------------------------------------
# Nerf-style patch embedder and final layer
# ---------------------------------------------------------------------------
class MageVAENerfEmbedder(nn.Module):
    """Patch-position embedder for the DConv decoder x-pathway."""

    def __init__(self, in_channels, hidden_size_input, max_freqs=8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True),
        )
        self._pos_cache = {}

    def _compute_pos(self, patch_size, device, dtype):
        key = (patch_size, device, dtype)
        if key in self._pos_cache:
            return self._pos_cache[key]

        pos = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos, pos, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        fx = freqs[None, :, None]
        fy = freqs[None, None, :]
        coeffs = (1 + fx * fy) ** -1
        dct_x = torch.cos(pos_x * fx * torch.pi)
        dct_y = torch.cos(pos_y * fy * torch.pi)

        result = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs**2)
        self._pos_cache[key] = result
        return result

    def forward(self, x):
        batch_size, num_patches, _ = x.shape
        patch_size = int(num_patches**0.5)
        dct = self._compute_pos(patch_size, x.device, x.dtype).expand(batch_size, -1, -1)
        return self.embedder(torch.cat([x, dct], dim=-1))


class MageVAENerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = MageVAERMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm(x))


# ---------------------------------------------------------------------------
# MLP decoder (SimpleMLPAdaLN + MLPResBlock)
# ---------------------------------------------------------------------------
class _MageVAEMLPResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x, y):
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class MageVAESimpleMLPAdaLN(nn.Module):
    """Small MLP that maps NerfEmbedder features to per-patch output, conditioned on spatial features."""

    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([_MageVAEMLPResBlock(model_channels) for _ in range(num_res_blocks)])

    def forward(self, x, conditioning):
        x = self.input_proj(x)
        conditioning = self.cond_embed(conditioning).reshape(conditioning.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            x = block(x, conditioning)
        return x


# ---------------------------------------------------------------------------
# CoD Decoder building blocks (ResNet + Attention)
# ---------------------------------------------------------------------------
class MageVAEResnetBlock(nn.Module):
    """GroupNorm + Conv ResBlock used by the CoD Decoder."""

    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        hidden_states = self.conv1(F.silu(self.norm1(x)))
        hidden_states = self.conv2(self.dropout(F.silu(self.norm2(hidden_states))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden_states


class MageVAEAttnBlock(nn.Module):
    """Patched self-attention for the CoD Decoder."""

    def __init__(self, in_channels, patch_size=32):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        normalized = self.norm(x)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)

        d = self.patch_size
        batch_size, channels, height, width = query.shape
        pad_h = (d - height % d) % d
        pad_w = (d - width % d) % d
        if pad_h or pad_w:
            query = F.pad(query, (0, pad_w, 0, pad_h), mode="replicate")
            key = F.pad(key, (0, pad_w, 0, pad_h), mode="replicate")
            value = F.pad(value, (0, pad_w, 0, pad_h), mode="replicate")

        _, _, height_padded, width_padded = query.shape
        num_patches_h = height_padded // d
        num_patches_w = width_padded // d
        num_patches = num_patches_h * num_patches_w

        # Reshape to patches: [B*num_patches, C, d*d]
        query = (
            query.reshape(batch_size, channels, num_patches_h, d, num_patches_w, d)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size * num_patches, channels, d * d)
        )
        key = (
            key.reshape(batch_size, channels, num_patches_h, d, num_patches_w, d)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size * num_patches, channels, d * d)
        )
        value = (
            value.reshape(batch_size, channels, num_patches_h, d, num_patches_w, d)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size * num_patches, channels, d * d)
        )

        # Attention via F.scaled_dot_product_attention
        # query/key/value: [B*np, C, d*d] -> [B*np, 1, d*d, C] for SDPA (batch, heads, seq, head_dim)
        q = query.permute(0, 2, 1).unsqueeze(1)
        k = key.permute(0, 2, 1).unsqueeze(1)
        v = value.permute(0, 2, 1).unsqueeze(1)
        h_ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        h_ = h_.squeeze(1).permute(0, 2, 1)  # back to [B*np, C, d*d]

        # Reconstruct
        hidden_states = (
            h_
            .reshape(batch_size, num_patches_h, num_patches_w, channels, d, d)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch_size, channels, height_padded, width_padded)
        )
        if pad_h or pad_w:
            hidden_states = hidden_states[:, :, :height, :width]

        return x + self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------
class MageVAEBottleneckPatchEmbed(nn.Module):
    """Image patch embed concatenated with a per-patch conditioning vector."""

    def __init__(self, patch_size=16, in_channels=3, bottleneck_dim=128, embed_dim=384, bias=True):
        super().__init__()
        self.proj1 = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim + embed_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x, conditioning):
        return self.proj2(torch.cat([self.proj1(x), conditioning], dim=1))


# ---------------------------------------------------------------------------
# adaLN constant folding
# ---------------------------------------------------------------------------
class _MageVAEConstAdaLN(nn.Module):
    """Replaces an adaLN_modulation MLP with a precomputed constant buffer."""

    def __init__(self, modulation: torch.Tensor):
        super().__init__()
        self.register_buffer("modulation", modulation.detach().clone())

    def forward(self, conditioning):
        batch_size = conditioning.shape[0]
        if self.modulation.shape[0] != batch_size:
            return self.modulation.expand(batch_size, *self.modulation.shape[1:])
        return self.modulation


# ---------------------------------------------------------------------------
# DConv Encoder
# ---------------------------------------------------------------------------
class MageVAEDConvEncoder(nn.Module):
    """One-step diffusion encoder: image -> packed (mean, logvar) latent."""

    def __init__(
        self,
        latent_channels=128,
        hidden_size=384,
        num_blocks=21,
        patch_size=16,
        mlp_ratio=4.0,
        head_size=768,
        num_head_blocks=2,
        out_ch_mult=2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size

        self.patch_cond_embed = nn.Conv2d(3, head_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.head_blocks = nn.ModuleList(
            [MageVAEEncoderDiCoBlock(head_size, mlp_ratio=mlp_ratio) for _ in range(num_head_blocks)]
        )
        self.proj_down = nn.Conv2d(head_size, hidden_size, kernel_size=1, bias=True)

        self.z_proj = nn.Conv2d(latent_channels, hidden_size, kernel_size=1, bias=True)
        self.fuse_proj = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=True)

        self.t_embedder = MageVAETimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([MageVAEDiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)])

        self.norm_out = MageVAELayerNorm2d(hidden_size, affine=True)
        self.proj_out = nn.Conv2d(hidden_size, latent_channels * out_ch_mult, kernel_size=1, bias=True)

    def forward(self, z_t, t, image):
        conditioning = self.patch_cond_embed(image)
        for block in self.head_blocks:
            conditioning = block(conditioning)
        conditioning = self.proj_down(conditioning)

        hidden_states = self.fuse_proj(torch.cat([conditioning, self.z_proj(z_t)], dim=1))
        timestep_embedding = self.t_embedder(t.view(-1))
        for block in self.blocks:
            hidden_states = block(hidden_states, timestep_embedding)
        return self.proj_out(self.norm_out(hidden_states))


# ---------------------------------------------------------------------------
# CoD Decoder: latent -> conditioning features for the denoiser
# ---------------------------------------------------------------------------
class MageVAEDecoder(nn.Module):
    """CoD (Cascaded-of-Decoders) decoder: latent -> spatial conditioning features."""

    def __init__(self, out_ch=384, z_ch=128):
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.block = nn.Sequential(
            MageVAEResnetBlock(in_channels=out_ch, out_channels=out_ch),
            MageVAEAttnBlock(out_ch, patch_size=32),
            MageVAEResnetBlock(in_channels=out_ch, out_channels=out_ch),
            MageVAEAttnBlock(out_ch, patch_size=32),
            MageVAEResnetBlock(in_channels=out_ch, out_channels=out_ch),
        )
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        hidden_states = self.block(self.conv_in(z))
        hidden_states = self.conv_out(F.silu(self.norm_out(hidden_states)))
        return hidden_states


# ---------------------------------------------------------------------------
# Y-Embedder wrapper (holds the CoD decoder)
# ---------------------------------------------------------------------------
class _MageVAEYEmbedder(nn.Module):
    """Namespace wrapper for the CoD decoder, matching the original checkpoint's
    ``pipeline.y_embedder.decoder.*`` weight key hierarchy."""

    def __init__(self, hidden_size=384, latent_channels=128):
        super().__init__()
        self.decoder = MageVAEDecoder(out_ch=hidden_size, z_ch=latent_channels)


# ---------------------------------------------------------------------------
# DConv Denoiser: conditioning + zero noise -> reconstructed image
# ---------------------------------------------------------------------------
class MageVAEDConvDenoiser(nn.Module):
    """One-step denoiser: takes conditioning from CoD decoder and produces the output image."""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        hidden_size=384,
        hidden_size_x=32,
        mlp_ratio=4.0,
        num_blocks=24,
        num_cond_blocks=21,
        bottleneck_dim=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_cond_blocks = num_cond_blocks

        self.t_embedder = MageVAETimestepEmbedder(hidden_size)
        self.y_embedder_x = nn.Conv2d(hidden_size, hidden_size_x * patch_size**2, 1, 1, 0)
        self.x_embedder = MageVAENerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = MageVAEBottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        self.blocks = nn.ModuleList(
            [MageVAEDiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_cond_blocks)]
        )

        self.dec_net = MageVAESimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.final_layer = MageVAENerfFinalLayer(hidden_size_x, in_channels)
        self.y_embedder = _MageVAEYEmbedder(hidden_size=hidden_size, latent_channels=bottleneck_dim)

    def forward(self, x, t, conditioning, is_latent=False):
        if is_latent:
            conditioning = self.y_embedder.decoder(conditioning)

        batch_size, _, height, width = x.shape
        timestep_embedding = self.t_embedder(t.view(-1))

        # Spatial conditioning path
        spatial = self.s_embedder(x, conditioning)
        for block in self.blocks:
            spatial = block(spatial, timestep_embedding)

        num_spatial = spatial.shape[-2] * spatial.shape[-1]
        spatial_flat = spatial.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)

        # Per-patch x-pathway
        x_unfolded = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        y_x = self.y_embedder_x(conditioning).flatten(2)
        x_combined = torch.cat([x_unfolded, y_x], dim=1)

        # Reshape: [B, (in_ch + hidden_x), ps^2, num_spatial] -> [B*num_spatial, ps^2, (in_ch + hidden_x)]
        x_combined = (
            x_combined.reshape(batch_size, -1, self.patch_size**2, num_spatial).permute(0, 3, 2, 1).flatten(0, 1)
        )

        x_embedded = self.x_embedder(x_combined)
        x_decoded = self.dec_net(x_embedded, spatial_flat)
        x_final = self.final_layer(x_decoded)

        # Fold patches back to image: [B*num_spatial, ps^2, in_ch] -> [B, in_ch, H, W]
        x_final = x_final.transpose(1, 2).reshape(batch_size, num_spatial, -1)
        return F.fold(
            x_final.transpose(1, 2).contiguous(),
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )


# ---------------------------------------------------------------------------
# Main autoencoder
# ---------------------------------------------------------------------------
class AutoencoderMageVAE(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    MageVAE autoencoder model using a one-step diffusion encoder and a DConv denoiser
    with a CoD (Cascaded-of-Decoders) decoder.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods
    implemented for all models (such as downloading or saving).

    Encoder: DConvEncoder takes an image [B, 3, H, W] and produces a latent [B, 128, H/16, W/16].
    Decoder: CoD Decoder + DConvDenoiser takes a latent and reconstructs the image [B, 3, H, W].

    Args:
        latent_channels (`int`, defaults to `128`):
            Number of channels in the latent space.
        downsample_factor (`int`, defaults to `16`):
            Spatial downsampling factor from image to latent.
        encoder_hidden_size (`int`, defaults to `384`):
            Hidden dimension of the encoder DConv blocks.
        encoder_num_blocks (`int`, defaults to `21`):
            Number of adaLN-modulated DConv blocks in the encoder.
        encoder_patch_size (`int`, defaults to `16`):
            Patch size for the encoder's image tokenization.
        encoder_head_size (`int`, defaults to `768`):
            Channel dimension of the encoder's head blocks.
        encoder_num_head_blocks (`int`, defaults to `2`):
            Number of head blocks in the encoder (without adaLN).
        decoder_hidden_size (`int`, defaults to `384`):
            Hidden dimension of the decoder DConv blocks.
        decoder_hidden_size_x (`int`, defaults to `32`):
            Hidden dimension of the decoder's per-patch x-pathway.
        decoder_num_blocks (`int`, defaults to `24`):
            Total number of blocks in the decoder (cond blocks + MLP res blocks).
        decoder_num_cond_blocks (`int`, defaults to `21`):
            Number of adaLN-modulated DConv blocks in the decoder.
        decoder_bottleneck_dim (`int`, defaults to `128`):
            Bottleneck dimension for the patch embedding and CoD decoder input.
        decoder_patch_size (`int`, defaults to `16`):
            Patch size for the decoder.
        sample_posterior (`bool`, defaults to `True`):
            Whether to sample from the posterior (mean + noise * std) or use the mean directly.
    """

    _no_split_modules = ["MageVAEDiCoBlock", "MageVAEResnetBlock", "MageVAEAttnBlock"]
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 128,
        downsample_factor: int = 16,
        encoder_hidden_size: int = 384,
        encoder_num_blocks: int = 21,
        encoder_patch_size: int = 16,
        encoder_head_size: int = 768,
        encoder_num_head_blocks: int = 2,
        decoder_hidden_size: int = 384,
        decoder_hidden_size_x: int = 32,
        decoder_num_blocks: int = 24,
        decoder_num_cond_blocks: int = 21,
        decoder_bottleneck_dim: int = 128,
        decoder_patch_size: int = 16,
        sample_posterior: bool = True,
    ):
        super().__init__()

        self.encoder = MageVAEDConvEncoder(
            latent_channels=latent_channels,
            hidden_size=encoder_hidden_size,
            num_blocks=encoder_num_blocks,
            patch_size=encoder_patch_size,
            head_size=encoder_head_size,
            num_head_blocks=encoder_num_head_blocks,
        )

        self.decoder = MageVAEDConvDenoiser(
            patch_size=decoder_patch_size,
            in_channels=3,
            hidden_size=decoder_hidden_size,
            hidden_size_x=decoder_hidden_size_x,
            num_blocks=decoder_num_blocks,
            num_cond_blocks=decoder_num_cond_blocks,
            bottleneck_dim=decoder_bottleneck_dim,
        )

    def encode(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Encode images to latents.

        Args:
            x (`torch.Tensor`): Input images of shape `[B, 3, H, W]`. H and W must be
                multiples of `encoder_patch_size`.
            generator (`torch.Generator`, *optional*):
                A torch generator for reproducible sampling.

        Returns:
            `torch.Tensor`: Latent of shape `[B, 128, H/16, W/16]`.
        """
        batch_size, _, height, width = x.shape
        patch_size = self.config.encoder_patch_size
        latent_channels = self.config.latent_channels

        z_t = torch.zeros(
            batch_size,
            latent_channels,
            height // patch_size,
            width // patch_size,
            device=x.device,
            dtype=x.dtype,
        )
        t = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        out = self.encoder(z_t, t, x)
        mean = out[:, :latent_channels]
        logvar = out[:, latent_channels:].clamp(min=-20.0, max=10.0)

        if self.config.sample_posterior:
            noise = randn_tensor(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
            return mean + torch.exp(0.5 * logvar) * noise
        return mean

    def forward(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        return self.decode(z, return_dict=return_dict)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        """
        Decode latents to images.

        Args:
            z (`torch.Tensor`): Latent of shape `[B, 128, H/16, W/16]`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] or a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                Decoded images of shape `[B, 3, H, W]`.
        """
        batch_size = z.shape[0]
        height = z.shape[2] * self.config.downsample_factor
        width = z.shape[3] * self.config.downsample_factor
        noise = torch.zeros(batch_size, 3, height, width, device=z.device, dtype=z.dtype)
        t = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        sample = self.decoder(noise, t, z, is_latent=True)

        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def freeze_adaln(self):
        """Constant-fold adaLN_modulation MLPs at t=0 for both encoder and decoder.

        At t=0 the adaLN modulation outputs are constant (they only depend on the
        timestep embedding). This method precomputes those constants and replaces the
        MLP modules with lightweight buffer wrappers, saving compute and parameters.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        t = torch.zeros(1, device=device, dtype=dtype)

        c_enc = self.encoder.t_embedder(t)
        count_enc = self._replace_adaln_with_const(self.encoder, c_enc)

        c_dec = self.decoder.t_embedder(t)
        count_dec = self._replace_adaln_with_const(self.decoder, c_dec)

        logger.info(f"MageVAE: folded {count_enc} encoder + {count_dec} decoder adaLN blocks")

    @staticmethod
    def _replace_adaln_with_const(module: nn.Module, conditioning: torch.Tensor) -> int:
        """Replace adaLN_modulation MLPs in MageVAEDiCoBlock instances with constant buffers."""
        count = 0
        for child in module.modules():
            if not isinstance(child, MageVAEDiCoBlock):
                continue
            adaln = child.adaLN_modulation
            if isinstance(adaln, _MageVAEConstAdaLN):
                continue
            with torch.no_grad():
                modulation = adaln(conditioning)
            child.adaLN_modulation = _MageVAEConstAdaLN(modulation)
            count += 1
        return count
