# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# This model is adapted from https://github.com/jd-opensource/JoyAI-Image

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin
from ...loaders import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)
CACHE_T = 2


# Copied from diffusers.models.autoencoders.autoencoder_kl_wan.WanCausalConv3d
class CausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


# Copied from diffusers.models.autoencoders.autoencoder_kl_wan.WanRMS_norm
class RMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
            t in str(x.dtype) for t in ("float4_", "float8_")
        )
        normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=(1 if self.channel_first else -1)).to(
            x.dtype
        )

        return normalized * self.scale * self.gamma + self.bias


# Copied from diffusers.models.autoencoders.autoencoder_kl_wan.WanUpsample
class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


# Copied from diffusers.models.autoencoders.autoencoder_kl_wan.WanResample
class Resample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


# Copied from diffusers.models.autoencoders.autoencoder_kl_wan.WanResidualBlock
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = RMS_norm(in_dim, images=False)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


# Copied from diffusers.models.autoencoders.autoencoder_kl_wan.WanAttentionBlock
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.ModuleList(downsamples)

        # middle blocks
        self.middle = nn.ModuleList(
            [
                ResidualBlock(out_dim, out_dim, dropout),
                AttentionBlock(out_dim),
                ResidualBlock(out_dim, out_dim, dropout),
            ]
        )

        # output blocks
        self.head = nn.ModuleList(
            [
                RMS_norm(out_dim, images=False),
                nn.SiLU(),
                CausalConv3d(out_dim, z_dim, 3, padding=1),
            ]
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.ModuleList(
            [
                ResidualBlock(dims[0], dims[0], dropout),
                AttentionBlock(dims[0]),
                ResidualBlock(dims[0], dims[0], dropout),
            ]
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.ModuleList(upsamples)

        # output blocks
        self.head = nn.ModuleList(
            [
                RMS_norm(out_dim, images=False),
                nn.SiLU(),
                CausalConv3d(out_dim, 3, 3, padding=1),
            ]
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

    @property
    def quant_conv(self):
        return self.conv1

    @property
    def post_quant_conv(self):
        return self.conv2

    def _encode_frames(self, x):
        num_frames = x.shape[2]
        num_chunks = 1 + (num_frames - 1) // 4

        for chunk_idx in range(num_chunks):
            self._enc_conv_idx = [0]
            if chunk_idx == 0:
                encoded = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                encoded_chunk = self.encoder(
                    x[:, :, 1 + 4 * (chunk_idx - 1) : 1 + 4 * chunk_idx, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                encoded = torch.cat([encoded, encoded_chunk], dim=2)
        return encoded

    def _decode_frames(self, x):
        num_frames = x.shape[2]
        for frame_idx in range(num_frames):
            self._conv_idx = [0]
            decoded_chunk = self.decoder(
                x[:, :, frame_idx : frame_idx + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
            if frame_idx == 0:
                decoded = decoded_chunk
            else:
                decoded = torch.cat([decoded, decoded_chunk], dim=2)
        return decoded

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale=None, return_posterior=False):
        self.clear_cache()
        encoded = self._encode_frames(x)
        mu, log_var = self.quant_conv(encoded).chunk(2, dim=1)
        if scale is None or return_posterior:
            return mu, log_var

        mu = self.reparameterize(mu, log_var)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale=None):
        self.clear_cache()
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
        decoded = self._decode_frames(self.post_quant_conv(z))
        self.clear_cache()
        return decoded

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False, scale=None):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        mu = mu + std * torch.randn_like(std)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _build_video_vae(z_dim=None, use_meta=False, **kwargs):
    """Build the JoyAI/Wan-derived VAE backbone without loading external weights."""
    cfg = {
        "dim": 96,
        "z_dim": z_dim,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
    }
    cfg.update(**kwargs)

    if use_meta:
        with torch.device("meta"):
            return WanVAE_(**cfg)
    return WanVAE_(**cfg)


def _remap_joyai_vae_state_dict_keys(pretrained_state_dict):
    remapped_state_dict = {}
    for key, value in pretrained_state_dict.items():
        key = key.replace(".residual.0.gamma", ".norm1.gamma")
        key = key.replace(".residual.2.weight", ".conv1.weight")
        key = key.replace(".residual.2.bias", ".conv1.bias")
        key = key.replace(".residual.3.gamma", ".norm2.gamma")
        key = key.replace(".residual.6.weight", ".conv2.weight")
        key = key.replace(".residual.6.bias", ".conv2.bias")
        key = key.replace(".shortcut.weight", ".conv_shortcut.weight")
        key = key.replace(".shortcut.bias", ".conv_shortcut.bias")
        remapped_state_dict[key] = value
    return remapped_state_dict


def _load_pretrained_weights(model, pretrained_path):
    if not pretrained_path:
        return model

    logger.info(f"loading {pretrained_path}")

    if pretrained_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        pretrained_state_dict = load_file(pretrained_path, device="cpu")
    else:
        pretrained_state_dict = torch.load(pretrained_path, map_location="cpu")

    pretrained_state_dict = _remap_joyai_vae_state_dict_keys(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict, assign=True)
    return model


def _video_vae(pretrained_path=None, z_dim=None, use_meta=False, **kwargs):
    model = _build_video_vae(z_dim=z_dim, use_meta=use_meta, **kwargs)
    return _load_pretrained_weights(model, pretrained_path)


class JoyAIImageVAE(ModelMixin, ConfigMixin, AutoencoderMixin, FromOriginalModelMixin):
    def __init__(
        self,
        pretrained: str = "",
        torch_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        z_dim: int = 16,
        latent_channels: int | None = None,
        dim: int = 96,
        dim_mult: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: list[float] | tuple[float, ...] = (),
        temperal_downsample: list[bool] | tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
        latents_mean: list[float] | tuple[float, ...] = (
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
        ),
        latents_std: list[float] | tuple[float, ...] = (
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
        ),
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
    ):
        super().__init__()

        if latent_channels is not None:
            z_dim = latent_channels

        self.register_to_config(
            pretrained=pretrained,
            z_dim=z_dim,
            dim=dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=list(attn_scales),
            temperal_downsample=list(temperal_downsample),
            dropout=dropout,
            latent_channels=z_dim,
            latents_mean=list(latents_mean),
            latents_std=list(latents_std),
            spatial_compression_ratio=spatial_compression_ratio,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        self.register_buffer("mean", torch.tensor(latents_mean, dtype=torch.float32), persistent=True)
        self.register_buffer("std", torch.tensor(latents_std, dtype=torch.float32), persistent=True)

        self.ffactor_spatial = spatial_compression_ratio
        self.ffactor_temporal = temporal_compression_ratio

        use_meta = bool(pretrained)
        self.model = _video_vae(
            pretrained_path=pretrained,
            z_dim=z_dim,
            dim=dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=list(attn_scales),
            temperal_downsample=list(temperal_downsample),
            dropout=dropout,
            use_meta=use_meta,
        )
        self.model.eval()

    def _latent_scale_tensors(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        inv_std = self.std.to(device=device, dtype=dtype).reciprocal().view(1, -1, 1, 1, 1)
        return mean, inv_std

    @apply_forward_hook
    def encode(self, videos: torch.Tensor, return_dict: bool = True, return_posterior: bool = False, **kwargs):
        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float32)
            if videos.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            mean, logvar = self.model.encode(videos, scale=None, return_posterior=True)
            if return_posterior:
                return mean, logvar

            latent_mean, latent_inv_std = self._latent_scale_tensors(mean.device, mean.dtype)
            scaled_mean = (mean - latent_mean) * latent_inv_std
            scaled_logvar = logvar + 2 * torch.log(latent_inv_std)
            posterior = DiagonalGaussianDistribution(torch.cat([scaled_mean, scaled_logvar], dim=1))

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, zs: torch.Tensor, return_dict: bool = True, **kwargs):
        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float32) if zs.device.type == "cuda" else nullcontext()
        )
        with autocast_context:
            mean, inv_std = self._latent_scale_tensors(zs.device, zs.dtype)
            scale = [mean.view(-1), inv_std.view(-1)]
            videos = [self.model.decode(z.unsqueeze(0), scale=scale).clamp_(-1, 1).squeeze(0) for z in zs]
            videos = torch.stack(videos, dim=0)

        if not return_dict:
            return (videos,)
        return DecoderOutput(sample=videos)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ):
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latents, return_dict=return_dict)


WanxVAE = JoyAIImageVAE

__all__ = ["JoyAIImageVAE", "WanxVAE"]
