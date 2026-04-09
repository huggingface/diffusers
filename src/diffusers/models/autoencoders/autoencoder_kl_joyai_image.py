# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from contextlib import nullcontext

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin
from ...loaders import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .autoencoder_kl_wan import (
    WanAttentionBlock as AttentionBlock,
)
from .autoencoder_kl_wan import (
    WanCausalConv3d as CausalConv3d,
)
from .autoencoder_kl_wan import (
    WanResample as Resample,
)
from .autoencoder_kl_wan import (
    WanResidualBlock as ResidualBlock,
)
from .autoencoder_kl_wan import (
    WanRMS_norm as RMS_norm,
)
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)
CACHE_T = 2


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
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
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
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
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
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
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

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
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
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
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
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
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
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

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

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
                    x[:, :, 1 + 4 * (chunk_idx - 1): 1 + 4 * chunk_idx, :, :],
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
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale=None):
        self.clear_cache()
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.z_dim, 1, 1, 1)
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
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
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
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ),
        latents_std: list[float] | tuple[float, ...] = (
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
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
        autocast_context = torch.amp.autocast(device_type="cuda", dtype=torch.float32) if videos.device.type == "cuda" else nullcontext()
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
        autocast_context = torch.amp.autocast(device_type="cuda", dtype=torch.float32) if zs.device.type == "cuda" else nullcontext()
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
