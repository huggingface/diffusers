# Copyright 2026 The NYU Vision-X and HuggingFace Teams. All rights reserved.
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
from math import sqrt
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ...utils.accelerate_utils import apply_forward_hook
from ...utils.import_utils import is_transformers_available
from ...utils.torch_utils import randn_tensor


if is_transformers_available():
    from transformers import (
        Dinov2WithRegistersConfig,
        Dinov2WithRegistersModel,
        SiglipVisionConfig,
        SiglipVisionModel,
        ViTMAEConfig,
        ViTMAEModel,
    )

from ..activations import get_activation
from ..attention import AttentionMixin
from ..attention_processor import Attention
from ..embeddings import get_2d_sincos_pos_embed
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, EncoderOutput


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-encoder forward functions
# ---------------------------------------------------------------------------
# Each function takes the raw transformers model + images and returns patch
# tokens of shape (B, N, C), stripping CLS / register tokens as needed.


def _dinov2_encoder_forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images, output_hidden_states=True)
    unused_token_num = 5  # 1 CLS + 4 register tokens
    return outputs.last_hidden_state[:, unused_token_num:]


def _siglip2_encoder_forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images, output_hidden_states=True, interpolate_pos_encoding=True)
    return outputs.last_hidden_state


def _mae_encoder_forward(model: nn.Module, images: torch.Tensor, patch_size: int) -> torch.Tensor:
    h, w = images.shape[2], images.shape[3]
    patch_num = int(h * w // patch_size**2)
    if patch_num * patch_size**2 != h * w:
        raise ValueError("Image size should be divisible by patch size.")
    noise = torch.arange(patch_num).unsqueeze(0).expand(images.shape[0], -1).to(images.device).to(images.dtype)
    outputs = model(images, noise, interpolate_pos_encoding=True)
    return outputs.last_hidden_state[:, 1:]  # remove cls token


# ---------------------------------------------------------------------------
# Encoder construction helpers
# ---------------------------------------------------------------------------


def _build_encoder(
    encoder_type: str, hidden_size: int, patch_size: int, num_hidden_layers: int, head_dim: int = 64
) -> nn.Module:
    """Build a frozen encoder from config (no pretrained download)."""
    num_attention_heads = hidden_size // head_dim  # all supported encoders use head_dim=64

    if encoder_type == "dinov2":
        config = Dinov2WithRegistersConfig(
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_size=518,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
        )
        model = Dinov2WithRegistersModel(config)
        # RAE strips the final layernorm affine params (identity LN). Remove them from
        # the architecture so `from_pretrained` doesn't leave them on the meta device.
        model.layernorm.weight = None
        model.layernorm.bias = None
    elif encoder_type == "siglip2":
        config = SiglipVisionConfig(
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_size=256,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
        )
        model = SiglipVisionModel(config)
        # See dinov2 comment above.
        model.vision_model.post_layernorm.weight = None
        model.vision_model.post_layernorm.bias = None
    elif encoder_type == "mae":
        config = ViTMAEConfig(
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_size=224,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            mask_ratio=0.0,
        )
        model = ViTMAEModel(config)
        # See dinov2 comment above.
        model.layernorm.weight = None
        model.layernorm.bias = None
    else:
        raise ValueError(f"Unknown encoder_type='{encoder_type}'. Available: dinov2, siglip2, mae")

    model.requires_grad_(False)
    return model


_ENCODER_FORWARD_FNS = {
    "dinov2": _dinov2_encoder_forward,
    "siglip2": _siglip2_encoder_forward,
    "mae": _mae_encoder_forward,
}


@dataclass
class RAEDecoderOutput(BaseOutput):
    """
    Output of `RAEDecoder`.

    Args:
        logits (`torch.Tensor`):
            Patch reconstruction logits of shape `(batch_size, num_patches, patch_size**2 * num_channels)`.
    """

    logits: torch.Tensor


class ViTMAEIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "gelu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTMAEOutput(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTMAELayer(nn.Module):
    """
    This matches the naming/parameter structure used in RAE-main (ViTMAE decoder block).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        qkv_bias: bool = True,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_attention_heads}"
            )
        self.attention = Attention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=hidden_size // num_attention_heads,
            dropout=attention_probs_dropout_prob,
            bias=qkv_bias,
        )
        self.intermediate = ViTMAEIntermediate(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act
        )
        self.output = ViTMAEOutput(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_dropout_prob=hidden_dropout_prob
        )
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


class RAEDecoder(nn.Module):
    """
    Decoder implementation ported from RAE-main to keep checkpoint compatibility.

    Key attributes (must match checkpoint keys):
    - decoder_embed
    - decoder_pos_embed
    - decoder_layers
    - decoder_norm
    - decoder_pred
    - trainable_cls_token
    """

    def __init__(
        self,
        hidden_size: int = 768,
        decoder_hidden_size: int = 512,
        decoder_num_hidden_layers: int = 8,
        decoder_num_attention_heads: int = 16,
        decoder_intermediate_size: int = 2048,
        num_patches: int = 256,
        patch_size: int = 16,
        num_channels: int = 3,
        image_size: int = 256,
        qkv_bias: bool = True,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        grid_size = int(num_patches**0.5)
        pos_embed = get_2d_sincos_pos_embed(
            decoder_hidden_size, grid_size, cls_token=True, extra_tokens=1, output_type="pt"
        )
        self.register_buffer("decoder_pos_embed", pos_embed.unsqueeze(0).float(), persistent=False)

        self.decoder_layers = nn.ModuleList(
            [
                ViTMAELayer(
                    hidden_size=decoder_hidden_size,
                    num_attention_heads=decoder_num_attention_heads,
                    intermediate_size=decoder_intermediate_size,
                    qkv_bias=qkv_bias,
                    layer_norm_eps=layer_norm_eps,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    hidden_act=hidden_act,
                )
                for _ in range(decoder_num_hidden_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_hidden_size, eps=layer_norm_eps)
        self.decoder_pred = nn.Linear(decoder_hidden_size, patch_size**2 * num_channels, bias=True)
        self.gradient_checkpointing = False

        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        class_pos_embed = self.decoder_pos_embed[:, 0, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
        dim = self.decoder_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            scale_factor=(1, embeddings_positions / num_positions),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def interpolate_latent(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        h = w = int(l**0.5)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        target_size = (int(self.num_patches**0.5), int(self.num_patches**0.5))
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)
        return x

    def unpatchify(self, patchified_pixel_values: torch.Tensor, original_image_size: tuple[int, int] | None = None):
        patch_size, num_channels = self.patch_size, self.num_channels
        original_image_size = (
            original_image_size if original_image_size is not None else (self.image_size, self.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        interpolate_pos_encoding: bool = False,
        drop_cls_token: bool = False,
        return_dict: bool = True,
    ) -> RAEDecoderOutput | tuple[torch.Tensor]:
        x = self.decoder_embed(hidden_states)
        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x)

        cls_token = self.trainable_cls_token.expand(x_.shape[0], -1, -1)
        x = torch.cat([cls_token, x_], dim=1)

        if interpolate_pos_encoding:
            if not drop_cls_token:
                raise ValueError("interpolate_pos_encoding only supports drop_cls_token=True")
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed

        hidden_states = x + decoder_pos_embed.to(device=x.device, dtype=x.dtype)

        for layer_module in self.decoder_layers:
            hidden_states = layer_module(hidden_states)

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]

        if not return_dict:
            return (logits,)
        return RAEDecoderOutput(logits=logits)


class AutoencoderRAE(ModelMixin, AttentionMixin, AutoencoderMixin, ConfigMixin):
    r"""
    Representation Autoencoder (RAE) model for encoding images to latents and decoding latents to images.

    This model uses a frozen pretrained encoder (DINOv2, SigLIP2, or MAE) with a trainable ViT decoder to reconstruct
    images from learned representations.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented for
    all models (such as downloading or saving).

    Args:
        encoder_type (`str`, *optional*, defaults to `"dinov2"`):
            Type of frozen encoder to use. One of `"dinov2"`, `"siglip2"`, or `"mae"`.
        encoder_hidden_size (`int`, *optional*, defaults to `768`):
            Hidden size of the encoder model.
        encoder_patch_size (`int`, *optional*, defaults to `14`):
            Patch size of the encoder model.
        encoder_num_hidden_layers (`int`, *optional*, defaults to `12`):
            Number of hidden layers in the encoder model.
        patch_size (`int`, *optional*, defaults to `16`):
            Decoder patch size (used for unpatchify and decoder head).
        encoder_input_size (`int`, *optional*, defaults to `224`):
            Input size expected by the encoder.
        image_size (`int`, *optional*):
            Decoder output image size. If `None`, it is derived from encoder token count and `patch_size` like
            RAE-main: `image_size = patch_size * sqrt(num_patches)`, where `num_patches = (encoder_input_size //
            encoder_patch_size) ** 2`.
        num_channels (`int`, *optional*, defaults to `3`):
            Number of input/output channels.
        encoder_norm_mean (`list`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            Channel-wise mean for encoder input normalization (ImageNet defaults).
        encoder_norm_std (`list`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            Channel-wise std for encoder input normalization (ImageNet defaults).
        latents_mean (`list` or `tuple`, *optional*):
            Optional mean for latent normalization. Tensor inputs are accepted and converted to config-serializable
            lists.
        latents_std (`list` or `tuple`, *optional*):
            Optional standard deviation for latent normalization. Tensor inputs are accepted and converted to
            config-serializable lists.
        noise_tau (`float`, *optional*, defaults to `0.0`):
            Noise level for training (adds noise to latents during training).
        reshape_to_2d (`bool`, *optional*, defaults to `True`):
            Whether to reshape latents to 2D (B, C, H, W) format.
        use_encoder_loss (`bool`, *optional*, defaults to `False`):
            Whether to use encoder hidden states in the loss (for advanced training).
    """

    # NOTE: gradient checkpointing is not wired up for this model yet.
    _supports_gradient_checkpointing = False
    _no_split_modules = ["ViTMAELayer"]
    _keys_to_ignore_on_load_unexpected = ["decoder.decoder_pos_embed"]

    @register_to_config
    def __init__(
        self,
        encoder_type: str = "dinov2",
        encoder_hidden_size: int = 768,
        encoder_patch_size: int = 14,
        encoder_num_hidden_layers: int = 12,
        decoder_hidden_size: int = 512,
        decoder_num_hidden_layers: int = 8,
        decoder_num_attention_heads: int = 16,
        decoder_intermediate_size: int = 2048,
        patch_size: int = 16,
        encoder_input_size: int = 224,
        image_size: int | None = None,
        num_channels: int = 3,
        encoder_norm_mean: list | None = None,
        encoder_norm_std: list | None = None,
        latents_mean: list | tuple | torch.Tensor | None = None,
        latents_std: list | tuple | torch.Tensor | None = None,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        use_encoder_loss: bool = False,
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        if encoder_type not in _ENCODER_FORWARD_FNS:
            raise ValueError(
                f"Unknown encoder_type='{encoder_type}'. Available: {sorted(_ENCODER_FORWARD_FNS.keys())}"
            )

        def _to_config_compatible(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, tuple):
                return [_to_config_compatible(v) for v in value]
            if isinstance(value, list):
                return [_to_config_compatible(v) for v in value]
            return value

        def _as_optional_tensor(value: torch.Tensor | list | tuple | None) -> torch.Tensor | None:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            return torch.tensor(value, dtype=torch.float32)

        latents_std_tensor = _as_optional_tensor(latents_std)

        # Ensure config values are JSON-serializable (list/None), even if caller passes torch.Tensors.
        self.register_to_config(
            latents_mean=_to_config_compatible(latents_mean),
            latents_std=_to_config_compatible(latents_std),
        )

        self.encoder_input_size = encoder_input_size
        self.noise_tau = float(noise_tau)
        self.reshape_to_2d = bool(reshape_to_2d)
        self.use_encoder_loss = bool(use_encoder_loss)

        # Validate early, before building the (potentially large) encoder/decoder.
        encoder_patch_size = int(encoder_patch_size)
        if self.encoder_input_size % encoder_patch_size != 0:
            raise ValueError(
                f"encoder_input_size={self.encoder_input_size} must be divisible by encoder_patch_size={encoder_patch_size}."
            )
        decoder_patch_size = int(patch_size)
        if decoder_patch_size <= 0:
            raise ValueError("patch_size must be a positive integer (this is decoder_patch_size).")

        # Frozen representation encoder (built from config, no downloads)
        self.encoder: nn.Module = _build_encoder(
            encoder_type=encoder_type,
            hidden_size=encoder_hidden_size,
            patch_size=encoder_patch_size,
            num_hidden_layers=encoder_num_hidden_layers,
        )
        self._encoder_forward_fn = _ENCODER_FORWARD_FNS[encoder_type]
        num_patches = (self.encoder_input_size // encoder_patch_size) ** 2

        grid = int(sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"Computed num_patches={num_patches} must be a perfect square.")

        derived_image_size = decoder_patch_size * grid
        if image_size is None:
            image_size = derived_image_size
        else:
            image_size = int(image_size)
            if image_size != derived_image_size:
                raise ValueError(
                    f"image_size={image_size} must equal decoder_patch_size*sqrt(num_patches)={derived_image_size} "
                    f"for patch_size={decoder_patch_size} and computed num_patches={num_patches}."
                )

        # Encoder input normalization stats (ImageNet defaults)
        if encoder_norm_mean is None:
            encoder_norm_mean = [0.485, 0.456, 0.406]
        if encoder_norm_std is None:
            encoder_norm_std = [0.229, 0.224, 0.225]
        encoder_mean_tensor = torch.tensor(encoder_norm_mean, dtype=torch.float32).view(1, 3, 1, 1)
        encoder_std_tensor = torch.tensor(encoder_norm_std, dtype=torch.float32).view(1, 3, 1, 1)

        self.register_buffer("encoder_mean", encoder_mean_tensor, persistent=True)
        self.register_buffer("encoder_std", encoder_std_tensor, persistent=True)

        # Latent normalization buffers (defaults are no-ops; actual values come from checkpoint)
        latents_mean_tensor = _as_optional_tensor(latents_mean)
        if latents_mean_tensor is None:
            latents_mean_tensor = torch.zeros(1)
        self.register_buffer("_latents_mean", latents_mean_tensor, persistent=True)

        if latents_std_tensor is None:
            latents_std_tensor = torch.ones(1)
        self.register_buffer("_latents_std", latents_std_tensor, persistent=True)

        # ViT-MAE style decoder
        self.decoder = RAEDecoder(
            hidden_size=int(encoder_hidden_size),
            decoder_hidden_size=int(decoder_hidden_size),
            decoder_num_hidden_layers=int(decoder_num_hidden_layers),
            decoder_num_attention_heads=int(decoder_num_attention_heads),
            decoder_intermediate_size=int(decoder_intermediate_size),
            num_patches=int(num_patches),
            patch_size=int(decoder_patch_size),
            num_channels=int(num_channels),
            image_size=int(image_size),
        )
        self.num_patches = int(num_patches)
        self.decoder_patch_size = int(decoder_patch_size)
        self.decoder_image_size = int(image_size)

        # Slicing support (batch dimension) similar to other diffusers autoencoders
        self.use_slicing = False

    def _noising(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        # Per-sample random sigma in [0, noise_tau]
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype, generator=generator
        )
        return x + noise_sigma * randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size), mode="bicubic", align_corners=False
            )
        mean = self.encoder_mean.to(device=x.device, dtype=x.dtype)
        std = self.encoder_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _denormalize_image(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.encoder_mean.to(device=x.device, dtype=x.dtype)
        std = self.encoder_std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        latents_mean = self._latents_mean.to(device=z.device, dtype=z.dtype)
        latents_std = self._latents_std.to(device=z.device, dtype=z.dtype)
        return (z - latents_mean) / (latents_std + 1e-5)

    def _denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        latents_mean = self._latents_mean.to(device=z.device, dtype=z.dtype)
        latents_std = self._latents_std.to(device=z.device, dtype=z.dtype)
        return z * (latents_std + 1e-5) + latents_mean

    def _encode(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        x = self._resize_and_normalize(x)

        if self.config.encoder_type == "mae":
            tokens = self._encoder_forward_fn(self.encoder, x, self.config.encoder_patch_size)
        else:
            tokens = self._encoder_forward_fn(self.encoder, x)  # (B, N, C)

        if self.training and self.noise_tau > 0:
            tokens = self._noising(tokens, generator=generator)

        if self.reshape_to_2d:
            b, n, c = tokens.shape
            side = int(sqrt(n))
            if side * side != n:
                raise ValueError(f"Token length n={n} is not a perfect square; cannot reshape to 2D.")
            z = tokens.transpose(1, 2).contiguous().view(b, c, side, side)  # (B, C, h, w)
        else:
            z = tokens

        z = self._normalize_latents(z)

        # Follow diffusers convention: optionally scale latents for diffusion
        if self.config.scaling_factor != 1.0:
            z = z * self.config.scaling_factor

        return z

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True, generator: torch.Generator | None = None
    ) -> EncoderOutput | tuple[torch.Tensor]:
        if self.use_slicing and x.shape[0] > 1:
            latents = torch.cat([self._encode(x_slice, generator=generator) for x_slice in x.split(1)], dim=0)
        else:
            latents = self._encode(x, generator=generator)

        if not return_dict:
            return (latents,)
        return EncoderOutput(latent=latents)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        # Undo scaling factor if applied at encode time
        if self.config.scaling_factor != 1.0:
            z = z / self.config.scaling_factor

        z = self._denormalize_latents(z)

        if self.reshape_to_2d:
            b, c, h, w = z.shape
            tokens = z.view(b, c, h * w).transpose(1, 2).contiguous()  # (B, N, C)
        else:
            tokens = z

        logits = self.decoder(tokens, return_dict=True).logits
        x_rec = self.decoder.unpatchify(logits)
        x_rec = self._denormalize_image(x_rec)
        return x_rec.to(device=z.device)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)], dim=0)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self, sample: torch.Tensor, return_dict: bool = True, generator: torch.Generator | None = None
    ) -> DecoderOutput | tuple[torch.Tensor]:
        latents = self.encode(sample, return_dict=False, generator=generator)[0]
        decoded = self.decode(latents, return_dict=False)[0]
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
