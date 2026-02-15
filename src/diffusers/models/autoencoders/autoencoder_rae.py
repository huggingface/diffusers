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

from dataclasses import dataclass
from math import sqrt
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import BaseOutput, logging
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, EncoderOutput


ENCODER_ARCHS: Dict[str, Type] = {}
ENCODER_DEFAULT_NAME_OR_PATH = {
    "dinov2": "facebook/dinov2-with-registers-base",
    "siglip2": "google/siglip2-base-patch16-256",
    "mae": "facebook/vit-mae-base",
}
logger = logging.get_logger(__name__)


def register_encoder(cls: Optional[Type] = None, *, name: Optional[str] = None) -> Union[Callable[[Type], Type], Type]:
    def decorator(inner_cls: Type) -> Type:
        encoder_name = name or inner_cls.__name__
        if encoder_name in ENCODER_ARCHS and ENCODER_ARCHS[encoder_name] is not inner_cls:
            raise ValueError(f"Encoder '{encoder_name}' is already registered.")
        ENCODER_ARCHS[encoder_name] = inner_cls
        return inner_cls

    if cls is None:
        return decorator
    return decorator(cls)


@register_encoder(name="dinov2")
class Dinov2Encoder(nn.Module):
    def __init__(self, encoder_name_or_path: str = "facebook/dinov2-with-registers-base"):
        super().__init__()
        from transformers import Dinov2WithRegistersModel

        self.model = Dinov2WithRegistersModel.from_pretrained(encoder_name_or_path)
        self.model.requires_grad_(False)
        self.model.layernorm.elementwise_affine = False
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None

        self.patch_size = self.model.config.patch_size
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images is of shape (B, C, H, W) where B is batch size, C is number of channels, H and W are height and
        """
        outputs = self.model(images, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = outputs.last_hidden_state[:, unused_token_num:]
        return image_features


@register_encoder(name="siglip2")
class Siglip2Encoder(nn.Module):
    def __init__(self, encoder_name_or_path: str = "google/siglip2-base-patch16-256"):
        super().__init__()
        from transformers import SiglipModel

        self.model = SiglipModel.from_pretrained(encoder_name_or_path).vision_model
        self.model.requires_grad_(False)
        # remove the affine of final layernorm
        self.model.post_layernorm.elementwise_affine = False
        # remove the param
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images is of shape (B, C, H, W) where B is batch size, C is number of channels, H and W are height and
        """
        outputs = self.model(images, output_hidden_states=True, interpolate_pos_encoding=True)
        image_features = outputs.last_hidden_state
        return image_features


@register_encoder(name="mae")
class MAEEncoder(nn.Module):
    def __init__(self, encoder_name_or_path: str = "facebook/vit-mae-base"):
        super().__init__()
        from transformers import ViTMAEForPreTraining

        self.model = ViTMAEForPreTraining.from_pretrained(encoder_name_or_path).vit
        self.model.requires_grad_(False)
        # remove the affine of final layernorm
        self.model.layernorm.elementwise_affine = False
        # remove the param
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.model.config.mask_ratio = 0.0  # no masking

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images is of shape (B, C, H, W) where B is batch size, C is number of channels, H and W are height and width of
        the image
        """
        h, w = images.shape[2], images.shape[3]
        patch_num = int(h * w // self.patch_size**2)
        if patch_num * self.patch_size**2 != h * w:
            raise ValueError("Image size should be divisible by patch size.")
        noise = torch.arange(patch_num).unsqueeze(0).expand(images.shape[0], -1).to(images.device).to(images.dtype)
        outputs = self.model(images, noise, interpolate_pos_encoding=True)
        image_features = outputs.last_hidden_state[:, 1:]  # remove cls token
        return image_features


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, add_cls_token: bool = False) -> np.ndarray:
    """
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0)  # (2, grid, grid)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float64), pos_embed], axis=0)

    return pos_embed


@dataclass
class RAEDecoderOutput(BaseOutput):
    """
    Output of `RAEDecoder`.

    Args:
        logits (`torch.Tensor`):
            Patch reconstruction logits of shape `(batch_size, num_patches, patch_size**2 * num_channels)`.
    """

    logits: torch.Tensor


ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
}


class ViTMAESelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool = True, attn_dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_attention_heads}"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.dropout = nn.Dropout(attn_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size**0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class ViTMAESelfOutput(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTMAEAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool = True, attn_dropout: float = 0.0):
        super().__init__()
        self.attention = ViTMAESelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
        )
        self.output = ViTMAESelfOutput(hidden_size=hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(hidden_states)
        attn_output = self.output(attn_output)
        return attn_output


class ViTMAEIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "gelu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN.get(hidden_act, None)
        if self.intermediate_act_fn is None:
            raise ValueError(f"Unsupported hidden_act={hidden_act}")

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
        self.attention = ViTMAEAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attention_probs_dropout_prob,
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


class GeneralDecoder(nn.Module):
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

    def __init__(self, config, num_patches: int):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )

        self.decoder_layers = nn.ModuleList(
            [
                ViTMAELayer(
                    hidden_size=config.decoder_hidden_size,
                    num_attention_heads=config.decoder_num_attention_heads,
                    intermediate_size=config.decoder_intermediate_size,
                    qkv_bias=config.qkv_bias,
                    layer_norm_eps=config.layer_norm_eps,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    hidden_act=config.hidden_act,
                )
                for _ in range(config.decoder_num_hidden_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )
        self.gradient_checkpointing = False
        self.config = config
        self.num_patches = num_patches

        self._initialize_weights(num_patches)
        self.set_trainable_cls_token()

    def set_trainable_cls_token(self, tensor: Optional[torch.Tensor] = None):
        tensor = torch.zeros(1, 1, self.config.decoder_hidden_size) if tensor is None else tensor
        self.trainable_cls_token = nn.Parameter(tensor)

    def _initialize_weights(self, num_patches: int):
        grid_size = int(num_patches**0.5)
        pos_embed = _get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, add_cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    def unpatchify(self, patchified_pixel_values: torch.Tensor, original_image_size: Optional[Tuple[int, int]] = None):
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
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
    ) -> Union[RAEDecoderOutput, Tuple[torch.Tensor]]:
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


# Backward-compatible alias: keep `RAEDecoder` name used by `AutoencoderRAE`
class RAEDecoder(GeneralDecoder):
    pass


class AutoencoderRAE(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    Representation Autoencoder (RAE) model for encoding images to latents and decoding latents to images.

    This model uses a frozen pretrained encoder (DINOv2, SigLIP2, or MAE) with a trainable ViT decoder to reconstruct
    images from learned representations.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented for
    all models (such as downloading or saving).

    Args:
        encoder_cls (`str`, *optional*, defaults to `"dinov2"`):
            Type of frozen encoder to use. One of `"dinov2"`, `"siglip2"`, or `"mae"`.
        encoder_name_or_path (`str`, *optional*):
            Path to pretrained encoder model or model identifier from huggingface.co/models. If not provided, uses an
            encoder-specific default model id.
        patch_size (`int`, *optional*, defaults to `16`):
            Decoder patch size (used for unpatchify and decoder head).
        encoder_input_size (`int`, *optional*, defaults to `224`):
            Input size expected by the encoder.
        image_size (`int`, *optional*):
            Decoder output image size. If `None`, it is derived from encoder token count and `patch_size` like
            RAE-main: `image_size = patch_size * sqrt(num_patches)`, where `num_patches = (encoder_input_size //
            encoder.patch_size) ** 2`.
        num_channels (`int`, *optional*, defaults to `3`):
            Number of input/output channels.
        latent_mean (`list` or `tuple`, *optional*):
            Optional mean for latent normalization. Tensor inputs are accepted for backward compatibility and converted
            to config-serializable lists.
        latent_var (`list` or `tuple`, *optional*):
            Optional variance for latent normalization. Tensor inputs are accepted for backward compatibility and
            converted to config-serializable lists.
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

    @register_to_config
    def __init__(
        self,
        encoder_cls: str = "dinov2",
        encoder_name_or_path: Optional[str] = None,
        decoder_hidden_size: int = 512,
        decoder_num_hidden_layers: int = 8,
        decoder_num_attention_heads: int = 16,
        decoder_intermediate_size: int = 2048,
        patch_size: int = 16,
        encoder_input_size: int = 224,
        image_size: Optional[int] = None,
        num_channels: int = 3,
        latent_mean: Optional[Union[list, tuple, torch.Tensor]] = None,
        latent_var: Optional[Union[list, tuple, torch.Tensor]] = None,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        use_encoder_loss: bool = False,
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        if encoder_cls not in ENCODER_ARCHS:
            raise ValueError(f"Unknown encoder_cls='{encoder_cls}'. Available: {sorted(ENCODER_ARCHS.keys())}")
        if encoder_name_or_path is None:
            encoder_name_or_path = ENCODER_DEFAULT_NAME_OR_PATH[encoder_cls]

        def _to_config_compatible(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, tuple):
                return [_to_config_compatible(v) for v in value]
            if isinstance(value, list):
                return [_to_config_compatible(v) for v in value]
            return value

        # Ensure config values are JSON-serializable (list/None), even if caller passes torch.Tensors.
        self.register_to_config(
            encoder_name_or_path=encoder_name_or_path,
            latent_mean=_to_config_compatible(latent_mean),
            latent_var=_to_config_compatible(latent_var),
        )

        self.encoder_input_size = encoder_input_size
        self.noise_tau = float(noise_tau)
        self.reshape_to_2d = bool(reshape_to_2d)
        self.use_encoder_loss = bool(use_encoder_loss)

        # Frozen representation encoder
        self.encoder: nn.Module = ENCODER_ARCHS[encoder_cls](encoder_name_or_path=encoder_name_or_path)

        # RAE-main: base_patches = (encoder_input_size // encoder_patch_size) ** 2
        encoder_patch_size = getattr(self.encoder, "patch_size", None)
        if encoder_patch_size is None:
            raise ValueError(f"Encoder '{encoder_cls}' must define `.patch_size` attribute.")
        encoder_patch_size = int(encoder_patch_size)
        if self.encoder_input_size % encoder_patch_size != 0:
            raise ValueError(
                f"encoder_input_size={self.encoder_input_size} must be divisible by encoder.patch_size={encoder_patch_size}."
            )
        num_patches = (self.encoder_input_size // encoder_patch_size) ** 2

        # Decoder patch size is independent from encoder patch size.
        decoder_patch_size = int(patch_size)
        if decoder_patch_size <= 0:
            raise ValueError("patch_size must be a positive integer (this is decoder_patch_size).")

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

        # Normalization stats from the encoder's image processor
        # RAE-main uses AutoImageProcessor mean/std; we follow the same.
        encoder_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
        encoder_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
        try:
            from transformers import AutoImageProcessor

            try:
                proc = AutoImageProcessor.from_pretrained(encoder_name_or_path, local_files_only=True)
            except Exception:
                proc = AutoImageProcessor.from_pretrained(encoder_name_or_path, local_files_only=False)
            encoder_mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
            encoder_std = torch.tensor(proc.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        except (OSError, ValueError):
            # Keep default 0.5/0.5 if processor is unavailable.
            logger.warning(
                "Falling back to encoder mean/std [0.5, 0.5, 0.5] for `%s` because AutoImageProcessor could not be loaded.",
                encoder_name_or_path,
            )

        self.register_buffer("encoder_mean", encoder_mean, persistent=True)
        self.register_buffer("encoder_std", encoder_std, persistent=True)

        # Optional latent normalization (RAE-main uses mean/var)
        def _as_optional_tensor(value: Optional[Union[torch.Tensor, list, tuple]]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            return torch.tensor(value, dtype=torch.float32)

        latent_mean_tensor = _as_optional_tensor(latent_mean)
        latent_var_tensor = _as_optional_tensor(latent_var)
        self.do_latent_normalization = latent_mean is not None or latent_var is not None
        if latent_mean_tensor is not None:
            self.register_buffer("_latent_mean", latent_mean_tensor, persistent=True)
        else:
            self._latent_mean = None
        if latent_var_tensor is not None:
            self.register_buffer("_latent_var", latent_var_tensor, persistent=True)
        else:
            self._latent_var = None

        # ViT-MAE style decoder
        encoder_hidden_size = getattr(self.encoder, "hidden_size", None)
        if encoder_hidden_size is None:
            raise ValueError(f"Encoder '{encoder_cls}' must define `.hidden_size` attribute.")

        decoder_config = SimpleNamespace(
            hidden_size=int(encoder_hidden_size),
            decoder_hidden_size=int(decoder_hidden_size),
            decoder_num_hidden_layers=int(decoder_num_hidden_layers),
            decoder_num_attention_heads=int(decoder_num_attention_heads),
            decoder_intermediate_size=int(decoder_intermediate_size),
            patch_size=int(decoder_patch_size),
            image_size=int(image_size),
            num_channels=int(num_channels),
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            hidden_act="gelu",
        )
        self.decoder = RAEDecoder(decoder_config, num_patches=int(num_patches))
        self.num_patches = int(num_patches)
        self.decoder_patch_size = int(decoder_patch_size)
        self.decoder_image_size = int(image_size)

        # Slicing support (batch dimension) similar to other diffusers autoencoders
        self.use_slicing = False

    def _noising(self, x: torch.Tensor) -> torch.Tensor:
        # Per-sample random sigma in [0, noise_tau]
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype)
        return x + noise_sigma * torch.randn_like(x)

    def _maybe_resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size), mode="bicubic", align_corners=False
            )
        mean = self.encoder_mean.to(device=x.device, dtype=x.dtype)
        std = self.encoder_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _maybe_denormalize_image(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.encoder_mean.to(device=x.device, dtype=x.dtype)
        std = self.encoder_std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _maybe_normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        if not self.do_latent_normalization:
            return z
        latent_mean = self._latent_mean.to(device=z.device, dtype=z.dtype) if self._latent_mean is not None else 0
        latent_var = self._latent_var.to(device=z.device, dtype=z.dtype) if self._latent_var is not None else 1
        return (z - latent_mean) / torch.sqrt(latent_var + 1e-5)

    def _maybe_denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        if not self.do_latent_normalization:
            return z
        latent_mean = self._latent_mean.to(device=z.device, dtype=z.dtype) if self._latent_mean is not None else 0
        latent_var = self._latent_var.to(device=z.device, dtype=z.dtype) if self._latent_var is not None else 1
        return z * torch.sqrt(latent_var + 1e-5) + latent_mean

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_resize_and_normalize(x)

        # Encoder is frozen; many encoders already run under no_grad
        tokens = self.encoder(x)  # (B, N, C)

        if self.training and self.noise_tau > 0:
            tokens = self._noising(tokens)

        if self.reshape_to_2d:
            b, n, c = tokens.shape
            side = int(sqrt(n))
            if side * side != n:
                raise ValueError(f"Token length n={n} is not a perfect square; cannot reshape to 2D.")
            z = tokens.transpose(1, 2).contiguous().view(b, c, side, side)  # (B, C, h, w)
        else:
            z = tokens

        z = self._maybe_normalize_latents(z)

        # Follow diffusers convention: optionally scale latents for diffusion
        if self.config.scaling_factor != 1.0:
            z = z * self.config.scaling_factor

        return z

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[EncoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and x.shape[0] > 1:
            latents = torch.cat([self._encode(x_slice) for x_slice in x.split(1)], dim=0)
        else:
            latents = self._encode(x)

        if not return_dict:
            return (latents,)
        return EncoderOutput(latent=latents)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        # Undo scaling factor if applied at encode time
        if self.config.scaling_factor != 1.0:
            z = z / self.config.scaling_factor

        z = self._maybe_denormalize_latents(z)

        if self.reshape_to_2d:
            b, c, h, w = z.shape
            tokens = z.view(b, c, h * w).transpose(1, 2).contiguous()  # (B, N, C)
        else:
            tokens = z

        logits = self.decoder(tokens, return_dict=True).logits
        x_rec = self.decoder.unpatchify(logits)
        x_rec = self._maybe_denormalize_image(x_rec)
        return x_rec

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)], dim=0)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        latents = self.encode(sample, return_dict=False)[0]
        decoded = self.decode(latents, return_dict=False)[0]
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
