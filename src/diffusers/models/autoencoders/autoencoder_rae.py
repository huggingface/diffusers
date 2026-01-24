from dataclasses import dataclass
from math import sqrt
from typing import Dict, Type, Optional, Callable, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import BaseOutput
from ...utils.accelerate_utils import apply_forward_hook
from ..attention import FeedForward
from ..attention_processor import Attention
from ..modeling_utils import ModelMixin

ENCODER_ARCHS: Dict[str, Type] = {}

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
    def __init__(
        self,
        encoder_name_or_path: str = "facebook/dinov2-with-registers-base"
    ):
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
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """
        outputs = self.model(images, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = outputs.last_hidden_state[:, unused_token_num:]
        return image_features


@register_encoder(name="siglip2")
class Siglip2Encoder(nn.Module):
    def __init__(
        self,
        encoder_name_or_path: str = "google/siglip2-base-patch16-256"
    ):
        super().__init__()
        from transformers import SiglipModel
        self.model = SiglipModel.from_pretrained(encoder_name_or_path).vision_model
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
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """
        outputs = self.model(images, output_hidden_states=True, interpolate_pos_encoding = True)
        image_features = outputs.last_hidden_state
        return image_features


@register_encoder(name="mae")
class MAEEncoder(nn.Module):
    def __init__(self, encoder_name_or_path: str = "facebook/vit-mae-base"):
        super().__init__()
        from transformers import ViTMAEForPreTraining
        self.model = ViTMAEForPreTraining.from_pretrained(encoder_name_or_path).vit
        # remove the affine of final layernorm
        self.model.layernorm.elementwise_affine = False
        # remove the param
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.model.config.mask_ratio = 0. # no masking

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and width of the image
        """
        h,w = images.shape[2], images.shape[3]
        patch_num = int(h * w  // self.patch_size ** 2)
        assert patch_num * self.patch_size ** 2 == h * w, 'image size should be divisible by patch size'
        noise = torch.arange(patch_num).unsqueeze(0).expand(images.shape[0],-1).to(images.device).to(images.dtype)
        outputs = self.model(images, noise, interpolate_pos_encoding = True)
        image_features = outputs.last_hidden_state[:, 1:] # remove cls token
        return image_features


@dataclass
class AutoencoderRAEOutput(BaseOutput):
    """
    Output of AutoencoderRAE encoding method.

    Args:
        latent (`torch.Tensor`):
            Encoded outputs of the encoder (frozen representation encoder).
            Shape: (batch_size, hidden_size, latent_height, latent_width)
    """

    latent: torch.Tensor


class RAEDecoderOutput(BaseOutput):
    """
    Output of RAEDecoder.

    Args:
        sample (`torch.Tensor`):
            Decoded output from decoder. Shape: (batch_size, num_channels, image_height, image_width)
    """

    sample: torch.Tensor




class AutoencoderRAE(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin
):
    r"""
    Representation Autoencoder (RAE) model for encoding images to latents and decoding latents to images.

    This model uses a frozen pretrained encoder (DINOv2, SigLIP2, or MAE) with a trainable ViT decoder
    to reconstruct images from learned representations.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods
    implemented for all models (such as downloading or saving).

    Args:
        encoder_cls (`str`, *optional*, defaults to `"dinov2"`):
            Type of frozen encoder to use. One of `"dinov2"`, `"siglip2"`, or `"mae"`.
        encoder_name_or_path (`str`, *optional*, defaults to `"facebook/dinov2-with-registers-base"`):
            Path to pretrained encoder model or model identifier from huggingface.co/models.
        decoder_config (`ViTMAEDecoderConfig`, *optional*):
            Configuration for the decoder. If None, a default config will be used.
        num_patches (`int`, *optional*, defaults to `196`):
            Number of patches in the latent space (14x14 = 196 for 224x224 image with patch size 16).
        patch_size (`int`, *optional*, defaults to `16`):
            Patch size for both encoder and decoder.
        encoder_input_size (`int`, *optional*, defaults to `224`):
            Input size expected by the encoder.
        image_size (`int`, *optional*, defaults to `256`):
            Output image size.
        num_channels (`int`, *optional*, defaults to `3`):
            Number of input/output channels.
        latent_mean (`torch.Tensor`, *optional*):
            Optional mean for latent normalization.
        latent_var (`torch.Tensor`, *optional*):
            Optional variance for latent normalization.
        noise_tau (`float`, *optional*, defaults to `0.0`):
            Noise level for training (adds noise to latents during training).
        reshape_to_2d (`bool`, *optional*, defaults to `True`):
            Whether to reshape latents to 2D (B, C, H, W) format.
        use_encoder_loss (`bool`, *optional*, defaults to `False`):
            Whether to use encoder hidden states in the loss (for advanced training).
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ViTMAEDecoderLayer"]

    @register_to_config
    def __init__(
        self,
        encoder_cls: str = "dinov2",
        encoder_name_or_path: str = "facebook/dinov2-with-registers-base",
        decoder_config: str = None,
        num_patches: int = 196,
        patch_size: int = 16,
        encoder_input_size: int = 224,
        image_size: int = 256,
        num_channels: int = 3,
        latent_mean: Optional[torch.Tensor] = None,
        latent_var: Optional[torch.Tensor] = None,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        use_encoder_loss: bool = False,
    ):
        super().__init__()

