"""
ATTENTION: OUTDATED/ARCHIVED - FULL 5-Channel Multispectral AutoencoderKL Implementation

This module implements a Variational Autoencoder (VAE) specifically designed for 5-channel multispectral image data.
The implementation extends the standard AutoencoderKL from diffusers to handle 5-channel multispectral data
while maintaining compatibility with Stable Diffusion 3's latent space requirements.

Research Context:
- Multispectral imagery typically consists of 5 spectral bands: Blue, Green, Red, Near-Infrared (NIR), and Short-Wave Infrared (SWIR)
- Each spectral band contains unique information about the scene's reflectance properties
- The challenge is to compress this information while preserving spectral characteristics
- Maintaining compatibility with SD3's latent space (4 channels) is crucial for integration

Implementation Details:
1. Architecture:
   - Extends AutoencoderKL with 5 input/output channels
   - Maintains 4-channel latent space for SD3 compatibility
   - Uses 4 downsampling blocks to achieve 8x downsampling (matching SD3)
   - Implements group normalization (32 groups) for stable training

2. Key Design Decisions:
   - Preserves spectral information through careful normalization
   - Uses group normalization to handle increased channel count
   - Maintains same latent space dimensions as SD3 (8x downsampling)
   - Implements proper scaling and shifting of latent space

3. Technical Considerations:
   - Handles 16-bit multispectral data
   - Preserves relative differences between spectral bands
   - Ensures stable training through proper initialization
   - Maintains compatibility with existing diffusers pipelines

The implementation follows these scientific principles:
- Reproducibility: All components are deterministic where possible
- Modularity: Clear separation of encoder and decoder components
- Extensibility: Easy to modify for different spectral configurations
- Compatibility: Maintains interface with existing diffusers components

This implementation is crucial for:
1. Enabling multispectral image generation with diffusion models
2. Preserving spectral information in the latent space
3. Maintaining compatibility with existing pipelines
4. Providing a foundation for future multispectral research
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .autoencoder_kl import AutoencoderKL
from .vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder


class AutoencoderKLMultispectral5Ch(AutoencoderKL):
    r"""
    A VAE model with KL loss for encoding 5-channel multispectral images into latents and decoding latent representations 
    into multispectral images. This model extends AutoencoderKL to support 5 input channels while maintaining
    compatibility with Stable Diffusion 3.

    This model inherits from [`AutoencoderKL`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 5): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 5): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")`):
            Tuple of downsample block types. Uses 4 blocks to achieve 8x downsampling.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")`):
            Tuple of upsample block types. Matches down_block_types for symmetry.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64, 128, 256, 512)`):
            Tuple of block output channels. Matches SD3's channel progression.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 5,  # 5 spectral bands
        out_channels: int = 5,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels: Tuple[int] = (64, 128, 256, 512),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        """
        Initialize the 5-channel multispectral VAE.
        
        Args:
            in_channels: Number of input channels (5 for multispectral)
            out_channels: Number of output channels (5 for multispectral)
            down_block_types: Types of downsampling blocks (4 blocks for 8x downsampling)
            up_block_types: Types of upsampling blocks (matches downsampling)
            block_out_channels: Number of channels in each block (matches SD3)
            layers_per_block: Number of layers in each block
            act_fn: Activation function to use
            latent_channels: Number of channels in latent space (4 for SD3 compatibility)
            norm_num_groups: Number of groups for group normalization
            sample_size: Input sample size
            scaling_factor: Scaling factor for latent space
            shift_factor: Optional shift factor for latent space
            latents_mean: Optional mean for latent space
            latents_std: Optional standard deviation for latent space
            force_upcast: Whether to force upcasting to float32
            use_quant_conv: Whether to use quantized convolutions
            use_post_quant_conv: Whether to use post-quantization convolutions
            mid_block_add_attention: Whether to add attention in the middle block
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            force_upcast=force_upcast,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            mid_block_add_attention=mid_block_add_attention,
        )

        # The latent space dimensions remain the same as the parent class to maintain compatibility with SD3
        # This is crucial as the transformer expects a specific latent space structure 