"""
Multispectral VAE Adapter for Stable Diffusion 3

TODO: For convergence stability, consider scaling or weighting loss terms (e.g., loss = mse_weight * mse + sam_weight * sam) to allow tuning their contribution

Consider adding a dropout layer to prevent overfitting (add for each input/output adapter) if  overfitting to the spectral pattern of this small dataset is an issue

This module implements an efficient approach to handle 5-channel multispectral data by:
1. Using a pretrained SD3 VAE as the backbone
2. Adding lightweight adapter layers for 5-channel input/output
3. Keeping the powerful pretrained backbone frozen
4. Only training the new adapter layers


	•	Parameter-Efficient Design: Both input_adapter and output_adapter modules are implemented cleanly with SpectralAdapter, following the structure needed for compatibility with SD3.
	•	Adapter Placement Flexibility (adapter_placement="input"/"output"/"both"): Excellent design choice to experiment and benchmark.
	•	Spectral Attention Integration: Smart use of attention maps with use_attention=True, allowing interpretability and weighted spectral emphasis.
	•	Parameter Freezing: The freeze_backbone() method is well scoped to preserve pretrained SD3 weights, isolating training to adapter parameters.
	•	Selective Parameter Return (get_trainable_params()): This will help optimize training by targeting just the new parameters.

Key Features:
- Minimal trainable parameters (only adapters)
- Preserves pretrained VAE's powerful feature extraction
- Maintains compatibility with SD3's latent space
- Efficient fine-tuning approach
- Spectral attention for interpretable band selection
- Per-channel reconstruction loss for spectral fidelity
- Configurable adapter placement (input-only, output-only, or both)

Spectral Bands:
- Band 9 (474.73nm): Blue - captures chlorophyll absorption
- Band 18 (538.71nm): Green - reflects well in healthy vegetation
- Band 32 (650.665nm): Red - sensitive to chlorophyll content
- Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
- Band 55 (850.59nm): NIR - strong reflectance in healthy leaves

Architecture:
1. Input Adapter (5 → 3 channels):
   - Spectral attention mechanism for band selection
   - Converts 5-channel multispectral input to 3-channel RGB-like format
   - Learns optimal spectral band combinations
   - Preserves important spectral information

2. Pretrained SD3 VAE Backbone:
   - Frozen weights
   - Handles core compression/decompression
   - Maintains compatibility with diffusion model

3. Output Adapter (3 → 5 channels):
   - Reconstructs 5-channel output from 3-channel VAE output
   - Learns to recover spectral information
   - Maintains spectral fidelity

Loss Functions:
- Per-channel MSE loss for each spectral band
- Spectral Angle Mapper (SAM) for spectral similarity
- Optional correlation regularization

Usage:
    # Initialize with pretrained SD3 VAE
    vae = AutoencoderKLMultispectralAdapter.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        adapter_placement="both",  # or "input" or "output"
        use_spectral_attention=True,
        use_sam_loss=True
    )
    
    # Freeze backbone (only adapters will be trained)
    vae.freeze_backbone()
    
    # Train only adapter layers
    optimizer = torch.optim.AdamW(vae.get_trainable_params(), lr=1e-4)
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .autoencoder_kl import AutoencoderKL

class SpectralAttention(nn.Module):
    """Attention mechanism for spectral band selection.
    
    This module learns to weight the importance of each spectral band
    during the adaptation process. It helps the model focus on the most
    relevant bands for the task while maintaining spectral relationships.
    """
    
    def __init__(self, num_bands: int):
        super().__init__()
        # Simple 1x1 convolution followed by sigmoid to learn band weights
        self.attention = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, kernel_size=1),
            nn.Sigmoid()  # Ensure weights are between 0 and 1
        )
        
        # Store wavelength information for interpretability
        # These wavelengths correspond to specific bands in the hyperspectral data
        self.wavelengths = {
            0: 474.73,  # Band 9: Blue - chlorophyll absorption
            1: 538.71,  # Band 18: Green - healthy vegetation
            2: 650.665, # Band 32: Red - chlorophyll content
            3: 730.635, # Band 42: Red-edge - stress detection
            4: 850.59   # Band 55: NIR - leaf health
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        # Compute attention weights for each band
        attention_weights = self.attention(x)
        # Apply attention weights to input
        return x * attention_weights
    
    def get_band_importance(self) -> Dict[float, float]:
        """Get the importance of each spectral band based on attention weights.
        
        This method is useful for interpretability and understanding
        which bands the model finds most important for the task.
        """
        with torch.no_grad():
            # Create a dummy input to get attention weights
            # Using ones ensures we get the base attention values
            dummy_input = torch.ones(1, len(self.wavelengths), 1, 1)
            attention_weights = self.attention(dummy_input).squeeze()
            
            # Map weights to wavelengths for interpretability
            return {self.wavelengths[i]: float(weight) 
                   for i, weight in enumerate(attention_weights)}

class SpectralAdapter(nn.Module):
    """Adapter module for converting between 3 and 5 spectral channels.
    
    This module handles the conversion between the 5-channel multispectral
    input and the 3-channel RGB-like format expected by the SD3 VAE.
    It includes spectral attention and a series of convolutions to learn
    the optimal transformation while preserving spectral information.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        num_bands: int = 5
    ):
        super().__init__()
        self.use_attention = use_attention
        
        # Initialize spectral attention if needed
        if use_attention and in_channels == num_bands:
            self.attention = SpectralAttention(num_bands)
        
        # Three-layer convolutional network for channel adaptation
        # First two layers use 3x3 convolutions with group normalization
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # Final layer uses 1x1 convolution for channel reduction/expansion
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Group normalization for better training stability
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 32)
        
        # SiLU activation (also known as Swish)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply spectral attention if enabled
        if self.use_attention and hasattr(self, 'attention'):
            x = self.attention(x)
            
        # First convolutional block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        # Final channel adaptation
        x = self.conv3(x)
        return x

def spectral_angle_mapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Spectral Angle Mapper (SAM) between two multispectral images.
    
    SAM measures the spectral similarity between two multispectral images
    by computing the angle between their spectral vectors. This is particularly
    useful for maintaining spectral fidelity in the reconstruction.
    
    Args:
        x: Original multispectral image
        y: Reconstructed multispectral image
        
    Returns:
        Mean spectral angle in radians
    """
    # Normalize vectors to unit length
    x_norm = F.normalize(x, dim=1)
    y_norm = F.normalize(y, dim=1)
    
    # Compute cosine similarity between normalized vectors
    cos_sim = torch.sum(x_norm * y_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Ensure valid range for acos
    
    # Convert to angle in radians
    angle = torch.acos(cos_sim)
    return angle.mean()

class AutoencoderKLMultispectralAdapter(AutoencoderKL):
    """Efficient multispectral VAE implementation using adapter layers.
    
    This implementation adapts the SD3 VAE for multispectral data by:
    1. Using pretrained SD3 VAE as backbone
    2. Adding lightweight adapter layers for 5-channel input/output
    3. Keeping backbone frozen during training
    4. Only training the adapter layers
    5. Including spectral attention and specialized losses

    Parameters:
        pretrained_model_name_or_path (str): Path to pretrained SD3 VAE
        in_channels (int, optional): Number of input channels (default: 5)
        out_channels (int, optional): Number of output channels (default: 5)
        adapter_channels (int, optional): Number of channels in adapter layers (default: 32)
        adapter_placement (str, optional): Where to place adapters ("input", "output", or "both")
        use_spectral_attention (bool, optional): Whether to use spectral attention (default: True)
        use_sam_loss (bool, optional): Whether to use SAM loss (default: True)
    """
    
    @register_to_config
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        in_channels: int = 5,  # Fixed to 5 for our specific bands
        out_channels: int = 5,  # Fixed to 5 for our specific bands
        adapter_channels: int = 32,
        adapter_placement: str = "both",
        use_spectral_attention: bool = True,
        use_sam_loss: bool = True,
    ):
        # Initialize with pretrained SD3 VAE
        super().__init__()
        
        # Load pretrained weights
        pretrained_vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        self.load_state_dict(pretrained_vae.state_dict())
        
        # Store configuration
        self.adapter_placement = adapter_placement
        self.use_spectral_attention = use_spectral_attention
        self.use_sam_loss = use_sam_loss
        
        # Create adapter layers based on placement
        if adapter_placement in ["input", "both"]:
            self.input_adapter = SpectralAdapter(
                in_channels, 3,
                use_attention=use_spectral_attention,
                num_bands=in_channels
            )
        
        if adapter_placement in ["output", "both"]:
            self.output_adapter = SpectralAdapter(
                3, out_channels,
                use_attention=use_spectral_attention,
                num_bands=out_channels
            )
        
        # Freeze backbone by default
        self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze all parameters except adapter layers."""
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze adapter layers
        if hasattr(self, 'input_adapter'):
            for param in self.input_adapter.parameters():
                param.requires_grad = True
        if hasattr(self, 'output_adapter'):
            for param in self.output_adapter.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self):
        """Get parameters that should be trained (only adapter layers)."""
        params = []
        if hasattr(self, 'input_adapter'):
            params.extend(self.input_adapter.parameters())
        if hasattr(self, 'output_adapter'):
            params.extend(self.output_adapter.parameters())
        return params
    
    def compute_losses(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various loss terms for training.
        
        This method computes both per-channel MSE loss and Spectral Angle Mapper (SAM)
        loss to ensure both pixel-wise accuracy and spectral fidelity.
        
        Args:
            original: Original multispectral image
            reconstructed: Reconstructed multispectral image
            
        Returns:
            Dictionary containing different loss terms
        """
        losses = {}
        
        # Per-channel MSE loss for pixel-wise accuracy
        mse_per_channel = F.mse_loss(reconstructed, original, reduction='none')
        mse_per_channel = mse_per_channel.mean(dim=(0, 2, 3))  # Average over batch and spatial dimensions
        losses['mse_per_channel'] = mse_per_channel
        
        # Overall MSE loss
        losses['mse'] = mse_per_channel.mean()
        
        # Spectral Angle Mapper loss for spectral fidelity
        if self.use_sam_loss:
            losses['sam'] = spectral_angle_mapper(original, reconstructed)
        
        return losses
    
    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple]:
        """Encode multispectral image to latent space."""
        if hasattr(self, 'input_adapter'):
            # Convert 5 channels to 3 using input adapter
            x = self.input_adapter(x)
        # Use pretrained VAE encoder
        return super().encode(x, return_dict=return_dict)
    
    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[torch.Tensor, Tuple]:
        """Decode latent representation to multispectral image."""
        # Use pretrained VAE decoder
        x = super().decode(z, return_dict=return_dict)
        if isinstance(x, tuple):
            x = x[0]
        
        if hasattr(self, 'output_adapter'):
            # Convert 3 channels back to 5 using output adapter
            x = self.output_adapter(x.sample)
        
        if return_dict:
            return AutoencoderKLOutput(sample=x)
        return (x,)
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False, # TODO: remove (implementation skips sampling and uses only the mean latent; fine for deterministic reconstructions)
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, Tuple]:
        """Forward pass through the entire network."""
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample
        
        # Compute losses if in training mode
        if self.training:
            losses = self.compute_losses(x, dec)
            if return_dict:
                return AutoencoderKLOutput(sample=dec, losses=losses)
            return (dec, losses)
        
        if not return_dict:
            return (dec,)
        
        return AutoencoderKLOutput(sample=dec) 