"""
Test script for evaluating VAE reconstruction fidelity across spectral bands.

This script tests the AutoencoderKLMultispectralAdapter's ability to:
1. Faithfully reconstruct each spectral band independently
2. Use spectral attention for band selection
3. Properly handle adapter layers
4. Compute and apply SAM loss

Key Features:
- Mean Squared Error (MSE) per band
- Root Mean Squared Error (RMSE) per band
- Spectral attention weights
- SAM loss values
- Visual comparison of original vs reconstructed bands

The test passes a single multispectral image through the VAE (encode + decode). 
High errors on non-RGB bands means spectral info is lost in latent representation. 

Usage:
    pytest test_vae_multispectral.py \
        --data-dir "/path/to/multispectral/tiffs" \
        --vae-path "/path/to/vae/model" \
        --adapter-placement "both" \
        --use-spectral-attention \
        --use-sam-loss \
        -v
"""

import os
import sys
import numpy as np
import torch
import pytest
import rasterio
from pathlib import Path
from diffusers import AutoencoderKLMultispectralAdapter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    parser.addoption("--data-dir", action="store", default=None,
                     help="Directory containing multispectral TIFF files for testing")
    parser.addoption("--vae-path", action="store", default=None,
                     help="Path to the local VAE model directory")
    parser.addoption("--adapter-placement", action="store", default="both",
                     choices=["input", "output", "both"],
                     help="Where to place adapters")
    parser.addoption("--use-spectral-attention", action="store_true",
                     help="Use spectral attention mechanism")
    parser.addoption("--use-sam-loss", action="store_true",
                     help="Use Spectral Angle Mapper loss")

@pytest.fixture
def data_dir(request):
    data_dir = request.config.getoption("--data-dir")
    if data_dir is None:
        pytest.skip("--data-dir not specified")
    if not os.path.exists(data_dir):
        pytest.skip(f"Data directory {data_dir} does not exist")
    return data_dir

@pytest.fixture
def vae_path(request):
    vae_path = request.config.getoption("--vae-path")
    if vae_path is None:
        pytest.skip("--vae-path not specified")
    if not os.path.exists(vae_path):
        pytest.skip(f"VAE model directory {vae_path} does not exist")
    return vae_path

@pytest.fixture
def adapter_config(request):
    return {
        "adapter_placement": request.config.getoption("--adapter-placement"),
        "use_spectral_attention": request.config.getoption("--use-spectral-attention"),
        "use_sam_loss": request.config.getoption("--use-sam-loss")
    }

def load_multispectral_image(image_path):
    """Load a multispectral TIFF image and return as tensor."""
    with rasterio.open(image_path) as src:
        # Read all bands
        data = src.read()
        # Convert to float32 and normalize to [-1, 1]
        data = data.astype(np.float32)
        for i in range(data.shape[0]):
            band = data[i]
            min_val = np.min(band)
            max_val = np.max(band)
            if max_val > min_val:
                data[i] = 2 * (band - min_val) / (max_val - min_val) - 1
        # Convert to tensor and add batch dimension
        data = torch.from_numpy(data).unsqueeze(0)
        return data

def compute_band_metrics(original, reconstructed):
    """Compute MSE and RMSE for each spectral band."""
    mse_per_band = torch.mean((original - reconstructed) ** 2, dim=(0, 2, 3))
    rmse_per_band = torch.sqrt(mse_per_band)
    return mse_per_band, rmse_per_band

def visualize_bands(original, reconstructed, save_path=None):
    """Create a visualization comparing original and reconstructed bands."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    band_names = ['Blue (474.73nm)', 'Green (538.71nm)', 'Red (650.665nm)', 
                 'Red-edge (730.635nm)', 'NIR (850.59nm)']
    
    for i, (ax_orig, ax_recon) in enumerate(zip(axes[0], axes[1])):
        # Original band
        ax_orig.imshow(original[0, i].cpu().numpy(), cmap='gray')
        ax_orig.set_title(f'Original {band_names[i]}')
        ax_orig.axis('off')
        
        # Reconstructed band
        ax_recon.imshow(reconstructed[0, i].cpu().numpy(), cmap='gray')
        ax_recon.set_title(f'Reconstructed {band_names[i]}')
        ax_recon.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def test_spectral_attention(vae):
    """Test the spectral attention mechanism."""
    # Get attention weights
    band_importance = vae.input_adapter.attention.get_band_importance()
    
    # Verify all bands have importance values
    expected_wavelengths = [474.73, 538.71, 650.665, 730.635, 850.59]
    assert all(wavelength in band_importance for wavelength in expected_wavelengths), \
        "Missing importance values for some bands"
    
    # Verify importance values are between 0 and 1
    assert all(0 <= importance <= 1 for importance in band_importance.values()), \
        "Importance values outside [0,1] range"
    
    # Print band importance for analysis
    print("\nSpectral Band Importance:")
    for wavelength, importance in band_importance.items():
        print(f"Band {wavelength}nm: {importance:.3f}")

def test_adapter_layers(vae, batch):
    """Test the adapter layers' functionality."""
    # Test input adapter
    if hasattr(vae, 'input_adapter'):
        # Check input adapter output shape
        input_adapter_output = vae.input_adapter(batch)
        assert input_adapter_output.shape[1] == 3, \
            "Input adapter should output 3 channels"
        
        # Check normalization
        assert torch.all(input_adapter_output >= -1) and torch.all(input_adapter_output <= 1), \
            "Input adapter output outside [-1,1] range"
    
    # Test output adapter
    if hasattr(vae, 'output_adapter'):
        # Create a dummy 3-channel input
        dummy_input = torch.randn(1, 3, 512, 512)
        
        # Check output adapter output shape
        output_adapter_output = vae.output_adapter(dummy_input)
        assert output_adapter_output.shape[1] == 5, \
            "Output adapter should output 5 channels"
        
        # Check normalization
        assert torch.all(output_adapter_output >= -1) and torch.all(output_adapter_output <= 1), \
            "Output adapter output outside [-1,1] range"

def test_sam_loss(vae, original, reconstructed):
    """Test the Spectral Angle Mapper loss computation."""
    if vae.use_sam_loss:
        # Compute SAM loss
        sam_loss = spectral_angle_mapper(original, reconstructed)
        
        # Verify SAM loss is a scalar
        assert isinstance(sam_loss, torch.Tensor)
        assert sam_loss.ndim == 0, "SAM loss should be a scalar"
        
        # Verify SAM loss is between 0 and π/2
        assert 0 <= sam_loss <= np.pi/2, "SAM loss outside valid range [0, π/2]"
        
        print(f"\nSAM Loss: {sam_loss.item():.4f}")

def test_band_selection(vae):
    """Test that the VAE uses the correct spectral bands."""
    expected_bands = [9, 18, 32, 42, 55]
    actual_bands = list(vae.input_adapter.attention.wavelengths.values())
    assert set(actual_bands) == set(expected_bands), \
        "VAE band selection doesn't match expected bands"

def test_normalization_range(vae, batch):
    """Test that input and output are properly normalized."""
    # Test input normalization
    assert torch.all(batch >= -1) and torch.all(batch <= 1), \
        "Input not normalized to [-1, 1] range"
    
    # Test output normalization
    with torch.no_grad():
        output = vae(batch)
        assert torch.all(output.sample >= -1) and torch.all(output.sample <= 1), \
            "Output not normalized to [-1, 1] range"

def test_adapter_configuration(vae):
    """Test adapter configuration and parameter freezing."""
    # Test adapter placement
    assert hasattr(vae, 'input_adapter') or hasattr(vae, 'output_adapter'), \
        "No adapters found"
    
    # Test parameter freezing
    for name, param in vae.named_parameters():
        if 'adapter' in name:
            assert param.requires_grad, f"Adapter parameter {name} should be trainable"
        else:
            assert not param.requires_grad, f"Non-adapter parameter {name} should be frozen"

def test_loss_computation(vae, batch):
    """Test loss computation and components."""
    with torch.no_grad():
        output = vae(batch)
        losses = vae.compute_losses(batch, output.sample)
        
        # Test loss components
        assert 'mse' in losses, "MSE loss missing"
        assert 'mse_per_channel' in losses, "Per-channel MSE loss missing"
        if vae.use_sam_loss:
            assert 'sam' in losses, "SAM loss missing"

def test_vae_reconstruction_fidelity(data_dir, vae_path, adapter_config):
    """Test VAE reconstruction fidelity across all spectral bands."""
    # Load VAE from local path with adapter configuration
    vae = AutoencoderKLMultispectralAdapter.from_pretrained(
        vae_path,
        adapter_placement=adapter_config["adapter_placement"],
        use_spectral_attention=adapter_config["use_spectral_attention"],
        use_sam_loss=adapter_config["use_sam_loss"]
    )
    vae.eval()
    
    # Get first TIFF file from directory
    image_path = next(Path(data_dir).glob('*.tif'))
    logger.info(f"Testing with image: {image_path}")
    
    # Load and preprocess image
    original = load_multispectral_image(image_path)
    
    # Run all tests
    test_spectral_attention(vae)
    test_adapter_layers(vae, original)
    test_band_selection(vae)
    test_normalization_range(vae, original)
    test_adapter_configuration(vae)
    test_loss_computation(vae, original)
    
    # Encode and decode
    with torch.no_grad():
        latent = vae.encode(original).latent_dist.sample()
        reconstructed = vae.decode(latent).sample
    
    # Compute metrics
    mse_per_band, rmse_per_band = compute_band_metrics(original, reconstructed)
    
    # Test SAM loss
    test_sam_loss(vae, original, reconstructed)
    
    # Log results
    band_names = ['Blue (474.73nm)', 'Green (538.71nm)', 'Red (650.665nm)', 
                 'Red-edge (730.635nm)', 'NIR (850.59nm)']
    logger.info("\nReconstruction Metrics:")
    for i, (mse, rmse) in enumerate(zip(mse_per_band, rmse_per_band)):
        logger.info(f"{band_names[i]} Band:")
        logger.info(f"  MSE: {mse.item():.6f}")
        logger.info(f"  RMSE: {rmse.item():.6f}")
    
    # Visualize results
    save_path = "vae_reconstruction_comparison.png"
    visualize_bands(original, reconstructed, save_path)
    logger.info(f"Visualization saved to {save_path}")
    
    # Assertions
    # 1. Check that all bands are reconstructed with reasonable fidelity
    assert torch.all(mse_per_band < 0.1), "MSE too high for some bands"
    
    # 2. Check that non-RGB bands are reconstructed with similar fidelity to RGB
    rgb_mse = torch.mean(mse_per_band[:3])
    nir_rededge_mse = torch.mean(mse_per_band[3:])
    assert abs(rgb_mse - nir_rededge_mse) < 0.05, \
        "Significant difference in reconstruction fidelity between RGB and non-RGB bands"
    
    # 3. Check that the overall structure is preserved
    assert torch.allclose(original.mean(), reconstructed.mean(), rtol=0.1), \
        "Significant difference in overall image statistics"

if __name__ == "__main__":
    # Remove the script name from sys.argv
    sys.argv.pop(0)
    
    # Run pytest with the remaining arguments
    pytest.main(sys.argv) 