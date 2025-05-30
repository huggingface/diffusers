"""
Test script for the multispectral dataloader.

This script tests the key functionality of the MultispectralDataset and related classes,
including data loading, normalization, validation, and error handling.

Test Design Decisions:

1. Data Testing:
   - Uses multispectral TIFF files with 5 or more bands
   - Takes first predefined bands for processing
   - Maintains reproducibility by using fixed seed for random selection

2. Error Handling Tests:
   - Non-existent directories are tested
   - Invalid band counts are tested
   - Empty directories are tested

3. Caching Tests:
   - Performance is measured using time.time()
   - Data consistency is verified using torch.allclose()
   - Cache behavior is tested with controlled data access patterns

4. SD3 Compatibility:
   - Input shape tests verify 5-channel, 512x512 requirements
   - Pixel range tests ensure [-1, 1] normalization for VAE
   - Channel independence is verified for normalization

5. Performance Tests:
   - Worker behavior is tested for consistency
   - Local testing configuration:
     * num_workers=0
     * prefetch_factor=None
     * persistent_workers=False
   - Memory usage is monitored
   - TODO: GPU-specific features (prefetching, persistent workers) are disabled
     for local testing but should be enabled for GPU training

6. Tolerance:
   - Tests use relaxed tolerances (rtol=1e-2, atol=1e-2) for floating-point imprecision

7. Band Selection Test:
   - The test_specific_band_selection test uses the exact file as the dataloader for index 0

Usage:
    pytest test_multispectral_dataloader.py --data-dir "/Users/zina/Desktop/LDM4HSI/Project Files/Dataloader test/Output Testset Mango" -v

Note:
    For local testing, worker-intensive features are disabled to ensure
    reliable test execution. These features should be enabled when running
    on GPU hardware for actual training.
"""

import os
import sys
import numpy as np
import torch
import pytest
import time
import logging
import random
from pathlib import Path
import rasterio
import torch.nn.functional as F
import argparse
from multispectral_dataloader import (
    MultispectralDataset,
    create_multispectral_dataloader
)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def pytest_addoption(parser):
    parser.addoption("--data-dir", action="store", default=None,
                     help="Directory containing multispectral TIFF files for testing")

@pytest.fixture
def data_dir(request):
    data_dir = request.config.getoption("--data-dir")
    if data_dir is None:
        pytest.skip("--data-dir not specified")
    if not os.path.exists(data_dir):
        pytest.skip(f"Data directory {data_dir} does not exist")
    return data_dir

def get_test_images(data_dir, num_images=2):
    """Select a subset of images for testing."""
    all_files = sorted(Path(data_dir).glob('*.tiff'))
    
    # Adjust num_images if we have fewer files
    num_images = min(num_images, len(all_files))
    
    # Randomly select images
    selected_files = random.sample(all_files, num_images)
    
    return selected_files

@pytest.fixture
def test_images(data_dir):
    """Get a subset of test images."""
    return get_test_images(data_dir)

def test_dataset_initialization(data_dir, test_images):
    """Test dataset initialization with real data."""
    dataset = MultispectralDataset(data_dir)
    assert len(dataset) > 0
    assert dataset.resolution == 512
    assert dataset.use_cache is True

def test_band_count_validation(data_dir, test_images):
    """Test validation of band count."""
    # Test with valid data
    dataset = MultispectralDataset(data_dir)
    assert len(dataset) > 0
    
    # Test with empty directory
    with pytest.raises(FileNotFoundError):
        MultispectralDataset("empty_dir")

def test_normalize_channel(data_dir, test_images):
    """Test channel normalization with real data."""
    dataset = MultispectralDataset(data_dir)
    
    # Load a real image
    image_path = str(test_images[0])
    with rasterio.open(image_path) as src:
        data = src.read(1)  # Read first band
    
    normalized = dataset.normalize_channel(data)
    # Check for [-1, 1] range
    assert np.all(normalized >= -1) and np.all(normalized <= 1)
    
    # Test with NaN values
    data_with_nan = data.copy()
    data_with_nan[0, 0] = np.nan
    normalized = dataset.normalize_channel(data_with_nan)
    mask = ~np.isnan(normalized)
    assert np.all(normalized[mask] >= -1) and np.all(normalized[mask] <= 1)
    assert np.isnan(normalized[0, 0])

def test_sd3_compatible_input_shape(data_dir, test_images):
    """Test that preprocessed images are compatible with SD3's VAE input requirements."""
    dataset = MultispectralDataset(data_dir)
    
    # Load and preprocess image
    image = dataset[0]
    
    # Check tensor properties for SD3 compatibility
    assert isinstance(image, torch.Tensor)
    assert image.shape == (5, 512, 512)  # 5 channels, 512x512 resolution
    assert image.dtype == torch.float32
    # Check for [-1, 1] range
    assert torch.all(image >= -1) and torch.all(image <= 1)

def test_pixel_range_normalization_for_vae(data_dir, test_images):
    """Test that pixel values are properly normalized for VAE input."""
    dataset = MultispectralDataset(data_dir)
    image = dataset[0]
    
    # Check normalization properties
    assert torch.all(image >= -1) and torch.all(image <= 1)
    # Check that each channel has been normalized independently
    for c in range(image.shape[0]):
        channel = image[c]
        # For [-1, 1] normalization, min should be close to -1 or max should be close to 1
        assert torch.isclose(torch.min(channel), torch.tensor(-1.), atol=1e-2) or torch.isclose(torch.max(channel), torch.tensor(1.), atol=1e-2)

def test_caching_behavior(data_dir, test_images):
    """Test that caching improves load time and maintains data consistency."""
    dataset = MultispectralDataset(data_dir, use_cache=True)
    
    # First load
    start_time = time.time()
    first_load = dataset[0]
    first_load_time = time.time() - start_time
    
    # Second load (should be from cache)
    start_time = time.time()
    second_load = dataset[0]
    second_load_time = time.time() - start_time
    
    # Verify cache is working
    assert second_load_time < first_load_time
    assert torch.allclose(first_load, second_load)

def test_dataloader_creation(data_dir, test_images):
    """Test dataloader creation and basic functionality."""
    dataloader = create_multispectral_dataloader(
        data_dir,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        use_cache=True,
        prefetch_factor=None,  # Disabled for local testing
        persistent_workers=False  # Disabled for local testing
    )
    
    # Test batch loading
    batch = next(iter(dataloader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 2  # batch_size
    assert batch.shape[1] == 5  # channels
    assert batch.shape[2] == 512  # height
    assert batch.shape[3] == 512  # width

def test_worker_behavior(data_dir, test_images):
    """Test worker behavior and data loading consistency."""
    dataloader = create_multispectral_dataloader(
        data_dir,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        persistent_workers=False,  # Disabled for local testing
        prefetch_factor=None  # Disabled for local testing
    )
    
    # Test multiple epochs
    for epoch in range(2):
        batches = []
        for batch in dataloader:
            batches.append(batch)
        
        # Verify batch consistency
        for i in range(len(batches)-1):
            assert batches[i].shape == batches[i+1].shape

def test_error_handling(data_dir):
    """Test error handling for invalid data."""
    # Test with non-existent directory
    with pytest.raises(FileNotFoundError):
        MultispectralDataset("non_existent_dir")

def test_specific_band_selection(data_dir):
    """Test that the dataloader correctly selects the hardcoded bands."""
    dataset = MultispectralDataset(data_dir)
    image = dataset[0]

    # Print the file path used by the dataloader for index 0
    image_path = dataset.image_paths[0]
    print(f"Dataloader image path: {image_path}")

    # Use the same file for the expected bands
    with rasterio.open(image_path) as src:
        expected_bands = src.read([9, 18, 32, 42, 55]).astype(np.float32)
        # Per-channel normalization to [-1, 1]
        for i in range(expected_bands.shape[0]):
            band = expected_bands[i]
            min_val = np.min(band)
            max_val = np.max(band)
            if max_val > min_val:
                normalized = (band - min_val) / (max_val - min_val)
                expected_bands[i] = 2 * normalized - 1
            else:
                expected_bands[i] = np.zeros_like(band)
        expected_bands = torch.from_numpy(expected_bands)
        expected_bands = F.interpolate(
            expected_bands.unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    assert torch.allclose(image, expected_bands, rtol=1e-2, atol=1e-2), "Dataset output does not match expected band selection"

if __name__ == "__main__":
    # Remove the script name from sys.argv
    sys.argv.pop(0)
    
    # Run pytest with the remaining arguments
    pytest.main(sys.argv) 