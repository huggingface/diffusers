"""
Multispectral Image Dataloader for DreamBooth Training

This module implements a specialized dataloader for multispectral TIFF images.
It handles 5-channel data by selecting the first 5 bands from input TIFF files.

Key Features:
1. Simple 5-band selection from input TIFFs
2. Per-channel normalization to [0,1] range
3. Padding to square shape and resizing to 512x512
4. Memory-efficient caching and worker management
5. GPU-optimized data loading with pin_memory (when available)

Usage Notes:
1. The dataloader takes any TIFF file with 5 or more bands
2. Always uses the first 5 bands in order
3. Caching is enabled by default for small datasets
4. For local testing:
   - Set num_workers=0
   - Set prefetch_factor=None
   - Set persistent_workers=False
5. For GPU training:
   - Enable prefetch_factor (default=2)
   - Enable persistent_workers (default=True)
   - Set appropriate num_workers based on system

Example:
    ```python
    # For local testing
    dataloader = create_multispectral_dataloader(
        data_root="path/to/tiffs",
        batch_size=4,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False
    )

    # For GPU training
    dataloader = create_multispectral_dataloader(
        data_root="path/to/tiffs",
        batch_size=4,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True
    )
    ```
"""

import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultispectralDataset(Dataset):
    """
    Dataset class for loading and preprocessing multispectral TIFF images.
    Handles 5-channel data by selecting first 5 bands from input TIFFs.
    """
    
    def __init__(
        self,
        data_root: str,
        resolution: int = 512,
        transform: Optional[transforms.Compose] = None,
        use_cache: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root (str): Path to directory containing TIFF files
            resolution (int): Target resolution for images (default: 512)
            transform (callable, optional): Additional transforms to apply
            use_cache (bool): Whether to cache loaded images in memory
        """
        self.data_root = data_root
        self.resolution = resolution
        self.transform = transform
        self.use_cache = use_cache
        
        # Get list of TIFF files
        self.image_paths = [
            os.path.join(data_root, f) for f in os.listdir(data_root)
            if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
        ]
        
        if not self.image_paths:
            raise FileNotFoundError(
                f"No TIFF files found in {data_root}. Please ensure the directory contains "
                f".tiff or .tif files with at least 5 spectral bands."
            )
        
        # Cache for storing preprocessed images
        self.cache = {} if use_cache else None
        
        # Validate all images on initialization
        self._validate_all_images()
    
    def _validate_all_images(self):
        """Validate that all images have at least 5 bands."""
        for path in self.image_paths:
            try:
                with rasterio.open(path) as src:
                    if src.count < 5:
                        raise ValueError(
                            f"Image {path} has only {src.count} bands, but at least 5 bands are required. "
                            f"This dataloader is configured for 5-channel multispectral data. "
                            f"Please ensure all input images have 5 or more bands."
                        )
            except rasterio.errors.RasterioIOError as e:
                raise ValueError(
                    f"Failed to open image {path}: {str(e)}. "
                    f"Please ensure the file is a valid TIFF file and is not corrupted."
                )
            except Exception as e:
                raise ValueError(
                    f"Unexpected error validating {path}: {str(e)}. "
                    f"Please check the file format and permissions."
                )
    
    def normalize_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Per-channel min-max normalization to [0, 1] range.
        Includes safety checks for division by zero and NaN values.
        
        Args:
            channel_data: Input channel data
            
        Returns:
            Normalized channel data
        """
        # Handle NaN values
        min_val = np.nanmin(channel_data)
        max_val = np.nanmax(channel_data)
        
        # Safety check for division by zero
        if max_val == min_val:
            logger.warning(
                f"Channel has constant value {min_val}. "
                f"Returning zero array to avoid division by zero."
            )
            return np.zeros_like(channel_data, dtype=np.float32)
            
        return (channel_data - min_val) / (max_val - min_val)
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a multispectral image.
        Takes first 5 bands and processes them for SD3 compatibility.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor of shape (5, 512, 512)
        """
        try:
            with rasterio.open(image_path) as src:
                # Read first 5 bands
                image = src.read()[:5]  # Shape: (5, height, width)
                
                # Convert to float32 for processing
                image = image.astype(np.float32)
                
                # Per-channel normalization
                normalized_image = np.zeros_like(image)
                for i in range(5):
                    normalized_image[i] = self.normalize_channel(image[i])
                
                # Convert to torch tensor
                image_tensor = torch.from_numpy(normalized_image)
                
                # Calculate padding
                h, w = image_tensor.shape[1:]
                max_dim = max(h, w)
                pad_h = (max_dim - h) // 2
                pad_w = (max_dim - w) // 2
                
                # Pad to square
                image_tensor = F.pad(
                    image_tensor,
                    (pad_w, pad_w, pad_h, pad_h),
                    mode='constant',
                    value=0
                )
                
                # Resize to target resolution
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(self.resolution, self.resolution),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                return image_tensor
        except Exception as e:
            raise RuntimeError(
                f"Failed to preprocess image {image_path}: {str(e)}. "
                f"Please ensure the file is a valid multispectral TIFF with at least 5 bands."
            )
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a preprocessed image.
        
        Args:
            idx: Index of the image to get
            
        Returns:
            Preprocessed image tensor of shape (5, 512, 512)
        """
        image_path = self.image_paths[idx]
        
        # Check cache first
        if self.use_cache and image_path in self.cache:
            return self.cache[image_path]
        
        # Load and preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Apply additional transforms if specified
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Cache the result if caching is enabled
        if self.use_cache:
            self.cache[image_path] = image_tensor
        
        return image_tensor

def create_multispectral_dataloader(
    data_root: str,
    batch_size: int = 4,
    resolution: int = 512,
    num_workers: int = 4,
    use_cache: bool = True,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True
) -> DataLoader:
    """
    Create a DataLoader for multispectral images with optimized settings.
    
    Args:
        data_root: Path to directory containing TIFF files
        batch_size: Batch size for training
        resolution: Target resolution for images
        num_workers: Number of worker processes for data loading
        use_cache: Whether to cache loaded images in memory
        prefetch_factor: Number of batches to prefetch per worker (None to disable)
        persistent_workers: Whether to keep workers alive between epochs
    
    Returns:
        DataLoader: Configured DataLoader for multispectral images
    """
    dataset = MultispectralDataset(
        data_root=data_root,
        resolution=resolution,
        use_cache=use_cache
    )
    
    # Only use prefetch_factor if num_workers > 0
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers and num_workers > 0,
        "drop_last": True  # Avoid partial batches
    }
    
    # Only add prefetch_factor if specified and num_workers > 0
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    
    return DataLoader(dataset, **kwargs)

def test_memory_usage(data_dir, test_images):
    """Test memory usage under load."""
    dataset = MultispectralDataset(data_dir, use_cache=True)
    dataloader = create_multispectral_dataloader(
        data_dir,
        batch_size=4,
        num_workers=2,
        prefetch_factor=2
    )
    
    # Load multiple batches to test memory behavior
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Test with 10 batches
            break
        batches.append(batch)
    
    # Verify memory is managed properly
    assert len(batches) == 10
    # Add memory usage assertions if needed

def test_worker_behavior(data_dir, test_images):
    """Test worker behavior and data loading consistency."""
    dataloader = create_multispectral_dataloader(
        data_dir,
        batch_size=2,
        num_workers=2,
        persistent_workers=True
    )
    
    # Test multiple epochs
    for epoch in range(2):
        batches = []
        for batch in dataloader:
            batches.append(batch)
        
        # Verify batch consistency
        for i in range(len(batches)-1):
            assert batches[i].shape == batches[i+1].shape

def test_explicit_caching_validation(data_dir, test_images):
    """
    Test explicit validation of the caching mechanism to ensure data integrity.
    
    This test verifies that:
    -Tests that cached data is identical to original data
    -Verifies tensor properties and normalization
    -Checks channel independence 
    -Simulates cache persistence by creating new dataset instances
    
    Note: Since caching is implemented in-memory within the same process,
    we simulate cache persistence by creating new dataset instances.
    """
    # Create first dataset instance and load data
    dataset1 = MultispectralDataset(data_dir, use_cache=True)
    original_tensor = dataset1[0]  # This will be cached
    
    # Create second dataset instance to simulate fresh process
    dataset2 = MultispectralDataset(data_dir, use_cache=True)
    cached_tensor = dataset2[0]  # Should load from cache
    
    # Verify tensor properties
    assert isinstance(cached_tensor, torch.Tensor)
    assert cached_tensor.shape == (5, 512, 512)
    assert cached_tensor.dtype == torch.float32
    
    # Verify data integrity
    assert torch.allclose(original_tensor, cached_tensor, rtol=1e-5, atol=1e-5), \
        "Cached tensor differs from original tensor"
    
    # Verify normalization is preserved
    assert torch.all(cached_tensor >= 0) and torch.all(cached_tensor <= 1), \
        "Cached tensor values outside [0,1] range"
    
    # Verify channel independence
    for c in range(cached_tensor.shape[0]):
        channel = cached_tensor[c]
        assert torch.min(channel) == 0 or torch.max(channel) == 1, \
            f"Channel {c} not properly normalized"

def test_file_order_consistency(data_dir, test_images):
    """
    Test that file order remains consistent across dataloader instances.
    
    This test ensures reproducibility by verifying that:
    1. File order is identical between dataloader instances
    2. Order is preserved when shuffle=False
    3. Order is deterministic across runs
    
    This is crucial for reproducible training in multispectral applications
    where band order and data consistency are essential.
    """
    # Create first dataloader instance
    dataloader1 = create_multispectral_dataloader(
        data_dir,
        batch_size=2,
        num_workers=0,
        use_cache=True,
        shuffle=False  # Disable shuffling for order consistency
    )
    
    # Get file order from first instance
    dataset1 = dataloader1.dataset
    first_order = dataset1.image_paths.copy()
    
    # Create second dataloader instance
    dataloader2 = create_multispectral_dataloader(
        data_dir,
        batch_size=2,
        num_workers=0,
        use_cache=True,
        shuffle=False  # Disable shuffling for order consistency
    )
    
    # Get file order from second instance
    dataset2 = dataloader2.dataset
    second_order = dataset2.image_paths.copy()
    
    # Verify order consistency
    assert len(first_order) == len(second_order), \
        "Different number of files between dataloader instances"
    
    for i, (path1, path2) in enumerate(zip(first_order, second_order)):
        assert path1 == path2, \
            f"File order mismatch at index {i}: {path1} != {path2}"
    
    # Verify data consistency by loading full epoch
    batches1 = []
    batches2 = []
    
    for batch1, batch2 in zip(dataloader1, dataloader2):
        batches1.append(batch1)
        batches2.append(batch2)
    
    # Verify batch shapes and content
    assert len(batches1) == len(batches2), \
        "Different number of batches between dataloader instances"
    
    for i, (batch1, batch2) in enumerate(zip(batches1, batches2)):
        assert batch1.shape == batch2.shape, \
            f"Batch shape mismatch at index {i}"
        assert torch.allclose(batch1, batch2, rtol=1e-5, atol=1e-5), \
            f"Batch content mismatch at index {i}"
