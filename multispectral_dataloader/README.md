# Multispectral DataLoader

This module provides a specialized data loader for handling 5-channel multispectral TIFF images in the context of Stable Diffusion 3 training.

## Features

- Support for 5-channel multispectral TIFF images
- Per-channel normalization
- Efficient data loading with GPU optimization
- Automatic padding and resizing
- Memory-efficient processing

## Installation

The dataloader is included with the main package installation. No additional installation steps are required beyond the main package setup.

## Usage

```python
from multispectral_dataloader import MultispectralDataset

# Initialize the dataset
dataset = MultispectralDataset(
    data_dir="/path/to/your/tiff/files",
    resolution=1024,
    normalize=True
)

# Create a DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Data Format Requirements

1. Input files must be TIFF format
2. Each TIFF file must contain at least 5 bands
3. Bands should be in the following order:
   - Band 1: Red
   - Band 2: Green
   - Band 3: Blue
   - Band 4: Near Infrared
   - Band 5: Short Wave Infrared

## Configuration

The dataloader supports several configuration options:

- `resolution`: Output image size (default: 1024)
- `normalize`: Whether to normalize the data (default: True)
- `num_workers`: Number of worker processes (default: 4)
- `pin_memory`: Whether to pin memory for faster GPU transfer (default: True)

## Memory Management

The dataloader implements several memory optimization techniques:

1. Lazy loading of TIFF files
2. Efficient memory mapping
3. Automatic cleanup of unused resources
4. GPU memory optimization

## Testing

Run the test suite with:

```bash
python -m pytest test_multispectral_dataloader.py
```

## Troubleshooting

1. **Memory Issues**
   - Reduce batch size
   - Decrease number of workers
   - Enable memory mapping

2. **Loading Errors**
   - Verify TIFF file format
   - Check band count
   - Ensure file permissions

3. **Performance Issues**
   - Increase number of workers
   - Enable pin_memory
   - Use memory mapping
