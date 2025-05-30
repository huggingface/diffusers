# Multispectral DreamBooth Training for Stable Diffusion 3

This directory contains the implementation of DreamBooth training for Stable Diffusion 3 with multispectral image support. The implementation is specifically designed to work with 5-channel multispectral data.

updated 9.10.2025

## Overview

The training script (`train_dreambooth_sd3_multispectral.py`) and associated dataloader (`multispectral_dataloader.py`) enable fine-tuning of Stable Diffusion 3 on multispectral imagery. Key features include:

- Support for 5-channel multispectral TIFF images
- Custom VAE implementation for multispectral data
- Optimized data loading and preprocessing
- Integration with Stable Diffusion 3's architecture

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Rasterio for multispectral data handling

## Installation

1. Install the base requirements:
```bash
pip install -r requirements.txt
```

2. Install the diffusers package in development mode:
```bash
pip install -e .
```

3. Install additional dependencies:
```bash
pip install rasterio
```

## Data Preparation

Your multispectral data should be organized as follows:
```
/path/to/data/
└── Output Testset Mango/
    └── *.tif  # 5-channel multispectral TIFF files
```

Each TIFF file should contain at least 5 bands of spectral data in the following order:
1. Red
2. Green
3. Blue
4. Near Infrared
5. Short Wave Infrared

## Training

To start training, run:

```bash
PYTHONPATH=$PYTHONPATH:. accelerate launch train_dreambooth_sd3_multispectral.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --instance_data_dir="/path/to/your/data" \
    --output_dir="sd3-dreambooth-multispectral" \
    --instance_prompt="sks leaf with no background" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-6 \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --validation_prompt="sks leaf with no background" \
    --validation_epochs=25 \
    --seed=0
```

## Key Parameters

- `--pretrained_model_name_or_path`: Path to the base SD3 model
- `--instance_data_dir`: Directory containing your multispectral TIFF files
- `--resolution`: Input image resolution (default: 1024)
- `--train_batch_size`: Batch size for training
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients
- `--learning_rate`: Learning rate for training
- `--max_train_steps`: Total number of training steps

## Troubleshooting

1. **ImportError for AutoencoderKLMultispectral5Ch**
   - Ensure the package is installed in development mode
   - Verify the class is properly imported in `__init__.py` files

2. **ModuleNotFoundError for multispectral_dataloader**
   - Set PYTHONPATH to include the current directory
   - Verify the dataloader file is in the correct location

3. **Data Loading Issues**
   - Check TIFF file format and band count
   - Verify file permissions and paths
   - Ensure sufficient disk space and memory
