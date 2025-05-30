# Stable Diffusion 3 Multispectral Training

This repository contains the implementation of DreamBooth training for Stable Diffusion 3 with multispectral image support. The implementation is specifically designed to work with 5-channel multispectral data.

updated 9.10.2025

## Overview

This project extends the Hugging Face Diffusers library to support training Stable Diffusion 3 on multispectral imagery. Key features include:

- Custom VAE implementation for 5-channel multispectral data
- Specialized dataloader for multispectral TIFF files
- Integration with Stable Diffusion 3's architecture
- Memory-efficient training pipeline

## Repository Structure

```
diffusers/
├── src/
│   └── diffusers/
│       ├── __init__.py
│       └── models/
│           ├── __init__.py
│           └── autoencoders/
│               └── autoencoder_kl_multispectral_5ch.py
├── examples/
│   └── dreambooth/
│       └── train_dreambooth_sd3_multispectral.py
├── multispectral_dataloader.py
├── setup.py
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusers
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install additional requirements:
```bash
pip install -r requirements.txt
pip install rasterio  # For multispectral data handling
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
PYTHONPATH=$PYTHONPATH:. accelerate launch examples/dreambooth/train_dreambooth_sd3_multispectral.py \
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

## Key Components

### 1. Multispectral VAE
The `AutoencoderKLMultispectral5Ch` class extends the standard VAE to handle 5-channel multispectral data while maintaining compatibility with Stable Diffusion 3's latent space requirements.

### 2. Multispectral DataLoader
The custom dataloader (`multispectral_dataloader.py`) provides efficient loading and preprocessing of multispectral TIFF files, including:
- Per-channel normalization
- Automatic padding and resizing
- Memory-efficient processing
- GPU optimization

### 3. Training Script
The training script (`train_dreambooth_sd3_multispectral.py`) implements the DreamBooth training procedure with adaptations for multispectral data.

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
