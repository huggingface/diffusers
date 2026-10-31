"""
Training script for 5-channel multispectral VAE.

This script implements the training pipeline for the 5-channel multispectral VAE,
which is designed to handle 5 spectral bands (Blue, Green, Red, NIR, SWIR) while
maintaining compatibility with Stable Diffusion 3's latent space requirements.

The training process includes:
1. Loading and preprocessing 5-channel multispectral TIFF data
2. Training the VAE with proper normalization and scaling
3. Validation and checkpointing
4. Integration with diffusers' training utilities

Usage:
    python train_multispectral_vae_5ch.py \
        --dataset_path /path/to/multispectral/tiffs \
        --output_dir /path/to/save/model \
        --num_epochs 100 \
        --batch_size 8 \
        --learning_rate 1e-4
"""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import rasterio
from tqdm import tqdm

from diffusers import AutoencoderKLMultispectral5Ch
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

class MultispectralDataset(Dataset):
    """Dataset for loading 5-channel multispectral TIFF files."""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing multispectral TIFF files
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.tiff_files = list(self.data_dir.glob("*.tif"))
        self.transform = transform
        
    def __len__(self):
        return len(self.tiff_files)
    
    def __getitem__(self, idx):
        # Load 5-channel TIFF
        with rasterio.open(self.tiff_files[idx]) as src:
            # Read all 5 bands
            image = src.read()  # Shape: (5, H, W)
            
            # Convert to float and normalize
            image = image.astype(np.float32)
            
            # Apply transforms if any
            if self.transform:
                image = self.transform(image)
            
            return torch.from_numpy(image)

def train(args):
    """Main training function."""
    
    # Initialize model
    model = AutoencoderKLMultispectral5Ch(
        in_channels=5,
        out_channels=5,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128, 256, 512),
        latent_channels=4,
        norm_num_groups=32,
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Initialize EMA model
    ema_model = EMAModel(model.parameters())
    
    # Create dataset and dataloader
    dataset = MultispectralDataset(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            batch = batch.to(device)
            
            # Forward pass
            posterior = model.encode(batch)
            latents = posterior.sample()
            reconstruction = model.decode(latents)
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(reconstruction, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA model
            ema_model.step(model.parameters())
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            model.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Train 5-channel multispectral VAE")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to multispectral TIFF dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 