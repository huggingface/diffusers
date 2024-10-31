import os
import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from diffusers import UNet1DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm


def train_diffusion():
    """Train diffusion model using HuggingFace diffusers"""
    # Configs
    data_config = DataConfig()
    model_config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    print(f"Using device: {device}")
    
    # Create dataset (from zarr file)
    dataset = PushTStateDataset(
        dataset_path=data_config.dataset_path,
        pred_horizon=data_config.pred_horizon,
        obs_horizon=data_config.obs_horizon,
        action_horizon=data_config.action_horizon
    )
    
    # Assign stats and define save directory
    stats = dataset.stats
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create observation encoder
    obs_encoder = ObservationEncoder(
        obs_dim=data_config.state_dim,
        embed_dim=model_config.obs_embed_dim
    ).to(device)
    
    # Create UNet1D model from diffusers
    model = UNet1DModel(
        sample_size=model_config.sample_size,
        in_channels=model_config.total_in_channels,  # actions + conditioning
        out_channels=model_config.out_channels,
        layers_per_block=model_config.layers_per_block,
        block_out_channels=model_config.block_out_channels,
        norm_num_groups=model_config.norm_num_groups,
        down_block_types=model_config.down_block_types,
        up_block_types=model_config.up_block_types,
    ).to(device)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon"
    )
    
    # Create projection layer OUTSIDE the training loop
    obs_projection = nn.Linear(model_config.obs_embed_dim * data_config.obs_horizon, 
                             model_config.obs_embed_dim // 8).to(device)
    
    # Update optimizer to include projection layer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': obs_encoder.parameters()},
        {'params': obs_projection.parameters()}
    ], lr=1e-4)
    
    # Update EMA to include projection layer
    ema = EMAModel(
        parameters=list(model.parameters()) + 
                  list(obs_encoder.parameters()) + 
                  list(obs_projection.parameters()),
        power=0.75
    )
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}')
        epoch_loss = []
        
        for batch in dataloader:
            # Get batch data
            obs = batch['obs'].to(device)  # [batch, obs_horizon, obs_dim]
            actions = batch['action'].to(device)  # [batch, pred_horizon, action_dim]
            batch_size = obs.shape[0]
            
            # Encode observations for conditioning
            obs_embedding = obs_encoder(obs)  # [batch, obs_embed_dim * obs_horizon]
            
            # Sample noise and timesteps
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
            
            # Add noise to actions according to noise schedule
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
            
            # Reshape to channels format for UNet
            # [batch, pred_horizon, channels] -> [batch, channels, pred_horizon]
            noisy_actions = noisy_actions.transpose(1, 2)
            noise = noise.transpose(1, 2)
            
            # Project the observation embedding
            obs_cond = obs_projection(obs_embedding)  # [batch, obs_embed_dim//8]
            
            # Reshape to match sequence length
            obs_cond = obs_cond.unsqueeze(-1).expand(-1, -1, noisy_actions.shape[-1])
            
            # Concatenate along channel dimension
            model_input = torch.cat([noisy_actions, obs_cond], dim=1)
            
            noise_pred = model(
                model_input,
                timesteps,
            ).sample  # Removed slicing [:, :data_config.action_dim]
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            epoch_loss.append(loss.item())
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA parameters
            ema.step(list(model.parameters()) + 
                     list(obs_encoder.parameters()) + 
                     list(obs_projection.parameters()))
            
            # Update progress
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
        
        progress_bar.close()
        
        # Print epoch stats
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"\nEpoch {epoch} average loss: {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': obs_encoder.state_dict(),
                'projection_state_dict': obs_projection.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'noise_scheduler_state_dict': noise_scheduler.state_dict(),  # Removed
                'stats': stats,
                'loss': avg_loss,
            }, os.path.join(save_dir, f'diffusion_checkpoint_{epoch}.pt'))
    
    return model, obs_encoder, obs_projection, ema, noise_scheduler, optimizer, stats

def main():
    # Download dataset if needed
    config = DataConfig()
    if not os.path.isfile(config.dataset_path):
        print("Downloading dataset...")
        gdown.download(id=config.dataset_gdrive_id, output=config.dataset_path, quiet=False)
    
    # Create dataset
    dataset = PushTStateDataset(
        dataset_path=config.dataset_path,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon
    )
    
    # Test batch
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256, num_workers=1,
        shuffle=True, pin_memory=True, persistent_workers=True
    )
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape:", batch['action'].shape)

if __name__ == "__main__":
    main()
    
    print("\nStarting diffusion model training...")
    model, obs_encoder, obs_projection, ema, noise_scheduler, optimizer, stats = train_diffusion()
    
    # Save final model
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': obs_encoder.state_dict(),
        'projection_state_dict': obs_projection.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'noise_scheduler_state_dict': noise_scheduler.state_dict(), 
        'stats': stats
    }, os.path.join(save_dir, 'diffusion_final.pt'))
