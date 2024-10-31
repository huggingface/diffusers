import os
import gdown
import zarr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from diffusers import UNet1DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional


# data configurations
@dataclass
class DataConfig:
    """Configuration for dataset"""
    dataset_path: str = "pusht_cchi_v7_replay.zarr.zip"
    dataset_gdrive_id: str = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    
    pred_horizon: int = 16  # number of future steps to predict
    obs_horizon: int = 2    # number of past observations used to condition the predictions
    action_horizon: int = 8 # number of actions to execute
    
    # data dimensions
    image_size: Tuple[int, int] = (96, 96)
    image_channels: int = 3
    action_dim: int = 2 # [velocity in x direction, velocity in y direction]
    state_dim: int = 5  # [agent_x, agent_y, block_x, block_y, block_angle]

@dataclass
class ModelConfig:
    """Configuration for neural networks"""
    # Observation encoding
    obs_embed_dim: int = 256
    
    # UNet configuration
    sample_size: int = 16  # pred_horizon length
    in_channels: int = 2   # action dimension
    out_channels: int = 2  # action dimension
    layers_per_block: int = 2
    block_out_channels: Tuple[int, ...] = (128,)
    norm_num_groups: int = 8
    down_block_types: Tuple[str, ...] = ("DownBlock1D",) * 1
    up_block_types: Tuple[str, ...] = ("UpBlock1D",) * 1
    
    def __post_init__(self):
        # For conditioning through input channels
        self.total_in_channels = self.in_channels + self.obs_embed_dim //8 # actions + conditioning


"""
Helper Functions
"""
def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # Normalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # Normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# Dataset Class
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        # Read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        # All demonstration episodes are concatenated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # Compute start and end of each state-action sequence
        # Also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # Add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # Compute statistics and normalize data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # Get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # Discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample

# Model Classes
class ObservationEncoder(nn.Module):
    """Encodes observations for conditioning"""
    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x):
        # x: [batch, timesteps, obs_dim]
        batch_size, timesteps, obs_dim = x.shape
        x = x.reshape(-1, obs_dim)
        x = self.net(x)
        x = x.reshape(batch_size, timesteps * self.net[-1].out_features)
        return x

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
