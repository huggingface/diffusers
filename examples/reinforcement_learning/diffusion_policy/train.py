import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from config import DataConfig, ModelConfig
from dataset import SequentialDataset
from model import create_model

def train_diffusion(
    data_config: DataConfig,
    model_config: ModelConfig,
    num_epochs: int = 100,              # training duration
    batch_size: int = 256,              # samples per batch
    learning_rate: float = 1e-4,        
    save_dir: str = "checkpoints",      # save directory
    device: torch.device = None         
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # dataset and dataloader
    dataset = SequentialDataset(data_config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # model components
    model, obs_encoder, obs_projection, noise_scheduler = create_model(
        data_config, model_config, device
    )
    
    # optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': obs_encoder.parameters()},
        {'params': obs_projection.parameters()}
    ], lr=learning_rate)
    
    # EMA
    ema = EMAModel(
        parameters=list(model.parameters()) + 
                  list(obs_encoder.parameters()) + 
                  list(obs_projection.parameters()),
        power=0.75
    )
    
    # save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}')
        epoch_loss = []
        
        for batch in dataloader:
            # get batch data
            state = batch['state'].to(device)       # [batch, obs_horizon, state_dim]
            actions = batch['action'].to(device)    # [batch, pred_horizon, action_dim]
            batch_size = state.shape[0]
            
            # encode observations for conditioning
            obs_embedding = obs_encoder(state)      # [batch, obs_embed_dim * obs_horizon]
            
            # sample noise and timesteps
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
            
            # add noise to actions according to noise schedule
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
            
            # reshape to channels format for UNet
            noisy_actions = noisy_actions.transpose(1, 2)
            noise = noise.transpose(1, 2)
            
            # project the observation embedding
            obs_cond = obs_projection(obs_embedding)     # [batch, obs_embed_dim//8]
            
            # reshape to match sequence length
            obs_cond = obs_cond.unsqueeze(-1).expand(-1, -1, noisy_actions.shape[-1])
            
            # concatenate along channel dimension
            model_input = torch.cat([noisy_actions, obs_cond], dim=1)
            
            # predict noise
            noise_pred = model(model_input, timesteps).sample
            
            # calculate loss
            loss = F.mse_loss(noise_pred, noise)
            epoch_loss.append(loss.item())
            
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update EMA parameters
            ema.step(list(model.parameters()) + 
                     list(obs_encoder.parameters()) + 
                     list(obs_projection.parameters()))
            
            # update progress
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
        
        progress_bar.close()
        
        # epoch stats
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"\nEpoch {epoch} average loss: {avg_loss:.6f}")
        
        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                save_dir=save_dir,
                epoch=epoch,
                model=model,
                obs_encoder=obs_encoder,
                obs_projection=obs_projection,
                ema=ema,
                optimizer=optimizer,
                stats=dataset.stats,
                loss=avg_loss,
                filename=f'diffusion_checkpoint_{epoch}.pt'
            )
    
    # save final model
    save_checkpoint(
        save_dir=save_dir,
        epoch=num_epochs-1,
        model=model,
        obs_encoder=obs_encoder,
        obs_projection=obs_projection,
        ema=ema,
        optimizer=optimizer,
        stats=dataset.stats,
        loss=avg_loss,
        filename='diffusion_final.pt'
    )
    
    return {
        'model': model,
        'obs_encoder': obs_encoder,
        'obs_projection': obs_projection,
        'ema': ema,
        'noise_scheduler': noise_scheduler,
        'optimizer': optimizer,
        'stats': dataset.stats
    }

def save_checkpoint(save_dir, epoch, model, obs_encoder, obs_projection, 
                   ema, optimizer, stats, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': obs_encoder.state_dict(),
        'projection_state_dict': obs_projection.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'loss': loss,
    }, os.path.join(save_dir, filename))