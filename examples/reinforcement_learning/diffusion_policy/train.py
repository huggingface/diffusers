import os

import torch
import torch.nn.functional as F
from config import DataConfig, ModelConfig
from dataset import SequentialDataset
from model import create_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers.training_utils import EMAModel


def train_diffusion(
    data_config: DataConfig,
    model_config: ModelConfig,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    device: torch.device = None
):
    """Train diffusion model using HuggingFace diffusers"""

    # Setup device
    if device is None:
        device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = SequentialDataset(data_config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,  # This helps with faster data transfer to GPU
        persistent_workers=True
    )

    # Create model components - all moved to the same device
    model, obs_encoder, obs_projection, noise_scheduler = create_model(
        data_config, model_config, device
    )

    # Ensure models are in training mode
    model.train()
    obs_encoder.train()

    # Setup optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': obs_encoder.parameters()},
        {'params': obs_projection.parameters()}
    ], lr=learning_rate)

    # Setup EMA
    ema = EMAModel(
        parameters=list(model.parameters()) +
                  list(obs_encoder.parameters()) +
                  list(obs_projection.parameters()),
        power=0.75
    )

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}')
        epoch_loss = []

        for batch in dataloader:
            # Move batch to device and ensure float32
            state = batch['state'].to(device, dtype=torch.float32)  # Ensure float32
            actions = batch['action'].to(device, dtype=torch.float32)  # Ensure float32
            batch_size = state.shape[0]

            # Zero gradients
            optimizer.zero_grad()

            try:
                # Encode observations
                obs_embedding = obs_encoder(state)

                # Sample noise and timesteps
                noise = torch.randn_like(actions, device=device)  # Create noise on same device
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=device
                ).long()

                # Add noise to actions
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

                # Reshape to channels format for UNet
                noisy_actions = noisy_actions.transpose(1, 2)
                noise = noise.transpose(1, 2)

                # Project the observation embedding
                obs_cond = obs_projection(obs_embedding)

                # Reshape to match sequence length
                obs_cond = obs_cond.unsqueeze(-1).expand(-1, -1, noisy_actions.shape[-1])

                # Concatenate along channel dimension
                model_input = torch.cat([noisy_actions, obs_cond], dim=1)

                # Predict noise
                noise_pred = model(model_input, timesteps).sample

                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)

            except RuntimeError as e:
                print(f"\nError in forward pass: {e}")
                print(f"State device: {state.device}")
                print(f"Actions device: {actions.device}")
                print(f"Noise device: {noise.device}")
                print(f"Model device: {next(model.parameters()).device}")
                print(f"Encoder device: {next(obs_encoder.parameters()).device}")
                raise e

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update EMA parameters
            ema.step(list(model.parameters()) +
                     list(obs_encoder.parameters()) +
                     list(obs_projection.parameters()))

            epoch_loss.append(loss.item())

            # Update progress
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        progress_bar.close()

        # Print epoch stats
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"\nEpoch {epoch} average loss: {avg_loss:.6f}")

        # Save checkpoint
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
