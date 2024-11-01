import torch
import torch.nn as nn

from diffusers import DDPMScheduler, UNet1DModel


class ObservationEncoder(nn.Module):
    """
    Encodes observations for conditioning.
    Ie. takes raw observations and converts them to a fixed-size encoding.
    """
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

def create_model(data_config, model_config, device):
    """Creates all model components"""

    # observation encoder
    obs_encoder = ObservationEncoder(
        obs_dim=data_config.state_dim,
        embed_dim=model_config.obs_embed_dim
    ).to(device)

    # UNet1D model
    model = UNet1DModel(
        sample_size=model_config.sample_size,
        in_channels=model_config.total_in_channels,
        out_channels=model_config.out_channels,
        layers_per_block=model_config.layers_per_block,
        block_out_channels=model_config.block_out_channels,
        norm_num_groups=model_config.norm_num_groups,
        down_block_types=model_config.down_block_types,
        up_block_types=model_config.up_block_types,
    ).to(device)

    # observation projection layer
    obs_projection = nn.Linear(
        model_config.obs_embed_dim * data_config.obs_horizon,
        model_config.obs_embed_dim // 8
    ).to(device)

    # noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon"
    )

    with torch.no_grad():
        sample_input = torch.randn(1, model_config.total_in_channels, model_config.sample_size).to(device)
        sample_timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (1,), device=device
        ).long()
        try:
            output = model(sample_input, sample_timesteps).sample
            print(f"Sample output device: {output.device}")
        except RuntimeError as e:
            print(f"RuntimeError during sample forward pass: {e}")


    return model, obs_encoder, obs_projection, noise_scheduler
