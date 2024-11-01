import torch
import torch.nn as nn
import numpy as np
from diffusers import UNet1DModel, DDPMScheduler
from huggingface_hub import hf_hub_download

class ObservationEncoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256)
        )
    
    def forward(self, x): return self.net(x)

class ObservationProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32, 512))
        self.bias = nn.Parameter(torch.zeros(32))
    
    def forward(self, x):
        if x.size(-1) == 256:
            x = torch.cat([x, torch.zeros(*x.shape[:-1], 256, device=x.device)], dim=-1)
        return nn.functional.linear(x, self.weight, self.bias)

class DiffusionPolicy:
    def __init__(self, state_dim=5, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.stats = {'obs': {'min': torch.zeros(5), 'max': torch.tensor([512, 512, 512, 512, 2*np.pi])}, 'action': {'min': torch.zeros(2), 'max': torch.full((2,), 512)}}
        
        self.obs_encoder = ObservationEncoder(state_dim).to(device)
        self.obs_projection = ObservationProjection().to(device)

        self.model = UNet1DModel(
            sample_size=16, 
            in_channels=34, 
            out_channels=2, 
            layers_per_block=2, 
            block_out_channels=(128,),
            down_block_types=("DownBlock1D",), 
            up_block_types=("UpBlock1D",)
        ).to(device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, 
            beta_schedule="squaredcos_cap_v2"
        )
        
        checkpoint = torch.load(hf_hub_download("dorsar/diffusion_policy", "push_tblock.pt"), map_location=device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.obs_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.obs_projection.load_state_dict(checkpoint['projection_state_dict'])

    def normalize_data(self, data, stats):
        return ((data - stats['min']) / (stats['max'] - stats['min'])) * 2 - 1

    def unnormalize_data(self, ndata, stats):
        return ((ndata + 1) / 2) * (stats['max'] - stats['min']) + stats['min']

    @torch.no_grad()
    def predict(self, observation):
        observation = observation.to(self.device)
        normalized_obs = self.normalize_data(observation, self.stats['obs'])
        
        # conditioning through encoder and projection
        cond = self.obs_projection(self.obs_encoder(normalized_obs))
        cond = cond.view(normalized_obs.shape[0], -1, 1).expand(-1, -1, 16)
        
        # initialize action with noise
        action = torch.randn((observation.shape[0], 2, 16), device=self.device)
        self.noise_scheduler.set_timesteps(100)
        
        # denoise
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(torch.cat([action, cond], dim=1), t)
            action = self.noise_scheduler.step(
                model_output.sample, t, action
            ).prev_sample
        
        action = action.transpose(1, 2)  # [batch, 16, 2]
        action = self.unnormalize_data(action, self.stats['action'])
        return action


if __name__ == "__main__":
    policy = DiffusionPolicy()
    
    # sample of a single observation
    obs = torch.tensor([[
        256.0,  # robot arm x position (middle of screen)
        256.0,  # robot arm y position (middle of screen)
        200.0,  # block x position
        300.0,  # block y position
        np.pi/2 # block angle (90 degrees)
    ]])
    
    action = policy.predict(obs)
    
    print("Action shape:", action.shape)
    print("\nPredicted trajectory:")
    for i, (x, y) in enumerate(action[0]):
        print(f"Step {i:2d}: x={x:6.1f}, y={y:6.1f}")