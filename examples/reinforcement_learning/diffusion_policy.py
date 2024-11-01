import torch
import torch.nn as nn
import numpy as np
from diffusers import UNet1DModel, DDPMScheduler
from huggingface_hub import hf_hub_download

"""
An example of using HuggingFace's diffusers library for diffusion policy, 
generating smooth movement trajectories.

This implements a robot control model for pushing a T-shaped block into a target area.
The model takes in the robot arm position, block position, and block angle, 
then outputs a sequence of 16 (x,y) positions for the robot arm to follow.
"""

"""
Converts raw robot observations (positions/angles) into a more compact representation
- Input: 5 values (robot_x, robot_y, block_x, block_y, block_angle)
- Output: 256-dimensional vector
"""
class ObservationEncoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256)
        )
    
    def forward(self, x): return self.net(x)


"""
Takes the encoded observation and transforms it into 32 values that represent 
the current robot/block situation. These values are used as additional contextual 
information during the diffusion model's trajectory generation.
- Input: 256-dim vector (padded to 512)
- Output: 32 contextual information values for the diffusion model
"""
class ObservationProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32, 512))
        self.bias = nn.Parameter(torch.zeros(32))
    
    def forward(self, x):        # pad 256-dim input to 512-dim with zeros
        if x.size(-1) == 256:
            x = torch.cat([x, torch.zeros(*x.shape[:-1], 256, device=x.device)], dim=-1)
        return nn.functional.linear(x, self.weight, self.bias)

class DiffusionPolicy:
    def __init__(self, state_dim=5, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # define valid ranges for inputs/outputs
        self.stats = {'obs': {'min': torch.zeros(5), 'max': torch.tensor([512, 512, 512, 512, 2*np.pi])}, 'action': {'min': torch.zeros(2), 'max': torch.full((2,), 512)}}
        
        self.obs_encoder = ObservationEncoder(state_dim).to(device)
        self.obs_projection = ObservationProjection().to(device)
        
        # UNet model that performs the denoising process
        # takes in concatenated action (2 channels) and context (32 channels) = 34 channels
        # outputs predicted action (2 channels for x,y coordinates)
        self.model = UNet1DModel(
            sample_size=16, # length of trajectory sequence
            in_channels=34, 
            out_channels=2, 
            layers_per_block=2, 
            block_out_channels=(128,),
            down_block_types=("DownBlock1D",), 
            up_block_types=("UpBlock1D",)
        ).to(device)

        # noise scheduler that controls the denoising process
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,                # number of denoising steps 
            beta_schedule="squaredcos_cap_v2"       # type of noise schedule
        )   
        
        # load pre-trained weights from HuggingFace
        checkpoint = torch.load(hf_hub_download("dorsar/diffusion_policy", "push_tblock.pt"), map_location=device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.obs_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.obs_projection.load_state_dict(checkpoint['projection_state_dict'])

    # scales data to [-1, 1] range for neural network processing
    def normalize_data(self, data, stats):
        return ((data - stats['min']) / (stats['max'] - stats['min'])) * 2 - 1

    # converts normalized data back to original range
    def unnormalize_data(self, ndata, stats):
        return ((ndata + 1) / 2) * (stats['max'] - stats['min']) + stats['min']

    @torch.no_grad()
    def predict(self, observation):
        observation = observation.to(self.device)
        normalized_obs = self.normalize_data(observation, self.stats['obs'])
        
        # encode the observation into context values for the diffusion model
        cond = self.obs_projection(self.obs_encoder(normalized_obs))
        cond = cond.view(normalized_obs.shape[0], -1, 1).expand(-1, -1, 16)
        
        # initialize action with noise - random noise that will be refined into a trajectory
        action = torch.randn((observation.shape[0], 2, 16), device=self.device)
        self.noise_scheduler.set_timesteps(100)
        
        # denoise the random action into a smooth trajectory
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(torch.cat([action, cond], dim=1), t)
            action = self.noise_scheduler.step(
                model_output.sample, t, action
            ).prev_sample
        
        action = action.transpose(1, 2)  # reshape to [batch, 16, 2]
        action = self.unnormalize_data(action, self.stats['action']) # scale back to pixel coordinates
        return action


if __name__ == "__main__":
    policy = DiffusionPolicy()
    
    # sample of a single observation
    # robot arm starts in center, block is slightly left and up, rotated 90 degrees
    obs = torch.tensor([[
        256.0,  # robot arm x position (middle of screen)
        256.0,  # robot arm y position (middle of screen)
        200.0,  # block x position
        300.0,  # block y position
        np.pi/2 # block angle (90 degrees)
    ]])
    
    action = policy.predict(obs)
    
    print("Action shape:", action.shape)    # should be [1, 16, 2] - one trajectory of 16 x,y positions
    print("\nPredicted trajectory:")
    for i, (x, y) in enumerate(action[0]):
        print(f"Step {i:2d}: x={x:6.1f}, y={y:6.1f}")