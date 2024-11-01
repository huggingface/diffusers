import torch
import torch.nn as nn
from diffusers import UNet1DModel, DDPMScheduler

class DiffusionPolicy:
    def __init__(self, state_dim=5, action_dim=2, sequence_length=16, condition_dim=2, device= "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        
        # observation encoder - output dim matches condition_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, condition_dim),
        ).to(device)
        
        self.model = UNet1DModel(
            sample_size=sequence_length,
            in_channels=action_dim + condition_dim,
            out_channels=action_dim,
            layers_per_block=2,
            block_out_channels=(128,),
            down_block_types=("DownBlock1D",),
            up_block_types=("UpBlock1D",),
        ).to(device)
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2"
        )
    
    @torch.no_grad()
    def predict(self, observation, num_inference_steps=10):
        """Generate action sequence from an observation."""
        batch_size = observation.shape[0]
        
        # encode observation
        observation = observation.to(self.device)
        cond = self.obs_encoder(observation)  # [B, condition_dim]
        
        # expand condition to sequence length
        cond = cond.view(batch_size, self.condition_dim, 1)
        cond = cond.expand(batch_size, self.condition_dim, self.sequence_length)
        
        action = torch.randn(
            (batch_size, self.action_dim, self.sequence_length),
            device=self.device
        )
        
        # denoise
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_input = torch.cat([action, cond], dim=1)
            
            # predict and remove noise
            noise_pred = self.model(model_input, t).sample
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        return action.transpose(1, 2)  # [batch_size, sequence_length, action_dim]

if __name__ == "__main__":
    policy = DiffusionPolicy()
    
    # a sample single observation
    observation = torch.randn(1, 5)  # [batch_size, state_dim]
    actions = policy.predict(observation)
    print("Generated action sequence shape:", actions.shape)  # Should be [1, 16, 2]