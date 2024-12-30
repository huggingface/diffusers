import numpy as np
import numpy.core.multiarray as multiarray
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.serialization import add_safe_globals

from diffusers import DDPMScheduler, UNet1DModel


add_safe_globals(
    [
        multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtype(np.float32).type,
        np.dtype(np.float64).type,
        np.dtype(np.int32).type,
        np.dtype(np.int64).type,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int32)),
        type(np.dtype(np.int64)),
    ]
)

"""
An example of using HuggingFace's diffusers library for diffusion policy,
generating smooth movement trajectories.

This implements a robot control model for pushing a T-shaped block into a target area.
The model takes in the robot arm position, block position, and block angle,
then outputs a sequence of 16 (x,y) positions for the robot arm to follow.
"""


class ObservationEncoder(nn.Module):
    """
    Converts raw robot observations (positions/angles) into a more compact representation

    state_dim (int): Dimension of the input state vector (default: 5)
        [robot_x, robot_y, block_x, block_y, block_angle]

    - Input shape: (batch_size, state_dim)
    - Output shape: (batch_size, 256)
    """

    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(), nn.Linear(512, 256))

    def forward(self, x):
        return self.net(x)


class ObservationProjection(nn.Module):
    """
    Takes the encoded observation and transforms it into 32 values that represent the current robot/block situation.
    These values are used as additional contextual information during the diffusion model's trajectory generation.

    - Input: 256-dim vector (padded to 512)
            Shape: (batch_size, 256)
    - Output: 32 contextual information values for the diffusion model
            Shape: (batch_size, 32)
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32, 512))
        self.bias = nn.Parameter(torch.zeros(32))

    def forward(self, x):  # pad 256-dim input to 512-dim with zeros
        if x.size(-1) == 256:
            x = torch.cat([x, torch.zeros(*x.shape[:-1], 256, device=x.device)], dim=-1)
        return nn.functional.linear(x, self.weight, self.bias)


class DiffusionPolicy:
    """
    Implements diffusion policy for generating robot arm trajectories.
    Uses diffusion to generate sequences of positions for a robot arm, conditioned on
    the current state of the robot and the block it needs to push.

    The model expects observations in pixel coordinates (0-512 range) and block angle in radians.
    It generates trajectories as sequences of (x,y) coordinates also in the 0-512 range.
    """

    def __init__(self, state_dim=5, device="cpu"):
        self.device = device

        # define valid ranges for inputs/outputs
        self.stats = {
            "obs": {"min": torch.zeros(5), "max": torch.tensor([512, 512, 512, 512, 2 * np.pi])},
            "action": {"min": torch.zeros(2), "max": torch.full((2,), 512)},
        }

        self.obs_encoder = ObservationEncoder(state_dim).to(device)
        self.obs_projection = ObservationProjection().to(device)

        # UNet model that performs the denoising process
        # takes in concatenated action (2 channels) and context (32 channels) = 34 channels
        # outputs predicted action (2 channels for x,y coordinates)
        self.model = UNet1DModel(
            sample_size=16,  # length of trajectory sequence
            in_channels=34,
            out_channels=2,
            layers_per_block=2,  # number of layers per each UNet block
            block_out_channels=(128,),  # number of output neurons per layer in each block
            down_block_types=("DownBlock1D",),  # reduce the resolution of data
            up_block_types=("UpBlock1D",),  # increase the resolution of data
        ).to(device)

        # noise scheduler that controls the denoising process
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,  # number of denoising steps
            beta_schedule="squaredcos_cap_v2",  # type of noise schedule
        )

        # load pre-trained weights from HuggingFace
        checkpoint = torch.load(
            hf_hub_download("dorsar/diffusion_policy", "push_tblock.pt"), weights_only=True, map_location=device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.obs_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.obs_projection.load_state_dict(checkpoint["projection_state_dict"])

    # scales data to [-1, 1] range for neural network processing
    def normalize_data(self, data, stats):
        return ((data - stats["min"]) / (stats["max"] - stats["min"])) * 2 - 1

    # converts normalized data back to original range
    def unnormalize_data(self, ndata, stats):
        return ((ndata + 1) / 2) * (stats["max"] - stats["min"]) + stats["min"]

    @torch.no_grad()
    def predict(self, observation):
        """
        Generates a trajectory of robot arm positions given the current state.

        Args:
            observation (torch.Tensor): Current state [robot_x, robot_y, block_x, block_y, block_angle]
                                    Shape: (batch_size, 5)

        Returns:
            torch.Tensor: Sequence of (x,y) positions for the robot arm to follow
                        Shape: (batch_size, 16, 2) where:
                        - 16 is the number of steps in the trajectory
                        - 2 is the (x,y) coordinates in pixel space (0-512)

        The function first encodes the observation, then uses it to condition a diffusion
        process that gradually denoises random trajectories into smooth, purposeful movements.
        """
        observation = observation.to(self.device)
        normalized_obs = self.normalize_data(observation, self.stats["obs"])

        # encode the observation into context values for the diffusion model
        cond = self.obs_projection(self.obs_encoder(normalized_obs))
        # keeps first & second dimension sizes unchanged, and multiplies last dimension by 16
        cond = cond.view(normalized_obs.shape[0], -1, 1).expand(-1, -1, 16)

        # initialize action with noise - random noise that will be refined into a trajectory
        action = torch.randn((observation.shape[0], 2, 16), device=self.device)

        # denoise
        # at each step `t`, the current noisy trajectory (`action`) & conditioning info (context) are
        # fed into the model to predict a denoised trajectory, then uses self.noise_scheduler.step to
        # apply this prediction & slightly reduce the noise in `action` more

        self.noise_scheduler.set_timesteps(100)
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(torch.cat([action, cond], dim=1), t)
            action = self.noise_scheduler.step(model_output.sample, t, action).prev_sample

        action = action.transpose(1, 2)  # reshape to [batch, 16, 2]
        action = self.unnormalize_data(action, self.stats["action"])  # scale back to coordinates
        return action


if __name__ == "__main__":
    policy = DiffusionPolicy()

    # sample of a single observation
    # robot arm starts in center, block is slightly left and up, rotated 90 degrees
    obs = torch.tensor(
        [
            [
                256.0,  # robot arm x position (middle of screen)
                256.0,  # robot arm y position (middle of screen)
                200.0,  # block x position
                300.0,  # block y position
                np.pi / 2,  # block angle (90 degrees)
            ]
        ]
    )

    action = policy.predict(obs)

    print("Action shape:", action.shape)  # should be [1, 16, 2] - one trajectory of 16 x,y positions
    print("\nPredicted trajectory:")
    for i, (x, y) in enumerate(action[0]):
        print(f"Step {i:2d}: x={x:6.1f}, y={y:6.1f}")
