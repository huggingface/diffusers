from dataclasses import dataclass
from typing import Tuple


"""
Configurations for data and the model.
"""

# data configurations
@dataclass
class DataConfig:
    """Configurations for dataset"""
    dataset_path: str = None

    pred_horizon: int = 16      # number of future steps to predict
    obs_horizon: int = 2        # number of past observations used to condition the predictions
    action_horizon: int = 8     # number of actions to execute

    # data dimensions
    image_shape: Tuple[int, int, int] = (3, 96, 96)     # size of input images in pixels. (channels, height, width). channels: RGB=3, grayscale=1
    action_dim: int = 2                                 # eg. [vel_x, vel_y], joint velocities
    state_dim: int = 5                                   # eg. [x,y,z,angle,gripper], joint angles


@dataclass
class ModelConfig:
    """Configuration for neural networks"""
    # observation encoding
    obs_embed_dim: int = 256        # size of observation embedding

    # UNet configuration
    sample_size: int = 16           # length of the generated sequence - should equal pred_horizon
    in_channels: int = 2            # action space dimensions - should equal action_dim
    out_channels: int = 2           # action space dimensions - should equal action_dim
    layers_per_block: int = 2                                   # number of conv layers in each UNet block
    block_out_channels: Tuple[int, ...] = (128,)                # features per block
    norm_num_groups: int = 8                                    # group normalization
    down_block_types: Tuple[str, ...] = ("DownBlock1D",) * 1    # UNet architecture
    up_block_types: Tuple[str, ...] = ("UpBlock1D",) * 1        # UNet architecture

    def __post_init__(self):
        # For conditioning through input channels
        self.total_in_channels = self.in_channels + self.obs_embed_dim // 8 # actions + conditioning

