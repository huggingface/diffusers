from dataclasses import dataclass

from diffusers import UNet2DModel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


@dataclass
class DiffusionTrainingArgs:
    resolution: int = 64
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-6
    adam_epsilon: float = 1e-08
    use_ema: bool = True
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4
    ema_max_decay: float = 0.9999
    batch_size: int = 64
    num_epochs: int = 500


def get_train_transforms(training_config):
    # Get standard image transforms
    return Compose(
        [
            Resize(training_config.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(training_config.resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )


def get_unet(training_config):
    # Initialize a generic UNet model to use in our example
    return UNet2DModel(
        sample_size=training_config.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
