import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l, efficientnet_v2_s
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    PILToTensor,
    ConvertImageDtype,
)

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


EFFNET_PREPROCESS = Compose(
    [
        Resize(384, interpolation=InterpolationMode.BILINEAR, antialias=True),
        CenterCrop(384),
        PILToTensor(),
        ConvertImageDtype(torch.float),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class EfficientNetEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, c_latent=16, c_cond=1280, effnet="efficientnet_v2_s"):
        super().__init__()

        if effnet == "efficientnet_v2_s":
            self.backbone = efficientnet_v2_s(weights="DEFAULT").features.eval()
        else:
            self.backbone = efficientnet_v2_l(weights="DEFAULT").features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(c_cond, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))
