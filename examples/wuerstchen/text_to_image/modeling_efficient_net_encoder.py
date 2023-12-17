import torch.nn as nn
from torchvision.models import efficientnet_v2_l, efficientnet_v2_s

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class EfficientNetEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, c_latent=16, c_cond=1280, effnet="efficientnet_v2_s"):
        super().__init__()

        if effnet == "efficientnet_v2_s":
            self.backbone = efficientnet_v2_s(weights="DEFAULT").features
        else:
            self.backbone = efficientnet_v2_l(weights="DEFAULT").features
        self.mapper = nn.Sequential(
            nn.Conv2d(c_cond, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))
