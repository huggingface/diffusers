import torch.nn as nn
from torchvision.models import efficientnet_v2_s, efficientnet_v2_l


class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16, effnet="efficientnet_v2_s"):
        super().__init__()
        if effnet == "efficientnet_v2_s":
            self.backbone = efficientnet_v2_s(weights="DEFAULT").features.eval()
        else:
            print("Using EffNet L.")
            self.backbone = efficientnet_v2_l(weights="DEFAULT").features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))
