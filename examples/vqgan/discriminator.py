"""
Ported from Paella
"""
import torch
from torch import nn


class PaellaDiscriminator(nn.Module):
    def __init__(self, config):
        channels = config.discriminator.channels
        cond_channels = config.discriminator.cond_channels
        hidden_channels = config.discriminator.hidden_channels
        depth = config.discriminator.depth
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(channels, hidden_channels // (2**d), kernel_size=3, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = hidden_channels // (2 ** max((d - i), 0))
            c_out = hidden_channels // (2 ** max((d - 1 - i), 0))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = nn.Conv2d(
            (hidden_channels + cond_channels) if cond_channels > 0 else hidden_channels, 1, kernel_size=1
        )
        self.logits = nn.Sigmoid()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(
                cond.size(0),
                cond.size(1),
                1,
                1,
            ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x
