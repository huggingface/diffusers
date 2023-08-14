import math

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from .common import AttnBlock, LayerNorm2d, ResBlock, TimestepBlock


class WuerstchenPrior(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, c_in=16, c=1280, c_cond=1024, c_r=64, depth=16, nhead=16, latent_size=(12, 12), dropout=0.1):
        super().__init__()
        self.c_r = c_r
        self.projection = nn.Conv2d(c_in, c, kernel_size=1)
        self.cond_mapper = nn.Sequential(
            nn.Linear(c_cond, c),
            nn.LeakyReLU(0.2),
            nn.Linear(c, c),
        )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResBlock(c, dropout=dropout))
            self.blocks.append(TimestepBlock(c, c_r))
            self.blocks.append(AttnBlock(c, c, nhead, self_attn=True, dropout=dropout))
        self.out = nn.Sequential(
            LayerNorm2d(c, elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c, c_in * 2, kernel_size=1),
        )

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb.to(dtype=r.dtype)

    def forward(self, x, r, c):
        x_in = x
        x = self.projection(x)
        c_embed = self.cond_mapper(c)
        r_embed = self.gen_r_embedding(r)
        for block in self.blocks:
            if isinstance(block, AttnBlock):
                x = block(x, c_embed)
            elif isinstance(block, TimestepBlock):
                x = block(x, r_embed)
            else:
                x = block(x)
        a, b = self.out(x).chunk(2, dim=1)
        # denoised = a / (1-(1-b).pow(2)).sqrt()
        return (x_in - a) / ((1 - b).abs() + 1e-5)
