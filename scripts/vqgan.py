import torch
from torch import nn
import numpy as np
import math
from tqdm import tqdm
import time
from torchtools.nn import VectorQuantize

class ResBlock(nn.Module):
    def __init__(self, c, c_hidden):
        super().__init__()
        # depthwise/attention
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(c, c, kernel_size=3, groups=c)
        )

        # channelwise
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )
        
        self.gammas = nn.Parameter(torch.zeros(6),  requires_grad=True)
        
        # Init weights
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


    def _norm(self, x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        mods = self.gammas
            
        x_temp = self._norm(x, self.norm1) * (1 + mods[0]) + mods[1]
        x = x + self.depthwise(x_temp) * mods[2]

        x_temp = self._norm(x, self.norm2) * (1 + mods[3]) + mods[4]
        x = x + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]
        
        return x

class VQModel(nn.Module):
    def __init__(self, levels=2, bottleneck_blocks=12, c_hidden=384, c_latent=4, codebook_size=8192, scale_factor=0.3764): # 1.0
        super().__init__()
        self.c_latent = c_latent
        self.scale_factor = scale_factor
        c_levels = [c_hidden//(2**i) for i in reversed(range(levels))]
        
        # Encoder blocks
        self.in_block = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(3*4, c_levels[0], kernel_size=1)
        )
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(nn.Conv2d(c_levels[i-1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = ResBlock(c_levels[i], c_levels[i]*4)
            down_blocks.append(block)
        down_blocks.append(nn.Sequential(
            nn.Conv2d(c_levels[-1], c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent), # then normalize them to have mean 0 and std 1
        ))
        self.down_blocks = nn.Sequential(*down_blocks)
        self.down_blocks[0]
        
        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(c_latent, k=codebook_size)        

        # Decoder blocks
        up_blocks = [nn.Sequential(
            nn.Conv2d(c_latent, c_levels[-1], kernel_size=1)
        )]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ResBlock(c_levels[levels-1-i], c_levels[levels-1-i]*4)
                up_blocks.append(block)
            if i < levels-1:
                up_blocks.append(nn.ConvTranspose2d(c_levels[levels-1-i], c_levels[levels-2-i], kernel_size=4, stride=2, padding=1))
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            nn.Conv2d(c_levels[0], 3*4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def encode(self, x):
        x = self.in_block(x)
        x = self.down_blocks(x)
        qe, (vq_loss, commit_loss), indices = self.vquantizer.forward(x, dim=1)
        return qe / self.scale_factor, x / self.scale_factor, indices, vq_loss + commit_loss * 0.25

    def decode(self, x):
        x = x * self.scale_factor
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def decode_indices(self, x):
        x = self.vquantizer.idx2vq(x, dim=1)
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x
        
    def forward(self, x, quantize=False):
        qe, x, _, vq_loss = self.encode(x, quantize)
        x = self.decode(qe)
        return x, vq_loss

class Discriminator(nn.Module):
    def __init__(self, c_in=3, c_cond=0, c_hidden=512, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(c_in, c_hidden // (2 ** d), kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = c_hidden // (2 ** max((d - i), 0))
            c_out = c_hidden // (2 ** max((d - 1 - i), 0))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = nn.Conv2d((c_hidden + c_cond) if c_cond > 0 else c_hidden, 1, kernel_size=1)
        self.logits = nn.Sigmoid()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(cond.size(0), cond.size(1), 1, 1, ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x