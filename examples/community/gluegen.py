from typing import Any
from diffusers import DiffusionPipeline
import torch

import torch 
import torch.nn as nn
from ldm.modules.x_transformer import AttentionLayers
#----------------------------------------------------------------------------
class translator_base(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2):
        super().__init__()
        
        self.dim_in = dim
        self.dim_out = dim_out

        self.net_tok = nn.Sequential(
            nn.Linear(num_tok, int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), num_tok),
            nn.LayerNorm(num_tok),
            
        )
        
        self.net_sen = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), dim_out),
            nn.LayerNorm(dim_out)
        )

    def forward(self, x):
        if self.dim_in == self.dim_out:
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
            x = x.transpose(1,2)

            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            x = x.transpose(1,2)
        else:
            x = self.net_sen(x)
            x = x.transpose(1,2)

            x = self.net_tok(x)
            x = x.transpose(1,2)
        return x

class Translator(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2, depth=5):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [translator_base(num_tok, dim, dim, mult=2)
                for d in range(depth)]
        )
        self.gelu = nn.GELU()

        self.tail = translator_base(num_tok, dim, dim_out, mult=2)
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x) + x
            x = self.gelu(x)
            
        x = self.tail(x)
        return x
    
class translator_base_noln(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2):
        super().__init__()
        
        self.dim_in = dim
        self.dim_out = dim_out

        self.net_tok = nn.Sequential(
            nn.Linear(num_tok, int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), num_tok),            
        )
        
        self.net_sen = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), dim_out),
        )

    def forward(self, x):
        if self.dim_in == self.dim_out:
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
            x = x.transpose(1,2)

            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            x = x.transpose(1,2)
        else:
            x = self.net_sen(x)
            x = x.transpose(1,2)

            x = self.net_tok(x)
            x = x.transpose(1,2)
        return x
    
class Translator_noln(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2, depth=5):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [translator_base(num_tok, dim, dim, mult=2)
                for d in range(depth)]
        )
        self.gelu = nn.GELU()

        self.tail = translator_base_noln(num_tok, dim, dim_out, mult=2)
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x) + x
            x = self.gelu(x)
            
        x = self.tail(x)
        return x



class StableGluegenDiffusion(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
    
    def __call__(self):

        return