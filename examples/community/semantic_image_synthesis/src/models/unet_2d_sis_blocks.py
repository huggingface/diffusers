# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copied from here :
# https://github.com/WeilunWang/semantic-diffusion-model/blob/main/guided_diffusion/unet.py
# https://arxiv.org/abs/2207.00050

from diffusers.models.unet_2d import UNet2DOutput
import torch.nn as nn
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.activations import get_activation
from diffusers.utils import is_torch_version, logging

import shutil
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_down_block(
    down_block_type,
    in_channels,
    out_channels,
    embedding_dim,
    attention_head_dim=None,
    num_res_blocks=2,
    activation="silu"
):
    # If attn head dim is not defined, we default it to the number of heads
    if down_block_type == "ConvBlock":
        return SISEncConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,            
            num_res_blocks=num_res_blocks,
            activation=activation
        )
    elif down_block_type == "DownBlock":
        return SISDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            activation=activation
        )
    elif down_block_type == "DownAttnBlock":
        return SISDownAttnBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            attention_head_dim=attention_head_dim,
            activation=activation
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type,
    in_channels,
    out_channels,
    label_channels,
    embedding_dim,
    attention_head_dim=None,
    num_res_blocks=3,
    activation="silu"
):
    if up_block_type =="UpBlock":
        return SISUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            label_channels=label_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            activation=activation
        )        
    elif up_block_type == "UpAttnBlock":
        return SISUpAttnBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            label_channels=label_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            attention_head_dim=attention_head_dim,
            activation=activation
        )
    elif up_block_type == "ConvBlock":
        return SISDecConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            label_channels=label_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            activation=activation
        )
    raise ValueError(f"{up_block_type} does not exist.")

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class SPADEGroupNorm(nn.Module):
    """
    Inspired from theses papers :
        https://arxiv.org/pdf/1903.07291.pdf
        https://arxiv.org/pdf/2207.00050.pdf

    """
    def __init__(self, in_channels, label_channels, eps = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, affine=False) # 32/16
        self.eps = eps
        nhidden = 128
        self.conv_shared = nn.Sequential(
            nn.Conv2d(label_channels, nhidden, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv_gamma = nn.Conv2d(nhidden, in_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(nhidden, in_channels, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.conv_shared(segmap)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)

        return x*(1+gamma) + beta

class SISEncBlock(nn.Module):
    """
        Implementation based on this paper :
        https://arxiv.org/pdf/2207.00050.pdf
        This block contains :
        - [Optionnal] Downsampling
        - One SDM Encoder Resblock
        - [Optionnal] AttentionBlock
    """
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 embedding_dim:int,
                 downsample:bool=False,
                 attention:bool=False,
                 attention_head_dim:int=None,
                 activation='silu'):
        """_summary_

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            embeding_dim (int): size of the time embedding
            downsample (bool, optional): downsample or not in this block. Defaults to False.
            attention (bool, optional): attention or not in this block. Defaults to False.
            attention_head_dim (int, optional): total dimension of the attention layer. Defaults to None.
            activation (str, optional): activation function to use. Defaults to 'silu'.
        """
        super().__init__()
        # Optionnal layers
        self.downsample = nn.AvgPool2d(2,2) if downsample else nn.Identity()
        # Block layers
        self.gn1 = nn.GroupNorm(num_groups=32,num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.embmlp = nn.Linear(embedding_dim,2*out_channels)
        self.gn2 = nn.GroupNorm(num_groups=32,num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.skipconnection = nn.Conv2d(in_channels,out_channels,1)
        # Attention
        if attention and attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels
        self.attention = Attention(query_dim=out_channels,heads=out_channels//attention_head_dim,dim_head=attention_head_dim) if attention else nn.Identity()
        # Activation function
        self.activation = get_activation(activation)
        
    def forward(self,x,emb):
        # DOWNSAMPLE [OPTIONAL]
        x_ds = self.downsample(x)
        # SKIP CONNECTION
        x_skip = self.skipconnection(x_ds)
        # FIRST STAGE
        y1 = self.gn2(self.conv1(self.activation(self.gn1(x_ds))))
        ## Embedding FC
        b,w = torch.chunk(self.embmlp(emb),2,dim=-1)
        y2 = (1+w[...,None,None])*y1 + b[...,None,None] 
        # THIRD STAGE
        y3 = self.conv2(self.activation(y2))
        h = y3+x_skip
        # ATTN [OPTIONAL]
        h = self.attention(h)
        return h

class SISDecBlock(nn.Module):
    """
        Implementation based on this paper :
        https://arxiv.org/pdf/2207.00050.pdf
        This block contains :
        - One SDM Decoder Resblock
        - [Optionnal] AttentionBlock
        - [Optionnal] Upsampling
    """
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 label_channels:int,
                 embeding_dim:int,
                 upsample:bool=False,
                 attention:bool=False,
                 attention_head_dim:int=None,
                 activation='silu'):
        super().__init__()
        self.spade1 = SPADEGroupNorm(in_channels,label_channels)
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.spade2 = SPADEGroupNorm(out_channels,label_channels)
        self.embmlp = nn.Linear(embeding_dim,2*out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.skipconnection = nn.Conv2d(in_channels,out_channels,1)
        # Optionnal layers
        if attention and attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels
        self.attention = Attention(query_dim=out_channels,heads=out_channels//attention_head_dim,dim_head=attention_head_dim) if attention else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2) if upsample else nn.Identity()
        self.activation = get_activation(activation)
    def forward(self,x,emb,cond):
        ################# Skip connection + Downsample
        x_skip = self.skipconnection(x)
        ################# First stage
        s1 = self.spade1(x,cond)
        y1 = self.conv1(self.activation(s1))
        ################# Second stage
        s2 = self.spade2(y1,cond)
        ################ Embedding FC
        b,w = torch.chunk(self.embmlp(emb),2,dim=-1)
        y2 = (1+w[...,None,None]*s2) + b[...,None,None]
        ################# Third stage
        y3 = self.conv2(self.activation(y2))
        h = y3+x_skip
        ################# Attention
        h = self.attention(h)
        ################# Upsample
        h = self.upsample(h)
        return h

class SISSequential(nn.Sequential):
    """SISSequential

    A utility class in order to define a sequential that adapts forward inputs
    depending on the layer.

    """
    def forward(self,x,emb,cond=None):
        for layer in self:
            if isinstance(layer,SISEncBlock):
                x = layer.forward(x,emb)
            elif isinstance(layer,SISDecBlock):
                x = layer.forward(x,emb,cond)
            else:
                x = layer(x)
        return x

# Declaration of blocks
class SISEncConvBlock(SISSequential):
    def __init__(self,in_channels:int,out_channels:int,embedding_dim:int,num_res_blocks:int=2,activation:str='silu'):
        super().__init__()
        self.append(nn.Conv2d(in_channels,out_channels,3,1,1))
        for i in range(num_res_blocks):
            self.append(SISEncBlock(out_channels,out_channels,embedding_dim,activation=activation))

class SISDownBlock(SISSequential):
    def __init__(self,in_channels:int,out_channels:int,embedding_dim:int,num_res_blocks:int=2,activation:str='silu'):
        super().__init__()        
        for i in range(num_res_blocks):
            is_downsample = i==0
            self.append(SISEncBlock(
                in_channels if not i else out_channels,
                out_channels,
                embedding_dim,
                downsample=is_downsample,
                activation=activation
                ))

class SISDownAttnBlock(SISSequential):
    def __init__(self,in_channels:int,out_channels:int,embedding_dim:int,num_res_blocks:int=2,attention_head_dim:int=None,activation:str='silu'):
        super().__init__()
        for i in range(num_res_blocks):
            is_downsample = i==0 # Only first block downsample
            is_attention = i==num_res_blocks-1 # Only last block attention
            self.append(SISEncBlock(
                in_channels if not i else out_channels,
                out_channels,
                embedding_dim,
                downsample=is_downsample,
                attention=is_attention,
                attention_head_dim=attention_head_dim,
                activation=activation
                ))

class SISHeadAttnBlock(SISSequential):
    def __init__(self,in_channels:int,out_channels:int,label_channels:int,embedding_dim:int,attention_head_dim:int=None,activation:str='silu'):
        super().__init__()
        for i in range(2):
            self.append(SISDecBlock(
                in_channels,
                out_channels,
                label_channels,
                embedding_dim,
                upsample=False,
                attention=True if i==0 else False,
                attention_head_dim=attention_head_dim,
                activation=activation))

class SISUpBlock(SISSequential):
    def __init__(self,in_channels:int,out_channels:int,label_channels:int,embedding_dim:int,num_res_blocks:int=3,activation:str='silu'):
        super().__init__()
        for i in range(num_res_blocks):
            is_upsample = i==num_res_blocks-1
            self.append(SISDecBlock(
                in_channels if not i else out_channels,
                out_channels,
                label_channels,
                embedding_dim,
                upsample=is_upsample,
                attention=False,
                activation=activation))

class SISUpAttnBlock(SISSequential):
    def __init__(self,in_channels:int,out_channels:int,label_channels:int,embedding_dim:int,attention_head_dim:int=None,num_res_blocks:int=3,activation:str='silu'):
        super().__init__()
        for i in range(num_res_blocks):
            is_attention = i==num_res_blocks-1
            is_upsample = i==num_res_blocks-1
            self.append(SISDecBlock(
                in_channels if not i else out_channels,
                out_channels,
                label_channels,
                embedding_dim,
                upsample=is_upsample,
                attention=is_attention,
                attention_head_dim=attention_head_dim,
                activation=activation))

class SISDecConvBlock(SISSequential):
    def __init__(self,in_channels:int,out_channels:int,label_channels:int,embedding_dim:int,num_res_blocks:int=3,activation:str='silu'):
        super().__init__()
        for i in range(num_res_blocks):
            self.append(SISDecBlock(
                in_channels,
                in_channels,
                label_channels,
                embedding_dim,
                activation=activation))
        # We add the convolution layer...
        self.append(nn.Conv2d(in_channels,out_channels,3,1,1))