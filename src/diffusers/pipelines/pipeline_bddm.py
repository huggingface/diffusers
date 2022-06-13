#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  DiffWave: A Versatile Diffusion Model for Audio Synthesis
#  (https://arxiv.org/abs/2009.09761)
#  Modified from https://github.com/philsyn/DiffWave-Vocoder
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from ..modeling_utils import ModelMixin
from ..configuration_utils import ConfigMixin
from ..pipeline_utils import DiffusionPipeline


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
        E.g. the embedding vector in the 128-dimensional space is
        [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)),
         cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
        diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                    diffusion steps for batch data
        diffusion_step_embed_dim_in (int, default=128):
                                    dimensionality of the embedding space for discrete diffusion steps
    Returns:
        the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)
    return diffusion_step_embed


"""
Below scripts were borrowed from
https://github.com/philsyn/DiffWave-Vocoder/blob/master/WaveNet.py
"""


def swish(x):
    return x * torch.sigmoid(x)


# dilated conv layer with kaiming_normal initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


# conv1x1 layer with zero initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


# every residual block (named residual layer in paper)
# contains one noncausal dilated conv
class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation,
                 diffusion_step_embed_dim_out):
        super().__init__()
        self.res_channels = res_channels

        # Use a FC layer for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # Dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels,
                                       kernel_size=3, dilation=dilation)

        # Add mel spectrogram upsampler and conditioner conv1x1 layer
        self.upsample_conv2d = nn.ModuleList()
        for s in [16, 16]:
            conv_trans2d = nn.ConvTranspose2d(1, 1, (3, 2 * s),
                                              padding=(1, s // 2),
                                              stride=(1, s))
            conv_trans2d = nn.utils.weight_norm(conv_trans2d)
            nn.init.kaiming_normal_(conv_trans2d.weight)
            self.upsample_conv2d.append(conv_trans2d)

        # 80 is mel bands
        self.mel_conv = Conv(80, 2 * self.res_channels, kernel_size=1)

        # Residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # Skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x, mel_spec, diffusion_step_embed = input_data
        h = x
        batch_size, n_channels, seq_len = x.shape
        assert n_channels == self.res_channels

        # Add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([batch_size, self.res_channels, 1])
        h += part_t

        # Dilated conv layer
        h = self.dilated_conv_layer(h)

        # Upsample spectrogram to size of audio
        mel_spec = torch.unsqueeze(mel_spec, dim=1)
        mel_spec = F.leaky_relu(self.upsample_conv2d[0](mel_spec), 0.4, inplace=False)
        mel_spec = F.leaky_relu(self.upsample_conv2d[1](mel_spec), 0.4, inplace=False)
        mel_spec = torch.squeeze(mel_spec, dim=1)

        assert mel_spec.size(2) >= seq_len
        if mel_spec.size(2) > seq_len:
            mel_spec = mel_spec[:, :, :seq_len]

        mel_spec = self.mel_conv(mel_spec)
        h += mel_spec

        # Gated-tanh nonlinearity
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        # Residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        # Normalize for training stability
        return (x + res) * math.sqrt(0.5), skip


class ResidualGroup(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out):
        super().__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # Use the shared two FC layers for diffusion step embedding
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        # Stack all residual blocks with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                ResidualBlock(res_channels, skip_channels,
                               dilation=2 ** (n % dilation_cycle),
                               diffusion_step_embed_dim_out=diffusion_step_embed_dim_out))

    def forward(self, input_data):
        x, mel_spectrogram, diffusion_steps = input_data

        # Embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # Pass all residual layers
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            # Use the output from last residual layer
            h, skip_n = self.residual_blocks[n]((h, mel_spectrogram, diffusion_step_embed))
            # Accumulate all skip outputs
            skip += skip_n

        # Normalize for training stability
        return skip * math.sqrt(1.0 / self.num_res_layers)


class DiffWave(ModelMixin, ConfigMixin):
    def __init__(
        self,
        in_channels=1,
        res_channels=128,
        skip_channels=128,
        out_channels=1,
        num_res_layers=30,
        dilation_cycle=10,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
    ):
        super().__init__()

        # register all init arguments with self.register
        self.register(
            in_channels=in_channels,
            res_channels=res_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            num_res_layers=num_res_layers,
            dilation_cycle=dilation_cycle,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
        )


        # Initial conv1x1 with relu
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU(inplace=False))
        # All residual layers
        self.residual_layer = ResidualGroup(res_channels,
                                            skip_channels,
                                            num_res_layers,
                                            dilation_cycle,
                                            diffusion_step_embed_dim_in,
                                            diffusion_step_embed_dim_mid,
                                            diffusion_step_embed_dim_out)
        # Final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(inplace=False), ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):
        audio, mel_spectrogram, diffusion_steps = input_data
        x = audio
        x = self.init_conv(x).clone()
        x = self.residual_layer((x, mel_spectrogram, diffusion_steps))
        return self.final_conv(x)


class BDDM(DiffusionPipeline):
    def __init__(self, diffwave, noise_scheduler):
        super().__init__()
        noise_scheduler = noise_scheduler.set_format("pt")
        self.register_modules(diffwave=diffwave, noise_scheduler=noise_scheduler)
    
    @torch.no_grad()
    def __call__(self, mel_spectrogram, generator, torch_device=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.diffwave.to(torch_device)
        
        mel_spectrogram = mel_spectrogram.to(torch_device)
        audio_length = mel_spectrogram.size(-1) * 256
        audio_size = (1, 1, audio_length)

        # Sample gaussian noise to begin loop
        audio = torch.normal(0, 1, size=audio_size, generator=generator).to(torch_device)

        timestep_values = self.noise_scheduler.timestep_values
        num_prediction_steps = len(self.noise_scheduler)
        for t in tqdm.tqdm(reversed(range(num_prediction_steps)), total=num_prediction_steps):
            # 1. predict noise residual
            ts = (torch.tensor(timestep_values[t]) * torch.ones((1, 1))).to(torch_device)
            residual = self.diffwave((audio, mel_spectrogram, ts))

            # 2. predict previous mean of audio x_t-1
            pred_prev_audio = self.noise_scheduler.step(residual, audio, t)

            # 3. optionally sample variance
            variance = 0
            if t > 0:
                noise = torch.normal(0, 1, size=audio_size, generator=generator).to(torch_device)
                variance = self.noise_scheduler.get_variance(t).sqrt() * noise

            # 4. set current audio to prev_audio: x_t -> x_t-1
            audio = pred_prev_audio + variance

        return audio