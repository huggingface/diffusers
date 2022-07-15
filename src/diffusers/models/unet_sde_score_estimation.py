# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# helpers functions

import functools
import math

import numpy as np
import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin
from .attention import AttentionBlock
from .embeddings import GaussianFourierProjection, get_timestep_embedding
from .resnet import Downsample2D, FirDownsample2D, FirUpsample2D, ResnetBlock2D, Upsample2D
from .unet_new import UNetMidBlock2D


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method="cat"):
        super().__init__()
        # 1x1 convolution with DDPM initialization.
        self.Conv_0 = nn.Conv2d(dim1, dim2, kernel_size=1, padding=0)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        elif self.method == "sum":
            return h + y
        else:
            raise ValueError(f"Method {self.method} not recognized.")


class NCSNpp(ModelMixin, ConfigMixin):
    """NCSN++ model"""

    def __init__(
        self,
        image_size=1024,
        num_channels=3,
        centered=False,
        attn_resolutions=(16,),
        ch_mult=(1, 2, 4, 8, 16, 32, 32, 32),
        conditional=True,
        conv_size=3,
        dropout=0.0,
        embedding_type="fourier",
        fir=True,
        fir_kernel=(1, 3, 3, 1),
        fourier_scale=16,
        init_scale=0.0,
        nf=16,
        num_res_blocks=1,
        progressive="output_skip",
        progressive_combine="sum",
        progressive_input="input_skip",
        resamp_with_conv=True,
        scale_by_sigma=True,
        skip_rescale=True,
        continuous=True,
    ):
        super().__init__()
        self.register_to_config(
            image_size=image_size,
            num_channels=num_channels,
            centered=centered,
            attn_resolutions=attn_resolutions,
            ch_mult=ch_mult,
            conditional=conditional,
            conv_size=conv_size,
            dropout=dropout,
            embedding_type=embedding_type,
            fir=fir,
            fir_kernel=fir_kernel,
            fourier_scale=fourier_scale,
            init_scale=init_scale,
            nf=nf,
            num_res_blocks=num_res_blocks,
            progressive=progressive,
            progressive_combine=progressive_combine,
            progressive_input=progressive_input,
            resamp_with_conv=resamp_with_conv,
            scale_by_sigma=scale_by_sigma,
            skip_rescale=skip_rescale,
            continuous=continuous,
        )
        self.act = nn.SiLU()
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2**i) for i in range(self.num_resolutions)]

        self.conditional = conditional
        self.skip_rescale = skip_rescale
        self.progressive = progressive
        self.progressive_input = progressive_input
        self.embedding_type = embedding_type
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            modules.append(GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        modules.append(nn.Linear(embed_dim, nf * 4))
        modules.append(nn.Linear(nf * 4, nf * 4))

        AttnBlock = functools.partial(AttentionBlock, overwrite_linear=True, rescale_output_factor=math.sqrt(2.0))

        if self.fir:
            Up_sample = functools.partial(FirUpsample2D, fir_kernel=fir_kernel, use_conv=resamp_with_conv)
        else:
            Up_sample = functools.partial(Upsample2D, name="Conv2d_0")

        if progressive == "output_skip":
            self.pyramid_upsample = Up_sample(channels=None, use_conv=False)
        elif progressive == "residual":
            pyramid_upsample = functools.partial(Up_sample, use_conv=True)

        if self.fir:
            Down_sample = functools.partial(FirDownsample2D, fir_kernel=fir_kernel, use_conv=resamp_with_conv)
        else:
            Down_sample = functools.partial(Downsample2D, padding=0, name="Conv2d_0")

        if progressive_input == "input_skip":
            self.pyramid_downsample = Down_sample(channels=None, use_conv=False)
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(Down_sample, use_conv=True)

        channels = num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(nn.Conv2d(channels, nf, kernel_size=3, padding=1))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(
                    ResnetBlock2D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        temb_channels=4 * nf,
                        output_scale_factor=np.sqrt(2.0),
                        non_linearity="silu",
                        groups=min(in_ch // 4, 32),
                        groups_out=min(out_ch // 4, 32),
                        overwrite_for_score_vde=True,
                    )
                )
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != self.num_resolutions - 1:
                modules.append(
                    ResnetBlock2D(
                        in_channels=in_ch,
                        temb_channels=4 * nf,
                        output_scale_factor=np.sqrt(2.0),
                        non_linearity="silu",
                        groups=min(in_ch // 4, 32),
                        groups_out=min(out_ch // 4, 32),
                        overwrite_for_score_vde=True,
                        down=True,
                        kernel="fir" if self.fir else "sde_vp",
                        use_nin_shortcut=True,
                    )
                )

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(pyramid_downsample(channels=input_pyramid_ch, out_channels=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        # mid
        self.mid = UNetMidBlock2D(
            in_channels=in_ch,
            temb_channels=4 * nf,
            output_scale_factor=math.sqrt(2.0),
            resnet_act_fn="silu",
            resnet_groups=min(in_ch // 4, 32),
            dropout=dropout,
        )

        in_ch = hs_c[-1]
        modules.append(
            ResnetBlock2D(
                in_channels=in_ch,
                temb_channels=4 * nf,
                output_scale_factor=np.sqrt(2.0),
                non_linearity="silu",
                groups=min(in_ch // 4, 32),
                groups_out=min(out_ch // 4, 32),
                overwrite_for_score_vde=True,
            )
        )
        modules.append(AttnBlock(channels=in_ch))
        modules.append(
            ResnetBlock2D(
                in_channels=in_ch,
                temb_channels=4 * nf,
                output_scale_factor=np.sqrt(2.0),
                non_linearity="silu",
                groups=min(in_ch // 4, 32),
                groups_out=min(out_ch // 4, 32),
                overwrite_for_score_vde=True,
            )
        )
        self.mid.resnets[0] = modules[len(modules) - 3]
        self.mid.attentions[0] = modules[len(modules) - 2]
        self.mid.resnets[1] = modules[len(modules) - 1]

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                in_ch = in_ch + hs_c.pop()
                modules.append(
                    ResnetBlock2D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        temb_channels=4 * nf,
                        output_scale_factor=np.sqrt(2.0),
                        non_linearity="silu",
                        groups=min(in_ch // 4, 32),
                        groups_out=min(out_ch // 4, 32),
                        overwrite_for_score_vde=True,
                    )
                )
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(nn.Conv2d(in_ch, in_ch, bias=True, kernel_size=3, padding=1))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(nn.Conv2d(in_ch, channels, bias=True, kernel_size=3, padding=1))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(channels=pyramid_ch, out_channels=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                modules.append(
                    ResnetBlock2D(
                        in_channels=in_ch,
                        temb_channels=4 * nf,
                        output_scale_factor=np.sqrt(2.0),
                        non_linearity="silu",
                        groups=min(in_ch // 4, 32),
                        groups_out=min(out_ch // 4, 32),
                        overwrite_for_score_vde=True,
                        up=True,
                        kernel="fir" if self.fir else "sde_vp",
                        use_nin_shortcut=True,
                    )
                )

        assert not hs_c

        if progressive != "output_skip":
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
            modules.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, sample, step_value, sigmas=None):
        timesteps = step_value
        x = sample
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = timesteps
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = timesteps
            used_sigmas = sigmas
            temb = get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        # If input data is in [0, 1]
        if not self.config.centered:
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        #        h = hs[-1]
        #        h = modules[m_idx](h, temb)
        #        m_idx += 1
        #        h = modules[m_idx](h)
        #        m_idx += 1
        #        h = modules[m_idx](h, temb)
        #        m_idx += 1

        h = self.mid(h, temb)
        m_idx += 3

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        if self.config.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h
