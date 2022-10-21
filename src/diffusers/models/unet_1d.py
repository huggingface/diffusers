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
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .embeddings import TimestepEmbedding, Timesteps


@dataclass
class UNet1DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch, horizon, obs_dimension)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class UNet1DModel(ModelMixin, ConfigMixin):
    """
    UNet1DModel is a 1D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels:
        out_channels:
        down_block_types:
        up_block_types:
        block_out_channels:
        act_fn:
        norm_num_groups:
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 14,
        out_channels: int = 14,
        down_block_types: Tuple[str] = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
        up_block_types: Tuple[str] = ("UpResnetBlock1D", "UpResnetBlock1D"),
        mid_block_types: Tuple[str] = ("MidResTemporalBlock1D", "MidResTemporalBlock1D"),
        out_block_type: str = "OutConv1DBlock",
        block_out_channels: Tuple[int] = (32, 128, 256),
        act_fn: str = "mish",
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        always_downsample: bool = False,
    ):
        super().__init__()

        time_embed_dim = block_out_channels[0] * 4

        # time
        self.time_proj = Timesteps(num_channels=block_out_channels[0], flip_sin_to_cos=False, downscale_freq_shift=1)
        self.time_mlp = TimestepEmbedding(
            channel=block_out_channels[0], time_embed_dim=time_embed_dim, act_fn=act_fn, out_dim=block_out_channels[0]
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        mid_dim = block_out_channels[-1]

        # down
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block_type = down_block_types[i]
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or always_downsample,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_blocks = nn.ModuleList([])
        for i, mid_block_type in enumerate(mid_block_types):
            if always_downsample:
                mid_block = get_mid_block(
                    mid_block_type,
                    in_channels=mid_dim // (i + 1),
                    out_channels=mid_dim // ((i + 1) * 2),
                    embed_dim=block_out_channels[0],
                    add_downsample=True,
                )
            else:
                mid_block = get_mid_block(
                    mid_block_type,
                    in_channels=mid_dim,
                    out_channels=mid_dim,
                    embed_dim=block_out_channels[0],
                    add_downsample=False,
                )
            self.mid_blocks.append(mid_block)
        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        for i, up_block_type in enumerate(up_block_types):
            input_channel = reversed_block_out_channels[i]
            output_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=mid_dim // 4,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, horizon, obs_dimension + action_dimension) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): batch (batch) timesteps
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        temb = self.time_proj(timesteps)
        temb = self.time_mlp(temb)
        down_block_res_samples = []

        # 2. down
        for down_block in self.down_blocks:
            sample, res_samples = down_block(hidden_states=sample, temb=temb)
            down_block_res_samples.append(res_samples[0])

        # 3. mid
        for mid_block in self.mid_blocks:
            sample = mid_block(sample, temb)

        # 4. up
        for up_block in self.up_blocks:
            sample = up_block(hidden_states=sample, res_hidden_states=down_block_res_samples.pop(), temb=temb)

        # 5. post-process
        sample = self.out_block(sample, temb)

        if not return_dict:
            return (sample,)

        return UNet1DOutput(sample=sample)
