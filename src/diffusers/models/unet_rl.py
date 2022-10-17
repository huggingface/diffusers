# model adapted from diffuser https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.resnet import Downsample1D, ResidualTemporalBlock1D
from diffusers.models.unet_1d_blocks import get_down_block

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .embeddings import TimestepEmbedding, Timesteps


@dataclass
class ValueFunctionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch, horizon, 1)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class ValueFunction(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels=14,
        down_block_types: Tuple[str] = (
            "DownResnetBlock1D",
            "DownResnetBlock1D",
            "DownResnetBlock1D",
            "DownResnetBlock1D",
        ),
        block_out_channels: Tuple[int] = (32, 64, 128, 256),
        act_fn: str = "mish",
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
    ):
        super().__init__()
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(num_channels=block_out_channels[0], flip_sin_to_cos=False, downscale_freq_shift=1)
        self.time_mlp = TimestepEmbedding(
            channel=block_out_channels[0], time_embed_dim=time_embed_dim, act_fn="mish", out_dim=block_out_channels[0]
        )

        self.blocks = nn.ModuleList([])
        mid_dim = block_out_channels[-1]

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
                add_downsample=True,
            )
            self.blocks.append(down_block)

        ##
        self.mid_block1 = ResidualTemporalBlock1D(mid_dim, mid_dim // 2, embed_dim=block_out_channels[0])
        self.mid_down1 = Downsample1D(mid_dim // 2, use_conv=True)
        ##
        self.mid_block2 = ResidualTemporalBlock1D(mid_dim // 2, mid_dim // 4, embed_dim=block_out_channels[0])
        self.mid_down2 = Downsample1D(mid_dim // 4, use_conv=True)
        ##
        fc_dim = mid_dim // 4
        self.final_block = nn.ModuleList(
            [
                nn.Linear(fc_dim + block_out_channels[0], fc_dim // 2),
                nn.Mish(),
                nn.Linear(fc_dim // 2, 1),
            ]
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[ValueFunctionOutput, Tuple]:
        """r
        Args:
            sample (`torch.FloatTensor`): (batch, horizon, obs_dimension + action_dimension) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): batch (batch) timesteps
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_rl.ValueFunctionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_rl.ValueFunctionOutput`] or `tuple`: [`~models.unet_rl.ValueFunctionOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        sample = sample.permute(0, 2, 1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        t = self.time_proj(timesteps)
        t = self.time_mlp(t)
        down_block_res_samples = []

        # 2. down
        for downsample_block in self.blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=t)
            down_block_res_samples.append(res_samples[0])

        # 3. mid
        sample = self.mid_block1(sample, t)
        sample = self.mid_down1(sample)
        sample = self.mid_block2(sample, t)
        sample = self.mid_down2(sample)

        sample = sample.view(sample.shape[0], -1)
        sample = torch.cat((sample, t), dim=-1)
        for layer in self.final_block:
            sample = layer(sample)

        if not return_dict:
            return (sample,)

        return ValueFunctionOutput(sample=sample)
