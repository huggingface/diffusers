# model adapted from diffuser https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.resnet import ResidualTemporalBlock1D
from diffusers.models.unet_1d_blocks import DownResnetBlock1D, UpResnetBlock1D, Downsample1D

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .embeddings import TimestepEmbedding, Timesteps
from .resnet import rearrange_dims


@dataclass
class ValueFunctionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch, horizon, obs_dimension)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class ValueFunction(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        transition_dim=14,
        dim=32,
        dim_mults=(1, 4, 8),
    ):
        super().__init__()

        self.transition_dim = transition_dim
        self.time_proj = Timesteps(num_channels=dim, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.time_mlp = TimestepEmbedding(channel=dim, time_embed_dim=4 * dim, act_fn="mish", out_dim=dim)

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                DownResnetBlock1D(
                    in_channels=dim_in, out_channels=dim_out, temb_channels=dim, add_downsample=True
                )
            )

    
        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock1D(mid_dim, mid_dim_2, embed_dim=dim)
        self.mid_down1 = Downsample1D(mid_dim_2, use_conv=True)
        ##
        self.mid_block2 = ResidualTemporalBlock1D(mid_dim_2, mid_dim_3, embed_dim=dim)
        self.mid_down2 = Downsample1D(mid_dim_3, use_conv=True)
        ##
        fc_dim = mid_dim_3
        self.final_block = nn.ModuleList([
            nn.Linear(fc_dim + dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, 1),]
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
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
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
