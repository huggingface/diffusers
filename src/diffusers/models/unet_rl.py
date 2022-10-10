# model adapted from diffuser https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.resnet import ResidualTemporalBlock
from diffusers.models.unet_blocks import DownResnetBlock1D, UpResnetBlock1D

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .embeddings import TimestepEmbedding, Timesteps
from .resnet import rearrange_dims


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
    A UNet for multi-dimensional temporal data. This model takes the batch over the `training_horizon`.

    Parameters:
        transition_dim: state-dimension of samples to predict over
        dim: embedding dimension of model
        dim_mults: dimension multiples of the up/down blocks
    """

    @register_to_config
    def __init__(
        self,
        transition_dim=14,
        dim=32,
        dim_mults=(1, 4, 8),
    ):
        super().__init__()

        self.transition_dim = transition_dim

        # time
        self.time_proj = Timesteps(num_channels=dim, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.time_mlp = TimestepEmbedding(channel=dim, time_embed_dim=4 * dim, act_fn="mish", out_dim=dim)

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        # down
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.down_blocks.append(
                DownResnetBlock1D(
                    in_channels=dim_in, out_channels=dim_out, temb_channels=dim, add_downsample=(not is_last)
                )
            )

        # mid
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim)

        # up
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.up_blocks.append(
                UpResnetBlock1D(
                    in_channels=dim_out * 2, out_channels=dim_in, temb_channels=dim, add_upsample=(not is_last)
                )
            )

        # out
        self.final_conv1d_1 = nn.Conv1d(dim, dim, 5, padding=2)
        self.final_conv1d_gn = nn.GroupNorm(8, dim)
        self.final_conv1d_act = nn.Mish()
        self.final_conv1d_2 = nn.Conv1d(dim, transition_dim, 1)

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
        sample = sample.permute(0, 2, 1)

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
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=temb)
            down_block_res_samples.append(res_samples[0])

        # 3. mid
        sample = self.mid_block1(sample, temb)
        sample = self.mid_block2(sample, temb)

        # 4. up
        for up_block in self.up_blocks:
            sample = up_block(hidden_states=sample, res_hidden_states=down_block_res_samples.pop(), temb=temb)

        # 5. post-process
        sample = self.final_conv1d_1(sample)
        sample = rearrange_dims(sample)
        sample = self.final_conv1d_gn(sample)
        sample = rearrange_dims(sample)
        sample = self.final_conv1d_act(sample)
        sample = self.final_conv1d_2(sample)

        sample = sample.permute(0, 2, 1)

        if not return_dict:
            return (sample,)

        return UNet1DOutput(sample=sample)
