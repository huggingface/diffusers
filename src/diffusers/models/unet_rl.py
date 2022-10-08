# model adapted from diffuser https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.resnet import Downsample1D, ResidualTemporalBlock, Upsample1D

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .embeddings import get_timestep_embedding, Timesteps, TimestepEmbedding
from .resnet import rearrange_dims

@dataclass
class TemporalUNetOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch, horizon, obs_dimension)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor



class RearrangeDim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        if len(tensor.shape) == 2:
            return tensor[:, :, None]
        if len(tensor.shape) == 3:
            return tensor[:, :, None, :]
        elif len(tensor.shape) == 4:
            return tensor[:, :, 0, :]
        else:
            raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            RearrangeDim(),
            nn.GroupNorm(n_groups, out_channels),
            RearrangeDim(),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class TemporalUNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        training_horizon=128,
        transition_dim=14,
        cond_dim=3,
        predict_epsilon=False,
        clip_denoised=True,
        dim=32,
        dim_mults=(1, 4, 8),
    ):
        super().__init__()

        self.transition_dim = transition_dim
        self.cond_dim = cond_dim
        self.predict_epsilon = predict_epsilon
        self.clip_denoised = clip_denoised

        self.time_proj = Timesteps(num_channels=dim, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.time_mlp = TimestepEmbedding(channel=dim, time_embed_dim=4*dim, act_fn="mish", out_dim=dim)

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim=dim, horizon=training_horizon),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim=dim, horizon=training_horizon),
                        Downsample1D(dim_out, use_conv=True) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                training_horizon = training_horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim, horizon=training_horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim, horizon=training_horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=dim, horizon=training_horizon),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim=dim, horizon=training_horizon),
                        Upsample1D(dim_in, use_conv_transpose=True) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                training_horizon = training_horizon * 2

        self.final_conv1d_1 = nn.Conv1d(dim, dim, 5, padding=2)
        self.final_conv1d_gn = nn.GroupNorm(8, dim)
        self.final_conv1d_act = nn.Mish()
        self.final_conv1d_2 = nn.Conv1d(dim, transition_dim, 1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[TemporalUNetOutput, Tuple]:
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
        h = []

        # 2. down
        for resnet, resnet2, downsample in self.downs:
            sample = resnet(sample, t)
            sample = resnet2(sample, t)
            h.append(sample)
            sample = downsample(sample)

        # 3. mid
        sample = self.mid_block1(sample, t)
        sample = self.mid_block2(sample, t)

        # 4. up
        for resnet, resnet2, upsample in self.ups:
            sample = torch.cat((sample, h.pop()), dim=1)
            sample = resnet(sample, t)
            sample = resnet2(sample, t)
            sample = upsample(sample)

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

        return TemporalUNetOutput(sample=sample)
