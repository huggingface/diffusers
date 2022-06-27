# model adapted from diffuser https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py

import math

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin


# try:
#    import einops
#    from einops.layers.torch import Rearrange
# except:
#    print("Einops is not installed")
#    pass


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


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
            #            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            RearrangeDim(),
            #            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            RearrangeDim(),
            #            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUNet(ModelMixin, ConfigMixin):  # (nn.Module):
    def __init__(
        self,
        training_horizon,
        transition_dim,
        cond_dim,
        predict_epsilon=False,
        clip_denoised=True,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        self.transition_dim = transition_dim
        self.cond_dim = cond_dim
        self.predict_epsilon = predict_epsilon
        self.clip_denoised = clip_denoised

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=training_horizon),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=training_horizon),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                training_horizon = training_horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=training_horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=training_horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=training_horizon),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=training_horizon),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                training_horizon = training_horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        """
        x : [ batch x horizon x transition ]
        """

        #        x = einops.rearrange(x, "b h t -> b t h")
        x = x.permute(0, 2, 1)

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        #        x = einops.rearrange(x, "b t h -> b h t")
        x = x.permute(0, 2, 1)
        return x


class TemporalValue(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        print(in_out)
        for dim_in, dim_out in in_out:
            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                        ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                        Downsample1d(dim_out),
                    ]
                )
            )

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        """
        x : [ batch x horizon x transition ]
        """

        #        x = einops.rearrange(x, "b h t -> b t h")
        x = x.permute(0, 2, 1)

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
