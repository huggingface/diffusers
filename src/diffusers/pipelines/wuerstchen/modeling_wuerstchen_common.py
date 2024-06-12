import torch
import torch.nn as nn

from ...models.attention_processor import Attention


class WuerstchenLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep):
        super().__init__()

        self.mapper = nn.Linear(c_timestep, c * 2)

    def forward(self, x, t):
        a, b = self.mapper(t)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + a) + b


class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()

        self.depthwise = nn.Conv2d(c + c_skip, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4), nn.GELU(), GlobalResponseNorm(c * 4), nn.Dropout(dropout), nn.Linear(c * 4, c)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)
        x = self.channelwise(x).permute(0, 3, 1, 2)
        return x + x_res


# from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
class GlobalResponseNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        agg_norm = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        stand_div_norm = agg_norm / (agg_norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * stand_div_norm) + self.beta + x


class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()

        self.self_attn = self_attn
        self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(query_dim=c, heads=nhead, dim_head=c // nhead, dropout=dropout, bias=True)
        self.kv_mapper = nn.Sequential(nn.SiLU(), nn.Linear(c_cond, c))

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        if self.self_attn:
            batch_size, channel, _, _ = x.shape
            kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2), kv], dim=1)
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        return x
