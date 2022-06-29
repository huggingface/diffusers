import torch

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin
from .attention import LinearAttention
from .embeddings import get_timestep_embedding
from .resnet import Downsample
from .resnet import ResnetBlock
from .resnet import Upsample


class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Rezero(torch.nn.Module):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1), torch.nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class UNetGradTTSModel(ModelMixin, ConfigMixin):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(UNetGradTTSModel, self).__init__()

        self.register_to_config(
            dim=dim,
            dim_mults=dim_mults,
            groups=groups,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            n_feats=n_feats,
            pe_scale=pe_scale,
        )

        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(), torch.nn.Linear(spk_emb_dim * 4, n_feats)
            )

        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(), torch.nn.Linear(dim * 4, dim))

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(in_channels=dim_in, out_channels=dim_out, temb_channels=dim, groups=8, pre_norm=False, eps=1e-5, non_linearity="mish", overwrite_for_grad_tts=True),
                        ResnetBlock(in_channels=dim_out, out_channels=dim_out, temb_channels=dim, groups=8, pre_norm=False, eps=1e-5, non_linearity="mish", overwrite_for_grad_tts=True),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out, use_conv=True, padding=1) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(in_channels=mid_dim, out_channels=mid_dim, temb_channels=dim, groups=8, pre_norm=False, eps=1e-5, non_linearity="mish", overwrite_for_grad_tts=True)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(in_channels=mid_dim, out_channels=mid_dim, temb_channels=dim, groups=8, pre_norm=False, eps=1e-5, non_linearity="mish", overwrite_for_grad_tts=True)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(in_channels=dim_out * 2, out_channels=dim_in, temb_channels=dim, groups=8, pre_norm=False, eps=1e-5, non_linearity="mish", overwrite_for_grad_tts=True),
                        ResnetBlock(in_channels=dim_in, out_channels=dim_in, temb_channels=dim, groups=8, pre_norm=False, eps=1e-5, non_linearity="mish", overwrite_for_grad_tts=True),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in, use_conv_transpose=True),
                    ]
                )
            )
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, timesteps, mu, mask, spk=None):
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = get_timestep_embedding(timesteps, self.dim, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)
