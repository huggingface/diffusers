import math
from inspect import isfunction

import torch
import torch.nn.functional as F
from torch import nn


# unet_grad_tts.py
# TODO(Patrick) - weird linear attention layer. Check with: https://github.com/huawei-noah/Speech-Backbones/issues/15
class LinearAttention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = (
            qkv.reshape(b, 3, self.heads, self.dim_head, h, w)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(3, b, self.heads, self.dim_head, -1)
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(b, self.heads, self.dim_head, h, w).reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)


# the main attention block that is used for all models
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=None,
        num_groups=32,
        encoder_channels=None,
        overwrite_qkv=False,
        overwrite_linear=False,
        rescale_output_factor=1.0,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, eps=1e-5, affine=True)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.n_heads = self.num_heads
        self.rescale_output_factor = rescale_output_factor

        if encoder_channels is not None:
            self.encoder_kv = nn.Conv1d(encoder_channels, channels * 2, 1)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

        self.overwrite_qkv = overwrite_qkv
        if overwrite_qkv:
            in_channels = channels
            self.norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, eps=1e-6)
            self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.overwrite_linear = overwrite_linear
        if self.overwrite_linear:
            num_groups = min(channels // 4, 32)
            self.norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, eps=1e-6)
            self.NIN_0 = NIN(channels, channels)
            self.NIN_1 = NIN(channels, channels)
            self.NIN_2 = NIN(channels, channels)
            self.NIN_3 = NIN(channels, channels)

            self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)

        self.is_overwritten = False

    def set_weights(self, module):
        if self.overwrite_qkv:
            qkv_weight = torch.cat([module.q.weight.data, module.k.weight.data, module.v.weight.data], dim=0)[
                :, :, :, 0
            ]
            qkv_bias = torch.cat([module.q.bias.data, module.k.bias.data, module.v.bias.data], dim=0)

            self.qkv.weight.data = qkv_weight
            self.qkv.bias.data = qkv_bias

            proj_out = zero_module(nn.Conv1d(self.channels, self.channels, 1))
            proj_out.weight.data = module.proj_out.weight.data[:, :, :, 0]
            proj_out.bias.data = module.proj_out.bias.data

            self.proj_out = proj_out
        elif self.overwrite_linear:
            self.qkv.weight.data = torch.concat(
                [self.NIN_0.W.data.T, self.NIN_1.W.data.T, self.NIN_2.W.data.T], dim=0
            )[:, :, None]
            self.qkv.bias.data = torch.concat([self.NIN_0.b.data, self.NIN_1.b.data, self.NIN_2.b.data], dim=0)

            self.proj_out.weight.data = self.NIN_3.W.data.T[:, :, None]
            self.proj_out.bias.data = self.NIN_3.b.data

            self.norm.weight.data = self.GroupNorm_0.weight.data
            self.norm.bias.data = self.GroupNorm_0.bias.data

    def forward(self, x, encoder_out=None):
        if (self.overwrite_qkv or self.overwrite_linear) and not self.is_overwritten:
            self.set_weights(self)
            self.is_overwritten = True

        b, c, *spatial = x.shape
        hid_states = self.norm(x).view(b, c, -1)

        qkv = self.qkv(hid_states)
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if encoder_out is not None:
            encoder_kv = self.encoder_kv(encoder_out)
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum("bts,bcs->bct", weight, v)
        h = a.reshape(bs, -1, length)

        h = self.proj_out(h)
        h = h.reshape(b, c, *spatial)

        result = x + h

        result = result / self.rescale_output_factor

        return result


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + x_in


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, x, context=None, mask=None):
        batch_size, sequence_length, dim = x.shape

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = mask.reshape(batch_size, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = self.reshape_batch_dim_to_heads(out)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


# TODO(Patrick) - this can and should be removed
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# TODO(Patrick) - remove once all weights have been converted -> not needed anymore then
class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(in_dim, num_units), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
