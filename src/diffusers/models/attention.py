import math

import torch
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
        num_head_channels=-1,
        num_groups=32,
        use_checkpoint=False,
        encoder_channels=None,
        use_new_attention_order=False,  # TODO(Patrick) -> is never used, maybe delete?
        overwrite_qkv=False,
        overwrite_linear=False,
        rescale_output_factor=1.0,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
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


# unet_score_estimation.py
# class AttnBlockpp(nn.Module):
#    """Channel-wise self-attention block. Modified from DDPM."""
#
#    def __init__(
#        self,
#        channels,
#        skip_rescale=False,
#        init_scale=0.0,
#        num_heads=1,
#        num_head_channels=-1,
#        use_checkpoint=False,
#        encoder_channels=None,
#        use_new_attention_order=False,  # TODO(Patrick) -> is never used, maybe delete?
#        overwrite_qkv=False,
#        overwrite_from_grad_tts=False,
#    ):
#        super().__init__()
#        num_groups = min(channels // 4, 32)
#        self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
#        self.NIN_0 = NIN(channels, channels)
#        self.NIN_1 = NIN(channels, channels)
#        self.NIN_2 = NIN(channels, channels)
#        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
#        self.skip_rescale = skip_rescale
#
#        self.channels = channels
#        if num_head_channels == -1:
#            self.num_heads = num_heads
#        else:
#            assert (
#                channels % num_head_channels == 0
#            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
#            self.num_heads = channels // num_head_channels
#
#        self.use_checkpoint = use_checkpoint
#        self.norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, eps=1e-6)
#        self.qkv = nn.Conv1d(channels, channels * 3, 1)
#        self.n_heads = self.num_heads
#
#        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
#
#        self.is_weight_set = False
#
#    def set_weights(self):
#        self.qkv.weight.data = torch.concat([self.NIN_0.W.data.T, self.NIN_1.W.data.T, self.NIN_2.W.data.T], dim=0)[:, :, None]
#        self.qkv.bias.data = torch.concat([self.NIN_0.b.data, self.NIN_1.b.data, self.NIN_2.b.data], dim=0)
#
#        self.proj_out.weight.data = self.NIN_3.W.data.T[:, :, None]
#        self.proj_out.bias.data = self.NIN_3.b.data
#
#        self.norm.weight.data = self.GroupNorm_0.weight.data
#        self.norm.bias.data = self.GroupNorm_0.bias.data
#
#    def forward(self, x):
#        if not self.is_weight_set:
#            self.set_weights()
#            self.is_weight_set = True
#
#        B, C, H, W = x.shape
#        h = self.GroupNorm_0(x)
#        q = self.NIN_0(h)
#        k = self.NIN_1(h)
#        v = self.NIN_2(h)
#
#        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
#        w = torch.reshape(w, (B, H, W, H * W))
#        w = F.softmax(w, dim=-1)
#        w = torch.reshape(w, (B, H, W, H, W))
#        h = torch.einsum("bhwij,bcij->bchw", w, v)
#        h = self.NIN_3(h)
#
#        if not self.skip_rescale:
#            result = x + h
#        else:
#            result = (x + h) / np.sqrt(2.0)
#
#        result = self.forward_2(x)
#
#        return result
#
#    def forward_2(self, x, encoder_out=None):
#        b, c, *spatial = x.shape
#        hid_states = self.norm(x).view(b, c, -1)
#
#        qkv = self.qkv(hid_states)
#        bs, width, length = qkv.shape
#        assert width % (3 * self.n_heads) == 0
#        ch = width // (3 * self.n_heads)
#        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
#
#        if encoder_out is not None:
#            encoder_kv = self.encoder_kv(encoder_out)
#            assert encoder_kv.shape[1] == self.n_heads * ch * 2
#            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
#            k = torch.cat([ek, k], dim=-1)
#            v = torch.cat([ev, v], dim=-1)
#
#        scale = 1 / math.sqrt(math.sqrt(ch))
#        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
#        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#
#        a = torch.einsum("bts,bcs->bct", weight, v)
#        h = a.reshape(bs, -1, length)
#
#        h = self.proj_out(h)
#        h = h.reshape(b, c, *spatial)
#
#        return (x + h) / np.sqrt(2.0)


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
