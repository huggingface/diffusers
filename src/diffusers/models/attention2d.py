import math

import torch
import torch.nn.functional as F
from torch import nn


# unet_grad_tts.py
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
        #        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        q, k, v = (
            qkv.reshape(b, 3, self.heads, self.dim_head, h, w)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(3, b, self.heads, self.dim_head, -1)
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        #        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        out = out.reshape(b, self.heads, self.dim_head, h, w).reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)


# unet.py
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalization(in_channels, swish=None, eps=1e-6)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        print("x", x.abs().sum())
        h_ = x
        h_ = self.norm(h_)

        print("hid_states shape", h_.shape)
        print("hid_states", h_.abs().sum())
        print("hid_states - 3 - 3", h_.view(h_.shape[0], h_.shape[1], -1)[:, :3, -3:])

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        print(self.q)
        print("q_shape", q.shape)
        print("q", q.abs().sum())
#        print("k_shape", k.shape)
#        print("k", k.abs().sum())
#        print("v_shape", v.shape)
#        print("v", v.abs().sum())

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw

        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)

        print("weight", w_.abs().sum())

        # attend to values
        v = v.reshape(b, c, h * w)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


# unet_glide.py & unet_ldm.py
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
        use_checkpoint=False,
        encoder_channels=None,
        use_new_attention_order=False,  # TODO(Patrick) -> is never used, maybe delete?
        overwrite_qkv=False,
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
        self.norm = normalization(channels, swish=0.0)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.n_heads = self.num_heads

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        self.overwrite_qkv = overwrite_qkv
        if overwrite_qkv:
            in_channels = channels
            self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.is_overwritten = False

    def set_weights(self, module):
        if self.overwrite_qkv:
            qkv_weight = torch.cat([module.q.weight.data, module.k.weight.data, module.v.weight.data], dim=0)[:, :, :, 0]
            qkv_bias = torch.cat([module.q.bias.data, module.k.bias.data, module.v.bias.data], dim=0)

            self.qkv.weight.data = qkv_weight
            self.qkv.bias.data = qkv_bias

            proj_out = zero_module(conv_nd(1, self.channels, self.channels, 1))
            proj_out.weight.data = module.proj_out.weight.data[:, :, :, 0]
            proj_out.bias.data = module.proj_out.bias.data

            self.proj_out = proj_out

    def forward(self, x, encoder_out=None):
        if self.overwrite_qkv and not self.is_overwritten:
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

        return x + h.reshape(b, c, *spatial)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)
        self.swish = swish

    def forward(self, x):
        y = super().forward(x.float()).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y


def normalization(channels, swish=0.0, eps=1e-5):
    """
    Make a standard normalization layer, with an optional swish activation.

    :param channels: number of input channels. :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=32, swish=swish, eps=eps, affine=True)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# unet_score_estimation.py
# class AttnBlockpp(nn.Module):
#    """Channel-wise self-attention block. Modified from DDPM."""
#
#    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
#        super().__init__()
#        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
#        self.NIN_0 = NIN(channels, channels)
#        self.NIN_1 = NIN(channels, channels)
#        self.NIN_2 = NIN(channels, channels)
#        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
#        self.skip_rescale = skip_rescale
#
#    def forward(self, x):
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
#        if not self.skip_rescale:
#            return x + h
#        else:
#            return (x + h) / np.sqrt(2.0)
