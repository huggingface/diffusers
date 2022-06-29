from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin
from .attention import AttentionBlock
from .embeddings import get_timestep_embedding
from .resnet import Downsample, ResBlock, TimestepBlock, Upsample


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


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


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.swish = swish

    def forward(self, x):
        y = super().forward(x.float()).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y


def normalization(channels, swish=0.0):
    """
    Make a standard normalization layer, with an optional swish activation.

    :param channels: number of input channels. :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=32, swish=swish)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# class TimestepBlock(nn.Module):
#    """
# Any module where forward() takes timestep embeddings as a second argument. #"""
#
#    @abstractmethod
#    def forward(self, x, emb):
#        """
# Apply the module to `x` given `emb` timestep embeddings. #"""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x


# class ResBlock(TimestepBlock):
#    """
# A residual block that can optionally change the number of channels. # # :param channels: the number of input
channels. :param emb_channels: the number of timestep embedding channels. # :param dropout: the rate of dropout. :param
out_channels: if specified, the number of out channels. :param # use_conv: if True and out_channels is specified, use a
spatial # convolution instead of a smaller 1x1 convolution to change the channels in the skip connection. # :param
dims: determines if the signal is 1D, 2D, or 3D. :param use_checkpoint: if True, use gradient checkpointing # on this
module. :param up: if True, use this block for upsampling. :param down: if True, use this block for # downsampling. #"""
#
#    def __init__(
#        self,
#        channels,
#        emb_channels,
#        dropout,
#        out_channels=None,
#        use_conv=False,
#        use_scale_shift_norm=False,
#        dims=2,
#        use_checkpoint=False,
#        up=False,
#        down=False,
#    ):
#        super().__init__()
#        self.channels = channels
#        self.emb_channels = emb_channels
#        self.dropout = dropout
#        self.out_channels = out_channels or channels
#        self.use_conv = use_conv
#        self.use_checkpoint = use_checkpoint
#        self.use_scale_shift_norm = use_scale_shift_norm
#
#        self.in_layers = nn.Sequential(
#            normalization(channels, swish=1.0),
#            nn.Identity(),
#            conv_nd(dims, channels, self.out_channels, 3, padding=1),
#        )
#
#        self.updown = up or down
#
#        if up:
#            self.h_upd = Upsample(channels, use_conv=False, dims=dims)
#            self.x_upd = Upsample(channels, use_conv=False, dims=dims)
#        elif down:
#            self.h_upd = Downsample(channels, use_conv=False, dims=dims, padding=1, name="op")
#            self.x_upd = Downsample(channels, use_conv=False, dims=dims, padding=1, name="op")
#        else:
#            self.h_upd = self.x_upd = nn.Identity()
#
#        self.emb_layers = nn.Sequential(
#            nn.SiLU(),
#            linear(
#                emb_channels,
#                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
#            ),
#        )
#        self.out_layers = nn.Sequential(
#            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
#            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
#            nn.Dropout(p=dropout),
#            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
#        )
#
#        if self.out_channels == channels:
#            self.skip_connection = nn.Identity()
#        elif use_conv:
#            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
#        else:
#            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
#
#    def forward(self, x, emb):
#        """
# Apply the block to a Tensor, conditioned on a timestep embedding. # # :param x: an [N x C x ...] Tensor of features.
:param emb: an [N x emb_channels] Tensor of timestep embeddings. # :return: an [N x C x ...] Tensor of outputs. #"""
#        if self.updown:
#            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#            h = in_rest(x)
#            h = self.h_upd(h)
#            x = self.x_upd(x)
#            h = in_conv(h)
#        else:
#            h = self.in_layers(x)
#        emb_out = self.emb_layers(emb).type(h.dtype)
#        while len(emb_out.shape) < len(h.shape):
#            emb_out = emb_out[..., None]
#        if self.use_scale_shift_norm:
#            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
#            scale, shift = torch.chunk(emb_out, 2, dim=1)
#            h = out_norm(h) * (1 + scale) + shift
#            h = out_rest(h)
#        else:
#            h = h + emb_out
#            h = self.out_layers(h)
#        return self.skip_connection(x) + h


class GlideUNetModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor. :param model_channels: base channel count for the model. :param
    out_channels: channels in the output Tensor. :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple. For example, if this contains 4, then at 4x
        downsampling, attention will be used.
    :param dropout: the dropout probability. :param channel_mult: channel multiplier for each level of the UNet. :param
    conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D. :param num_classes: if specified (as an int), then this
    model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage. :param num_heads: the number of attention
    heads in each attention layer. :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism. :param resblock_updown: use residual blocks
    for up/downsampling.
    """

    def __init__(
        self,
        in_channels=3,
        resolution=64,
        model_channels=192,
        out_channels=6,
        num_res_blocks=3,
        attention_resolutions=(2, 4, 8),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        transformer_dim=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.resolution = resolution
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        # self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=transformer_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, use_conv=conv_resample, dims=dims, out_channels=out_ch, padding=1, name="op"
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=transformer_dim,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=transformer_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, use_conv=conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs. :param timesteps: a 1-D batch of timesteps. :param y: an [N]
        Tensor of labels, if class-conditional. :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(
            get_timestep_embedding(timesteps, self.model_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        )

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class GlideTextToImageUNetModel(GlideUNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(
        self,
        in_channels=3,
        resolution=64,
        model_channels=192,
        out_channels=6,
        num_res_blocks=3,
        attention_resolutions=(2, 4, 8),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        transformer_dim=512,
    ):
        super().__init__(
            in_channels=in_channels,
            resolution=resolution,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            transformer_dim=transformer_dim,
        )
        self.register_to_config(
            in_channels=in_channels,
            resolution=resolution,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            transformer_dim=transformer_dim,
        )

        self.transformer_proj = nn.Linear(transformer_dim, self.model_channels * 4)

    def forward(self, x, timesteps, transformer_out=None):
        hs = []
        emb = self.time_embed(
            get_timestep_embedding(timesteps, self.model_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        )

        # project the last token
        transformer_proj = self.transformer_proj(transformer_out[:, -1])
        transformer_out = transformer_out.permute(0, 2, 1)  # NLC -> NCL

        emb = emb + transformer_proj.to(emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, transformer_out)
            hs.append(h)
        h = self.middle_block(h, emb, transformer_out)
        for module in self.output_blocks:
            other = hs.pop()
            h = torch.cat([h, other], dim=1)
            h = module(h, emb, transformer_out)
        return self.out(h)


class GlideSuperResUNetModel(GlideUNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(
        self,
        in_channels=3,
        resolution=256,
        model_channels=192,
        out_channels=6,
        num_res_blocks=3,
        attention_resolutions=(2, 4, 8),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
    ):
        super().__init__(
            in_channels=in_channels,
            resolution=resolution,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
        )
        self.register_to_config(
            in_channels=in_channels,
            resolution=resolution,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
        )

    def forward(self, x, timesteps, low_res=None):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)

        hs = []
        emb = self.time_embed(
            get_timestep_embedding(timesteps, self.model_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        )

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)
