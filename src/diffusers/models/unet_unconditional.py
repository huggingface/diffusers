import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin
from .attention import AttentionBlock
from .embeddings import get_timestep_embedding
from .resnet import Downsample2D, ResnetBlock2D, Upsample2D
from .unet_new import (
    UNetMidBlock2D,
    UNetResAttnDownBlock2D,
    UNetResAttnUpBlock2D,
    UNetResDownBlock2D,
    UNetResUpBlock2D,
)


class UNetUnconditionalModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding. :param in_channels: channels in the input Tensor. :param
    model_channels: base channel count for the model. :param out_channels: channels in the output Tensor. :param
    num_res_blocks: number of residual blocks per downsample. :param attention_resolutions: a collection of downsample
    rates at which
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
    for up/downsampling. :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def init_for_ldm(
        self,
        in_channels,
        model_channels,
        channel_mult,
        num_res_blocks,
        dropout,
        time_embed_dim,
        attention_resolutions,
        num_head_channels,
        num_heads,
        legacy,
        use_spatial_transformer,
        transformer_depth,
        context_dim,
        conv_resample,
        out_channels,
    ):
        # TODO(PVP) - delete after weight conversion

        class TimestepEmbedSequential(nn.Sequential):
            """
            A sequential module that passes timestep embeddings to the children that support it as an extra input.
            """

            pass

        # TODO(PVP) - delete after weight conversion
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

        dims = 2
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock2D(
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        temb_channels=time_embed_dim,
                        eps=1e-5,
                        non_linearity="silu",
                        overwrite_for_ldm=True,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                        ),
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample2D(ch, use_conv=conv_resample, out_channels=out_ch, padding=1, name="op")
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = num_head_channels

        if dim_head < 0:
            dim_head = None

        # TODO(Patrick) - delete after weight conversion
        # init to be able to overwrite `self.mid`
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock2D(
                in_channels=ch,
                out_channels=None,
                dropout=dropout,
                temb_channels=time_embed_dim,
                eps=1e-5,
                non_linearity="silu",
                overwrite_for_ldm=True,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
            ),
            ResnetBlock2D(
                in_channels=ch,
                out_channels=None,
                dropout=dropout,
                temb_channels=time_embed_dim,
                eps=1e-5,
                non_linearity="silu",
                overwrite_for_ldm=True,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResnetBlock2D(
                        in_channels=ch + ich,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        temb_channels=time_embed_dim,
                        eps=1e-5,
                        non_linearity="silu",
                        overwrite_for_ldm=True,
                    ),
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=-1,
                            num_head_channels=dim_head,
                        ),
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample2D(ch, use_conv=conv_resample, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # ================ SET WEIGHTS OF ALL WEIGHTS ==================
        for i, input_layer in enumerate(self.input_blocks[1:]):
            block_id = i // (num_res_blocks + 1)
            layer_in_block_id = i % (num_res_blocks + 1)

            if layer_in_block_id == 2:
                self.downsample_blocks[block_id].downsamplers[0].op.weight.data = input_layer[0].op.weight.data
                self.downsample_blocks[block_id].downsamplers[0].op.bias.data = input_layer[0].op.bias.data
            elif len(input_layer) > 1:
                self.downsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                self.downsample_blocks[block_id].attentions[layer_in_block_id].set_weight(input_layer[1])
            else:
                self.downsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])

        self.mid.resnets[0].set_weight(self.middle_block[0])
        self.mid.resnets[1].set_weight(self.middle_block[2])
        self.mid.attentions[0].set_weight(self.middle_block[1])

        for i, input_layer in enumerate(self.output_blocks):
            block_id = i // (num_res_blocks + 1)
            layer_in_block_id = i % (num_res_blocks + 1)

            if len(input_layer) > 2:
                self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                self.upsample_blocks[block_id].attentions[layer_in_block_id].set_weight(input_layer[1])
                self.upsample_blocks[block_id].upsamplers[0].conv.weight.data = input_layer[2].conv.weight.data
                self.upsample_blocks[block_id].upsamplers[0].conv.bias.data = input_layer[2].conv.bias.data
            elif len(input_layer) > 1 and "Upsample2D" in input_layer[1].__class__.__name__:
                self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                self.upsample_blocks[block_id].upsamplers[0].conv.weight.data = input_layer[1].conv.weight.data
                self.upsample_blocks[block_id].upsamplers[0].conv.bias.data = input_layer[1].conv.bias.data
            elif len(input_layer) > 1:
                self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                self.upsample_blocks[block_id].attentions[layer_in_block_id].set_weight(input_layer[1])
            else:
                self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])

        self.conv_in.weight.data = self.input_blocks[0][0].weight.data
        self.conv_in.bias.data = self.input_blocks[0][0].bias.data

    def __init__(
        self,
        image_size,
        in_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        resnet_input_channels=(224, 224, 448, 672),
        resnet_output_channels=(224, 448, 672, 896),
        conv_resample=True,
        num_head_channels=32,
    ):
        super().__init__()

        # register all __init__ params with self.register
        self.register_to_config(
            image_size=image_size,
            in_channels=in_channels,
            resnet_input_channels=resnet_input_channels,
            resnet_output_channels=resnet_output_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            conv_resample=conv_resample,
            num_head_channels=num_head_channels,
        )

        # To delete - replace with config values
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout

        time_embed_dim = resnet_input_channels[0] * 4

        # ======================== Input ===================
        self.conv_in = nn.Conv2d(in_channels, resnet_input_channels[0], kernel_size=3, padding=(1, 1))

        # ======================== Time ====================
        self.time_embed = nn.Sequential(
            nn.Linear(resnet_input_channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ======================== Down ====================
        input_channels = list(resnet_input_channels)
        output_channels = list(resnet_output_channels)

        ds_new = 1
        self.downsample_blocks = nn.ModuleList([])
        for i, (input_channel, output_channel) in enumerate(zip(input_channels, output_channels)):
            is_final_block = i == len(input_channels) - 1

            if ds_new in attention_resolutions:
                down_block = UNetResAttnDownBlock2D(
                    num_layers=num_res_blocks,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=1e-5,
                    resnet_act_fn="silu",
                    attn_num_head_channels=num_head_channels,
                )
            else:
                down_block = UNetResDownBlock2D(
                    num_layers=num_res_blocks,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=1e-5,
                    resnet_act_fn="silu",
                )

            self.downsample_blocks.append(down_block)

            ds_new *= 2

        ds_new = ds_new / 2

        # ======================== Mid ====================
        self.mid = UNetMidBlock2D(
            in_channels=output_channels[-1],
            dropout=dropout,
            temb_channels=time_embed_dim,
            resnet_eps=1e-5,
            resnet_act_fn="silu",
            resnet_time_scale_shift="default",
            attn_num_head_channels=num_head_channels,
        )

        self.upsample_blocks = nn.ModuleList([])
        for i, (input_channel, output_channel) in enumerate(zip(reversed(input_channels), reversed(output_channels))):
            is_final_block = i == len(input_channels) - 1

            if ds_new in attention_resolutions:
                up_block = UNetResAttnUpBlock2D(
                    num_layers=num_res_blocks + 1,
                    in_channels=output_channel,
                    next_channels=input_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    resnet_eps=1e-5,
                    resnet_act_fn="silu",
                    attn_num_head_channels=num_head_channels,
                )
            else:
                up_block = UNetResUpBlock2D(
                    num_layers=num_res_blocks + 1,
                    in_channels=output_channel,
                    next_channels=input_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    resnet_eps=1e-5,
                    resnet_act_fn="silu",
                )

            self.upsample_blocks.append(up_block)

            ds_new /= 2

        # ======================== Out ====================
        self.out = nn.Sequential(
            nn.GroupNorm(num_channels=output_channels[0], num_groups=32, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(resnet_input_channels[0], out_channels, 3, padding=1),
        )

        # =========== TO DELETE AFTER CONVERSION ==========
        transformer_depth = 1
        context_dim = None
        legacy = True
        num_heads = -1
        model_channels = resnet_input_channels[0]
        channel_mult = tuple([x // model_channels for x in resnet_output_channels])
        self.init_for_ldm(
            in_channels,
            model_channels,
            channel_mult,
            num_res_blocks,
            dropout,
            time_embed_dim,
            attention_resolutions,
            num_head_channels,
            num_heads,
            legacy,
            False,
            transformer_depth,
            context_dim,
            conv_resample,
            out_channels,
        )

    def forward(self, sample, timesteps=None):
        # 1. time step embeddings
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        t_emb = get_timestep_embedding(
            timesteps, self.config.resnet_input_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0
        )
        emb = self.time_embed(t_emb)

        # 2. pre-process sample
        #        sample = sample.type(self.dtype_)
        sample = self.conv_in(sample)

        # 3. down blocks
        down_block_res_samples = (sample,)
        for downsample_block in self.downsample_blocks:
            sample, res_samples = downsample_block(sample, emb)

            # append to tuple
            down_block_res_samples += res_samples

        # 4. mid block
        sample = self.mid(sample, emb)

        # 5. up blocks
        for upsample_block in self.upsample_blocks:

            # pop from tuple
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            sample = upsample_block(sample, res_samples, emb)

        # 6. post-process sample
        sample = self.out(sample)

        return sample
