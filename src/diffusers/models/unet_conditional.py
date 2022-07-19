from typing import Dict, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin
from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import UNetMidBlock2DCrossAttn, get_down_block, get_up_block


class UNetConditionalModel(ModelMixin, ConfigMixin):
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

    def __init__(
        self,
        image_size=None,
        in_channels=4,
        out_channels=4,
        num_res_blocks=2,
        dropout=0,
        block_channels=(320, 640, 1280, 1280),
        down_blocks=(
            "UNetResCrossAttnDownBlock2D",
            "UNetResCrossAttnDownBlock2D",
            "UNetResCrossAttnDownBlock2D",
            "UNetResDownBlock2D",
        ),
        downsample_padding=1,
        up_blocks=(
            "UNetResUpBlock2D",
            "UNetResCrossAttnUpBlock2D",
            "UNetResCrossAttnUpBlock2D",
            "UNetResCrossAttnUpBlock2D",
        ),
        resnet_act_fn="silu",
        resnet_eps=1e-5,
        conv_resample=True,
        num_head_channels=8,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        mid_block_scale_factor=1,
        center_input_sample=False,
        resnet_num_groups=30,
        **kwargs,
    ):
        super().__init__()
        # remove automatically added kwargs
        for arg in self._automatically_saved_args:
            kwargs.pop(arg, None)

        if len(kwargs) > 0:
            raise ValueError(
                f"The following keyword arguments do not exist for {self.__class__}: {','.join(kwargs.keys())}"
            )

        # register all __init__ params to be accessible via `self.config.<...>`
        # should probably be automated down the road as this is pure boiler plate code
        self.register_to_config(
            image_size=image_size,
            in_channels=in_channels,
            block_channels=block_channels,
            downsample_padding=downsample_padding,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            down_blocks=down_blocks,
            up_blocks=up_blocks,
            dropout=dropout,
            resnet_eps=resnet_eps,
            conv_resample=conv_resample,
            num_head_channels=num_head_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            mid_block_scale_factor=mid_block_scale_factor,
            resnet_num_groups=resnet_num_groups,
            center_input_sample=center_input_sample,
        )

        self.image_size = image_size
        time_embed_dim = block_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_steps = Timesteps(block_channels[0], flip_sin_to_cos, downscale_freq_shift)
        timestep_input_dim = block_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.downsample_blocks = nn.ModuleList([])
        self.mid = None
        self.upsample_blocks = nn.ModuleList([])

        # down
        output_channel = block_channels[0]
        for i, down_block_type in enumerate(down_blocks):
            input_channel = output_channel
            output_channel = block_channels[i]
            is_final_block = i == len(block_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=num_res_blocks,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                attn_num_head_channels=num_head_channels,
                downsample_padding=downsample_padding,
            )
            self.downsample_blocks.append(down_block)

        # mid
        self.mid = UNetMidBlock2DCrossAttn(
            in_channels=block_channels[-1],
            dropout=dropout,
            temb_channels=time_embed_dim,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attn_num_head_channels=num_head_channels,
            resnet_groups=resnet_num_groups,
        )

        # up
        reversed_block_channels = list(reversed(block_channels))
        output_channel = reversed_block_channels[0]
        for i, up_block_type in enumerate(up_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_channels[i]
            input_channel = reversed_block_channels[min(i + 1, len(block_channels) - 1)]

            is_final_block = i == len(block_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=num_res_blocks + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                attn_num_head_channels=num_head_channels,
            )
            self.upsample_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_channels[0], num_groups=resnet_num_groups, eps=resnet_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_channels[0], out_channels, 3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.FloatTensor]:

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        t_emb = self.time_steps(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.downsample_blocks:

            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for upsample_block in self.upsample_blocks:

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

        # 6. post-process

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output = {"sample": sample}

        return output
