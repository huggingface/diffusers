# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ...utils.torch_utils import apply_freeu
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    Downsample2D,
    ResnetBlock2D,
    Transformer2DModel,
    UNetMidBlock2DCrossAttn,
    Upsample2D,
)
from ..unets.unet_2d_condition import UNet2DConditionModel
from .controlnet import ControlNetConditioningEmbedding


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetXSOutput(BaseOutput):
    """
    The output of [`UNetControlNetXSModel`].

    Args:
        sample (`Tensor` of shape `(batch_size, num_channels, height, width)`):
            The output of the `UNetControlNetXSModel`. Unlike `ControlNetOutput` this is NOT to be added to the base
            model output, but is already the final output.
    """

    sample: Tensor = None


class DownBlockControlNetXSAdapter(nn.Module):
    """Components that together with corresponding components from the base model will form a
    `ControlNetXSCrossAttnDownBlock2D`"""

    def __init__(
        self,
        resnets: nn.ModuleList,
        base_to_ctrl: nn.ModuleList,
        ctrl_to_base: nn.ModuleList,
        attentions: Optional[nn.ModuleList] = None,
        downsampler: Optional[nn.Conv2d] = None,
    ):
        super().__init__()
        self.resnets = resnets
        self.base_to_ctrl = base_to_ctrl
        self.ctrl_to_base = ctrl_to_base
        self.attentions = attentions
        self.downsamplers = downsampler


class MidBlockControlNetXSAdapter(nn.Module):
    """Components that together with corresponding components from the base model will form a
    `ControlNetXSCrossAttnMidBlock2D`"""

    def __init__(self, midblock: UNetMidBlock2DCrossAttn, base_to_ctrl: nn.ModuleList, ctrl_to_base: nn.ModuleList):
        super().__init__()
        self.midblock = midblock
        self.base_to_ctrl = base_to_ctrl
        self.ctrl_to_base = ctrl_to_base


class UpBlockControlNetXSAdapter(nn.Module):
    """Components that together with corresponding components from the base model will form a `ControlNetXSCrossAttnUpBlock2D`"""

    def __init__(self, ctrl_to_base: nn.ModuleList):
        super().__init__()
        self.ctrl_to_base = ctrl_to_base


def get_down_block_adapter(
    base_in_channels: int,
    base_out_channels: int,
    ctrl_in_channels: int,
    ctrl_out_channels: int,
    temb_channels: int,
    max_norm_num_groups: Optional[int] = 32,
    has_crossattn=True,
    transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,
    num_attention_heads: Optional[int] = 1,
    cross_attention_dim: Optional[int] = 1024,
    add_downsample: bool = True,
    upcast_attention: Optional[bool] = False,
    use_linear_projection: Optional[bool] = True,
):
    num_layers = 2  # only support sd + sdxl

    resnets = []
    attentions = []
    ctrl_to_base = []
    base_to_ctrl = []

    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * num_layers

    for i in range(num_layers):
        base_in_channels = base_in_channels if i == 0 else base_out_channels
        ctrl_in_channels = ctrl_in_channels if i == 0 else ctrl_out_channels

        # Before the resnet/attention application, information is concatted from base to control.
        # Concat doesn't require change in number of channels
        base_to_ctrl.append(make_zero_conv(base_in_channels, base_in_channels))

        resnets.append(
            ResnetBlock2D(
                in_channels=ctrl_in_channels + base_in_channels,  # information from base is concatted to ctrl
                out_channels=ctrl_out_channels,
                temb_channels=temb_channels,
                groups=find_largest_factor(ctrl_in_channels + base_in_channels, max_factor=max_norm_num_groups),
                groups_out=find_largest_factor(ctrl_out_channels, max_factor=max_norm_num_groups),
                eps=1e-5,
            )
        )

        if has_crossattn:
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    ctrl_out_channels // num_attention_heads,
                    in_channels=ctrl_out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    norm_num_groups=find_largest_factor(ctrl_out_channels, max_factor=max_norm_num_groups),
                )
            )

        # After the resnet/attention application, information is added from control to base
        # Addition requires change in number of channels
        ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))

    if add_downsample:
        # Before the downsampler application, information is concatted from base to control
        # Concat doesn't require change in number of channels
        base_to_ctrl.append(make_zero_conv(base_out_channels, base_out_channels))

        downsamplers = Downsample2D(
            ctrl_out_channels + base_out_channels, use_conv=True, out_channels=ctrl_out_channels, name="op"
        )

        # After the downsampler application, information is added from control to base
        # Addition requires change in number of channels
        ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
    else:
        downsamplers = None

    down_block_components = DownBlockControlNetXSAdapter(
        resnets=nn.ModuleList(resnets),
        base_to_ctrl=nn.ModuleList(base_to_ctrl),
        ctrl_to_base=nn.ModuleList(ctrl_to_base),
    )

    if has_crossattn:
        down_block_components.attentions = nn.ModuleList(attentions)
    if downsamplers is not None:
        down_block_components.downsamplers = downsamplers

    return down_block_components


def get_mid_block_adapter(
    base_channels: int,
    ctrl_channels: int,
    temb_channels: Optional[int] = None,
    max_norm_num_groups: Optional[int] = 32,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = 1,
    cross_attention_dim: Optional[int] = 1024,
    upcast_attention: bool = False,
    use_linear_projection: bool = True,
):
    # Before the midblock application, information is concatted from base to control.
    # Concat doesn't require change in number of channels
    base_to_ctrl = make_zero_conv(base_channels, base_channels)

    midblock = UNetMidBlock2DCrossAttn(
        transformer_layers_per_block=transformer_layers_per_block,
        in_channels=ctrl_channels + base_channels,
        out_channels=ctrl_channels,
        temb_channels=temb_channels,
        # number or norm groups must divide both in_channels and out_channels
        resnet_groups=find_largest_factor(gcd(ctrl_channels, ctrl_channels + base_channels), max_norm_num_groups),
        cross_attention_dim=cross_attention_dim,
        num_attention_heads=num_attention_heads,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
    )

    # After the midblock application, information is added from control to base
    # Addition requires change in number of channels
    ctrl_to_base = make_zero_conv(ctrl_channels, base_channels)

    return MidBlockControlNetXSAdapter(base_to_ctrl=base_to_ctrl, midblock=midblock, ctrl_to_base=ctrl_to_base)


def get_up_block_adapter(
    out_channels: int,
    prev_output_channel: int,
    ctrl_skip_channels: List[int],
):
    ctrl_to_base = []
    num_layers = 3  # only support sd + sdxl
    for i in range(num_layers):
        resnet_in_channels = prev_output_channel if i == 0 else out_channels
        ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))

    return UpBlockControlNetXSAdapter(ctrl_to_base=nn.ModuleList(ctrl_to_base))


class ControlNetXSAdapter(ModelMixin, ConfigMixin):
    r"""
    A `ControlNetXSAdapter` model. To use it, pass it into a `UNetControlNetXSModel` (together with a
    `UNet2DConditionModel` base model).

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    Like `UNetControlNetXSModel`, `ControlNetXSAdapter` is compatible with StableDiffusion and StableDiffusion-XL. It's
    default parameters are compatible with StableDiffusion.

    Parameters:
        conditioning_channels (`int`, defaults to 3):
            Number of channels of conditioning input (e.g. an image)
        conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
            The tuple of output channels for each block in the `controlnet_cond_embedding` layer.
        time_embedding_mix (`float`, defaults to 1.0):
            If 0, then only the control adapters's time embedding is used. If 1, then only the base unet's time
            embedding is used. Otherwise, both are combined.
        learn_time_embedding (`bool`, defaults to `False`):
            Whether a time embedding should be learned. If yes, `UNetControlNetXSModel` will combine the time
            embeddings of the base model and the control adapter. If no, `UNetControlNetXSModel` will use the base
            model's time embedding.
        num_attention_heads (`list[int]`, defaults to `[4]`):
            The number of attention heads.
        block_out_channels (`list[int]`, defaults to `[4, 8, 16, 16]`):
            The tuple of output channels for each block.
        base_block_out_channels (`list[int]`, defaults to `[320, 640, 1280, 1280]`):
            The tuple of output channels for each block in the base unet.
        cross_attention_dim (`int`, defaults to 1024):
            The dimension of the cross attention features.
        down_block_types (`list[str]`, defaults to `["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]`):
            The tuple of downsample blocks to use.
        sample_size (`int`, defaults to 96):
            Height and width of input/output sample.
        transformer_layers_per_block (`Union[int, Tuple[int]]`, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        upcast_attention (`bool`, defaults to `True`):
            Whether the attention computation should always be upcasted.
        max_norm_num_groups (`int`, defaults to 32):
            Maximum number of groups in group normal. The actual number will be the largest divisor of the respective
            channels, that is <= max_norm_num_groups.
    """

    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        time_embedding_mix: float = 1.0,
        learn_time_embedding: bool = False,
        num_attention_heads: Union[int, Tuple[int]] = 4,
        block_out_channels: Tuple[int, ...] = (4, 8, 16, 16),
        base_block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        cross_attention_dim: int = 1024,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        sample_size: Optional[int] = 96,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        upcast_attention: bool = True,
        max_norm_num_groups: int = 32,
        use_linear_projection: bool = True,
    ):
        super().__init__()

        time_embedding_input_dim = base_block_out_channels[0]
        time_embedding_dim = base_block_out_channels[0] * 4

        # Check inputs
        if conditioning_channel_order not in ["rgb", "bgr"]:
            raise ValueError(f"unknown `conditioning_channel_order`: {conditioning_channel_order}")

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(transformer_layers_per_block, (list, tuple)):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        if not isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = [cross_attention_dim] * len(down_block_types)
        # see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why `ControlNetXSAdapter` takes `num_attention_heads` instead of `attention_head_dim`
        if not isinstance(num_attention_heads, (list, tuple)):
            num_attention_heads = [num_attention_heads] * len(down_block_types)

        if len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        # 5 - Create conditioning hint embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        # time
        if learn_time_embedding:
            self.time_embedding = TimestepEmbedding(time_embedding_input_dim, time_embedding_dim)
        else:
            self.time_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.up_connections = nn.ModuleList([])

        # input
        self.conv_in = nn.Conv2d(4, block_out_channels[0], kernel_size=3, padding=1)
        self.control_to_base_for_conv_in = make_zero_conv(block_out_channels[0], base_block_out_channels[0])

        # down
        base_out_channels = base_block_out_channels[0]
        ctrl_out_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            base_in_channels = base_out_channels
            base_out_channels = base_block_out_channels[i]
            ctrl_in_channels = ctrl_out_channels
            ctrl_out_channels = block_out_channels[i]
            has_crossattn = "CrossAttn" in down_block_type
            is_final_block = i == len(down_block_types) - 1

            self.down_blocks.append(
                get_down_block_adapter(
                    base_in_channels=base_in_channels,
                    base_out_channels=base_out_channels,
                    ctrl_in_channels=ctrl_in_channels,
                    ctrl_out_channels=ctrl_out_channels,
                    temb_channels=time_embedding_dim,
                    max_norm_num_groups=max_norm_num_groups,
                    has_crossattn=has_crossattn,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim[i],
                    add_downsample=not is_final_block,
                    upcast_attention=upcast_attention,
                    use_linear_projection=use_linear_projection,
                )
            )

        # mid
        self.mid_block = get_mid_block_adapter(
            base_channels=base_block_out_channels[-1],
            ctrl_channels=block_out_channels[-1],
            temb_channels=time_embedding_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        # up
        # The skip connection channels are the output of the conv_in and of all the down subblocks
        ctrl_skip_channels = [block_out_channels[0]]
        for i, out_channels in enumerate(block_out_channels):
            number_of_subblocks = (
                3 if i < len(block_out_channels) - 1 else 2
            )  # every block has 3 subblocks, except last one, which has 2 as it has no downsampler
            ctrl_skip_channels.extend([out_channels] * number_of_subblocks)

        reversed_base_block_out_channels = list(reversed(base_block_out_channels))

        base_out_channels = reversed_base_block_out_channels[0]
        for i in range(len(down_block_types)):
            prev_base_output_channel = base_out_channels
            base_out_channels = reversed_base_block_out_channels[i]
            ctrl_skip_channels_ = [ctrl_skip_channels.pop() for _ in range(3)]

            self.up_connections.append(
                get_up_block_adapter(
                    out_channels=base_out_channels,
                    prev_output_channel=prev_base_output_channel,
                    ctrl_skip_channels=ctrl_skip_channels_,
                )
            )

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        size_ratio: Optional[float] = None,
        block_out_channels: Optional[List[int]] = None,
        num_attention_heads: Optional[List[int]] = None,
        learn_time_embedding: bool = False,
        time_embedding_mix: int = 1.0,
        conditioning_channels: int = 3,
        conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        r"""
        Instantiate a [`ControlNetXSAdapter`] from a [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model we want to control. The dimensions of the ControlNetXSAdapter will be adapted to it.
            size_ratio (float, *optional*, defaults to `None`):
                When given, block_out_channels is set to a fraction of the base model's block_out_channels. Either this
                or `block_out_channels` must be given.
            block_out_channels (`List[int]`, *optional*, defaults to `None`):
                Down blocks output channels in control model. Either this or `size_ratio` must be given.
            num_attention_heads (`List[int]`, *optional*, defaults to `None`):
                The dimension of the attention heads. The naming seems a bit confusing and it is, see
                https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
            learn_time_embedding (`bool`, defaults to `False`):
                Whether the `ControlNetXSAdapter` should learn a time embedding.
            time_embedding_mix (`float`, defaults to 1.0):
                If 0, then only the control adapter's time embedding is used. If 1, then only the base unet's time
                embedding is used. Otherwise, both are combined.
            conditioning_channels (`int`, defaults to 3):
                Number of channels of conditioning input (e.g. an image)
            conditioning_channel_order (`str`, defaults to `"rgb"`):
                The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
            conditioning_embedding_out_channels (`Tuple[int]`, defaults to `(16, 32, 96, 256)`):
                The tuple of output channel for each block in the `controlnet_cond_embedding` layer.
        """

        # Check input
        fixed_size = block_out_channels is not None
        relative_size = size_ratio is not None
        if not (fixed_size ^ relative_size):
            raise ValueError(
                "Pass exactly one of `block_out_channels` (for absolute sizing) or `size_ratio` (for relative sizing)."
            )

        # Create model
        block_out_channels = block_out_channels or [int(b * size_ratio) for b in unet.config.block_out_channels]
        if num_attention_heads is None:
            # The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
            num_attention_heads = unet.config.attention_head_dim

        model = cls(
            conditioning_channels=conditioning_channels,
            conditioning_channel_order=conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            time_embedding_mix=time_embedding_mix,
            learn_time_embedding=learn_time_embedding,
            num_attention_heads=num_attention_heads,
            block_out_channels=block_out_channels,
            base_block_out_channels=unet.config.block_out_channels,
            cross_attention_dim=unet.config.cross_attention_dim,
            down_block_types=unet.config.down_block_types,
            sample_size=unet.config.sample_size,
            transformer_layers_per_block=unet.config.transformer_layers_per_block,
            upcast_attention=unet.config.upcast_attention,
            max_norm_num_groups=unet.config.norm_num_groups,
            use_linear_projection=unet.config.use_linear_projection,
        )

        # ensure that the ControlNetXSAdapter is the same dtype as the UNet2DConditionModel
        model.to(unet.dtype)

        return model

    def forward(self, *args, **kwargs):
        raise ValueError(
            "A ControlNetXSAdapter cannot be run by itself. Use it together with a UNet2DConditionModel to instantiate a UNetControlNetXSModel."
        )


class UNetControlNetXSModel(ModelMixin, ConfigMixin):
    r"""
    A UNet fused with a ControlNet-XS adapter model

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    `UNetControlNetXSModel` is compatible with StableDiffusion and StableDiffusion-XL. It's default parameters are
    compatible with StableDiffusion.

    It's parameters are either passed to the underlying `UNet2DConditionModel` or used exactly like in
    `ControlNetXSAdapter` . See their documentation for details.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # unet configs
        sample_size: Optional[int] = 96,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = 8,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        upcast_attention: bool = True,
        use_linear_projection: bool = True,
        time_cond_proj_dim: Optional[int] = None,
        projection_class_embeddings_input_dim: Optional[int] = None,
        # additional controlnet configs
        time_embedding_mix: float = 1.0,
        ctrl_conditioning_channels: int = 3,
        ctrl_conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        ctrl_conditioning_channel_order: str = "rgb",
        ctrl_learn_time_embedding: bool = False,
        ctrl_block_out_channels: Tuple[int, ...] = (4, 8, 16, 16),
        ctrl_num_attention_heads: Union[int, Tuple[int]] = 4,
        ctrl_max_norm_num_groups: int = 32,
    ):
        super().__init__()

        if time_embedding_mix < 0 or time_embedding_mix > 1:
            raise ValueError("`time_embedding_mix` needs to be between 0 and 1.")
        if time_embedding_mix < 1 and not ctrl_learn_time_embedding:
            raise ValueError("To use `time_embedding_mix` < 1, `ctrl_learn_time_embedding` must be `True`")

        if addition_embed_type is not None and addition_embed_type != "text_time":
            raise ValueError(
                "As `UNetControlNetXSModel` currently only supports StableDiffusion and StableDiffusion-XL, `addition_embed_type` must be `None` or `'text_time'`."
            )

        if not isinstance(transformer_layers_per_block, (list, tuple)):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        if not isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = [cross_attention_dim] * len(down_block_types)
        if not isinstance(num_attention_heads, (list, tuple)):
            num_attention_heads = [num_attention_heads] * len(down_block_types)
        if not isinstance(ctrl_num_attention_heads, (list, tuple)):
            ctrl_num_attention_heads = [ctrl_num_attention_heads] * len(down_block_types)

        base_num_attention_heads = num_attention_heads

        self.in_channels = 4

        # # Input
        self.base_conv_in = nn.Conv2d(4, block_out_channels[0], kernel_size=3, padding=1)
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=ctrl_block_out_channels[0],
            block_out_channels=ctrl_conditioning_embedding_out_channels,
            conditioning_channels=ctrl_conditioning_channels,
        )
        self.ctrl_conv_in = nn.Conv2d(4, ctrl_block_out_channels[0], kernel_size=3, padding=1)
        self.control_to_base_for_conv_in = make_zero_conv(ctrl_block_out_channels[0], block_out_channels[0])

        # # Time
        time_embed_input_dim = block_out_channels[0]
        time_embed_dim = block_out_channels[0] * 4

        self.base_time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.base_time_embedding = TimestepEmbedding(
            time_embed_input_dim,
            time_embed_dim,
            cond_proj_dim=time_cond_proj_dim,
        )
        if ctrl_learn_time_embedding:
            self.ctrl_time_embedding = TimestepEmbedding(
                in_channels=time_embed_input_dim, time_embed_dim=time_embed_dim
            )
        else:
            self.ctrl_time_embedding = None

        if addition_embed_type is None:
            self.base_add_time_proj = None
            self.base_add_embedding = None
        else:
            self.base_add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.base_add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # # Create down blocks
        down_blocks = []
        base_out_channels = block_out_channels[0]
        ctrl_out_channels = ctrl_block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            base_in_channels = base_out_channels
            base_out_channels = block_out_channels[i]
            ctrl_in_channels = ctrl_out_channels
            ctrl_out_channels = ctrl_block_out_channels[i]
            has_crossattn = "CrossAttn" in down_block_type
            is_final_block = i == len(down_block_types) - 1

            down_blocks.append(
                ControlNetXSCrossAttnDownBlock2D(
                    base_in_channels=base_in_channels,
                    base_out_channels=base_out_channels,
                    ctrl_in_channels=ctrl_in_channels,
                    ctrl_out_channels=ctrl_out_channels,
                    temb_channels=time_embed_dim,
                    norm_num_groups=norm_num_groups,
                    ctrl_max_norm_num_groups=ctrl_max_norm_num_groups,
                    has_crossattn=has_crossattn,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    base_num_attention_heads=base_num_attention_heads[i],
                    ctrl_num_attention_heads=ctrl_num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim[i],
                    add_downsample=not is_final_block,
                    upcast_attention=upcast_attention,
                    use_linear_projection=use_linear_projection,
                )
            )

        # # Create mid block
        self.mid_block = ControlNetXSCrossAttnMidBlock2D(
            base_channels=block_out_channels[-1],
            ctrl_channels=ctrl_block_out_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            ctrl_max_norm_num_groups=ctrl_max_norm_num_groups,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            base_num_attention_heads=base_num_attention_heads[-1],
            ctrl_num_attention_heads=ctrl_num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        # # Create up blocks
        up_blocks = []
        rev_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        rev_num_attention_heads = list(reversed(base_num_attention_heads))
        rev_cross_attention_dim = list(reversed(cross_attention_dim))

        # The skip connection channels are the output of the conv_in and of all the down subblocks
        ctrl_skip_channels = [ctrl_block_out_channels[0]]
        for i, out_channels in enumerate(ctrl_block_out_channels):
            number_of_subblocks = (
                3 if i < len(ctrl_block_out_channels) - 1 else 2
            )  # every block has 3 subblocks, except last one, which has 2 as it has no downsampler
            ctrl_skip_channels.extend([out_channels] * number_of_subblocks)

        reversed_block_out_channels = list(reversed(block_out_channels))

        out_channels = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = out_channels
            out_channels = reversed_block_out_channels[i]
            in_channels = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            ctrl_skip_channels_ = [ctrl_skip_channels.pop() for _ in range(3)]

            has_crossattn = "CrossAttn" in up_block_type
            is_final_block = i == len(block_out_channels) - 1

            up_blocks.append(
                ControlNetXSCrossAttnUpBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    prev_output_channel=prev_output_channel,
                    ctrl_skip_channels=ctrl_skip_channels_,
                    temb_channels=time_embed_dim,
                    resolution_idx=i,
                    has_crossattn=has_crossattn,
                    transformer_layers_per_block=rev_transformer_layers_per_block[i],
                    num_attention_heads=rev_num_attention_heads[i],
                    cross_attention_dim=rev_cross_attention_dim[i],
                    add_upsample=not is_final_block,
                    upcast_attention=upcast_attention,
                    norm_num_groups=norm_num_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.base_conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups)
        self.base_conv_act = nn.SiLU()
        self.base_conv_out = nn.Conv2d(block_out_channels[0], 4, kernel_size=3, padding=1)

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet: Optional[ControlNetXSAdapter] = None,
        size_ratio: Optional[float] = None,
        ctrl_block_out_channels: Optional[List[float]] = None,
        time_embedding_mix: Optional[float] = None,
        ctrl_optional_kwargs: Optional[Dict] = None,
    ):
        r"""
        Instantiate a [`UNetControlNetXSModel`] from a [`UNet2DConditionModel`] and an optional [`ControlNetXSAdapter`]
        .

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model we want to control.
            controlnet (`ControlNetXSAdapter`):
                The ControlNet-XS adapter with which the UNet will be fused. If none is given, a new ControlNet-XS
                adapter will be created.
            size_ratio (float, *optional*, defaults to `None`):
                Used to construct the controlnet if none is given. See [`ControlNetXSAdapter.from_unet`] for details.
            ctrl_block_out_channels (`List[int]`, *optional*, defaults to `None`):
                Used to construct the controlnet if none is given. See [`ControlNetXSAdapter.from_unet`] for details,
                where this parameter is called `block_out_channels`.
            time_embedding_mix (`float`, *optional*, defaults to None):
                Used to construct the controlnet if none is given. See [`ControlNetXSAdapter.from_unet`] for details.
            ctrl_optional_kwargs (`Dict`, *optional*, defaults to `None`):
                Passed to the `init` of the new controlnet if no controlnet was given.
        """
        if controlnet is None:
            controlnet = ControlNetXSAdapter.from_unet(
                unet, size_ratio, ctrl_block_out_channels, **ctrl_optional_kwargs
            )
        else:
            if any(
                o is not None for o in (size_ratio, ctrl_block_out_channels, time_embedding_mix, ctrl_optional_kwargs)
            ):
                raise ValueError(
                    "When a controlnet is passed, none of these parameters should be passed: size_ratio, ctrl_block_out_channels, time_embedding_mix, ctrl_optional_kwargs."
                )

        # # get params
        params_for_unet = [
            "sample_size",
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "norm_num_groups",
            "cross_attention_dim",
            "transformer_layers_per_block",
            "addition_embed_type",
            "addition_time_embed_dim",
            "upcast_attention",
            "use_linear_projection",
            "time_cond_proj_dim",
            "projection_class_embeddings_input_dim",
        ]
        params_for_unet = {k: v for k, v in unet.config.items() if k in params_for_unet}
        # The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
        params_for_unet["num_attention_heads"] = unet.config.attention_head_dim

        params_for_controlnet = [
            "conditioning_channels",
            "conditioning_embedding_out_channels",
            "conditioning_channel_order",
            "learn_time_embedding",
            "block_out_channels",
            "num_attention_heads",
            "max_norm_num_groups",
        ]
        params_for_controlnet = {"ctrl_" + k: v for k, v in controlnet.config.items() if k in params_for_controlnet}
        params_for_controlnet["time_embedding_mix"] = controlnet.config.time_embedding_mix

        # # create model
        model = cls.from_config({**params_for_unet, **params_for_controlnet})

        # # load weights
        # from unet
        modules_from_unet = [
            "time_embedding",
            "conv_in",
            "conv_norm_out",
            "conv_out",
        ]
        for m in modules_from_unet:
            getattr(model, "base_" + m).load_state_dict(getattr(unet, m).state_dict())

        optional_modules_from_unet = [
            "add_time_proj",
            "add_embedding",
        ]
        for m in optional_modules_from_unet:
            if hasattr(unet, m) and getattr(unet, m) is not None:
                getattr(model, "base_" + m).load_state_dict(getattr(unet, m).state_dict())

        # from controlnet
        model.controlnet_cond_embedding.load_state_dict(controlnet.controlnet_cond_embedding.state_dict())
        model.ctrl_conv_in.load_state_dict(controlnet.conv_in.state_dict())
        if controlnet.time_embedding is not None:
            model.ctrl_time_embedding.load_state_dict(controlnet.time_embedding.state_dict())
        model.control_to_base_for_conv_in.load_state_dict(controlnet.control_to_base_for_conv_in.state_dict())

        # from both
        model.down_blocks = nn.ModuleList(
            ControlNetXSCrossAttnDownBlock2D.from_modules(b, c)
            for b, c in zip(unet.down_blocks, controlnet.down_blocks)
        )
        model.mid_block = ControlNetXSCrossAttnMidBlock2D.from_modules(unet.mid_block, controlnet.mid_block)
        model.up_blocks = nn.ModuleList(
            ControlNetXSCrossAttnUpBlock2D.from_modules(b, c)
            for b, c in zip(unet.up_blocks, controlnet.up_connections)
        )

        # ensure that the UNetControlNetXSModel is the same dtype as the UNet2DConditionModel
        model.to(unet.dtype)

        return model

    def freeze_unet_params(self) -> None:
        """Freeze the weights of the parts belonging to the base UNet2DConditionModel, and leave everything else unfrozen for fine
        tuning."""
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = True

        # Unfreeze ControlNetXSAdapter
        base_parts = [
            "base_time_proj",
            "base_time_embedding",
            "base_add_time_proj",
            "base_add_embedding",
            "base_conv_in",
            "base_conv_norm_out",
            "base_conv_act",
            "base_conv_out",
        ]
        base_parts = [getattr(self, part) for part in base_parts if getattr(self, part) is not None]
        for part in base_parts:
            for param in part.parameters():
                param.requires_grad = False

        for d in self.down_blocks:
            d.freeze_base_params()
        self.mid_block.freeze_base_params()
        for u in self.up_blocks:
            u.freeze_base_params()

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism from https://huggingface.co/papers/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        > [!WARNING] > This API is 🧪 experimental.
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        > [!WARNING] > This API is 🧪 experimental.

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        sample: Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor] = None,
        conditioning_scale: Optional[float] = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
        apply_control: bool = True,
    ) -> Union[ControlNetXSOutput, Tuple]:
        """
        The [`ControlNetXSModel`] forward method.

        Args:
            sample (`Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                How much the control model affects the base model outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnets.controlnet.ControlNetOutput`] instead of a plain
                tuple.
            apply_control (`bool`, defaults to `True`):
                If `False`, the input is run only through the base model.

        Returns:
            [`~models.controlnetxs.ControlNetXSOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnetxs.ControlNetXSOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        # check channel order
        if self.config.ctrl_conditioning_channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.base_time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.ctrl_learn_time_embedding and apply_control:
            ctrl_temb = self.ctrl_time_embedding(t_emb, timestep_cond)
            base_temb = self.base_time_embedding(t_emb, timestep_cond)
            interpolation_param = self.config.time_embedding_mix**0.3

            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = self.base_time_embedding(t_emb)

        # added time & text embeddings
        aug_emb = None

        if self.config.addition_embed_type is None:
            pass
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.base_add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(temb.dtype)
            aug_emb = self.base_add_embedding(add_embeds)
        else:
            raise ValueError(
                f"ControlNet-XS currently only supports StableDiffusion and StableDiffusion-XL, so addition_embed_type = {self.config.addition_embed_type} is currently not supported."
            )

        temb = temb + aug_emb if aug_emb is not None else temb

        # text embeddings
        cemb = encoder_hidden_states

        # Preparation
        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []

        # Cross Control
        guided_hint = self.controlnet_cond_embedding(controlnet_cond)

        # 1 - conv in & down

        h_base = self.base_conv_in(h_base)
        h_ctrl = self.ctrl_conv_in(h_ctrl)
        if guided_hint is not None:
            h_ctrl += guided_hint
        if apply_control:
            h_base = h_base + self.control_to_base_for_conv_in(h_ctrl) * conditioning_scale  # add ctrl -> base

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        for down in self.down_blocks:
            h_base, h_ctrl, residual_hb, residual_hc = down(
                hidden_states_base=h_base,
                hidden_states_ctrl=h_ctrl,
                temb=temb,
                encoder_hidden_states=cemb,
                conditioning_scale=conditioning_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                apply_control=apply_control,
            )
            hs_base.extend(residual_hb)
            hs_ctrl.extend(residual_hc)

        # 2 - mid
        h_base, h_ctrl = self.mid_block(
            hidden_states_base=h_base,
            hidden_states_ctrl=h_ctrl,
            temb=temb,
            encoder_hidden_states=cemb,
            conditioning_scale=conditioning_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            apply_control=apply_control,
        )

        # 3 - up
        for up in self.up_blocks:
            n_resnets = len(up.resnets)
            skips_hb = hs_base[-n_resnets:]
            skips_hc = hs_ctrl[-n_resnets:]
            hs_base = hs_base[:-n_resnets]
            hs_ctrl = hs_ctrl[:-n_resnets]
            h_base = up(
                hidden_states=h_base,
                res_hidden_states_tuple_base=skips_hb,
                res_hidden_states_tuple_ctrl=skips_hc,
                temb=temb,
                encoder_hidden_states=cemb,
                conditioning_scale=conditioning_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                apply_control=apply_control,
            )

        # 4 - conv out
        h_base = self.base_conv_norm_out(h_base)
        h_base = self.base_conv_act(h_base)
        h_base = self.base_conv_out(h_base)

        if not return_dict:
            return (h_base,)

        return ControlNetXSOutput(sample=h_base)


class ControlNetXSCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        base_in_channels: int,
        base_out_channels: int,
        ctrl_in_channels: int,
        ctrl_out_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        ctrl_max_norm_num_groups: int = 32,
        has_crossattn=True,
        transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,
        base_num_attention_heads: Optional[int] = 1,
        ctrl_num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        add_downsample: bool = True,
        upcast_attention: Optional[bool] = False,
        use_linear_projection: Optional[bool] = True,
    ):
        super().__init__()
        base_resnets = []
        base_attentions = []
        ctrl_resnets = []
        ctrl_attentions = []
        ctrl_to_base = []
        base_to_ctrl = []

        num_layers = 2  # only support sd + sdxl

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            base_in_channels = base_in_channels if i == 0 else base_out_channels
            ctrl_in_channels = ctrl_in_channels if i == 0 else ctrl_out_channels

            # Before the resnet/attention application, information is concatted from base to control.
            # Concat doesn't require change in number of channels
            base_to_ctrl.append(make_zero_conv(base_in_channels, base_in_channels))

            base_resnets.append(
                ResnetBlock2D(
                    in_channels=base_in_channels,
                    out_channels=base_out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )
            ctrl_resnets.append(
                ResnetBlock2D(
                    in_channels=ctrl_in_channels + base_in_channels,  # information from base is concatted to ctrl
                    out_channels=ctrl_out_channels,
                    temb_channels=temb_channels,
                    groups=find_largest_factor(
                        ctrl_in_channels + base_in_channels, max_factor=ctrl_max_norm_num_groups
                    ),
                    groups_out=find_largest_factor(ctrl_out_channels, max_factor=ctrl_max_norm_num_groups),
                    eps=1e-5,
                )
            )

            if has_crossattn:
                base_attentions.append(
                    Transformer2DModel(
                        base_num_attention_heads,
                        base_out_channels // base_num_attention_heads,
                        in_channels=base_out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        norm_num_groups=norm_num_groups,
                    )
                )
                ctrl_attentions.append(
                    Transformer2DModel(
                        ctrl_num_attention_heads,
                        ctrl_out_channels // ctrl_num_attention_heads,
                        in_channels=ctrl_out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        norm_num_groups=find_largest_factor(ctrl_out_channels, max_factor=ctrl_max_norm_num_groups),
                    )
                )

            # After the resnet/attention application, information is added from control to base
            # Addition requires change in number of channels
            ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))

        if add_downsample:
            # Before the downsampler application, information is concatted from base to control
            # Concat doesn't require change in number of channels
            base_to_ctrl.append(make_zero_conv(base_out_channels, base_out_channels))

            self.base_downsamplers = Downsample2D(
                base_out_channels, use_conv=True, out_channels=base_out_channels, name="op"
            )
            self.ctrl_downsamplers = Downsample2D(
                ctrl_out_channels + base_out_channels, use_conv=True, out_channels=ctrl_out_channels, name="op"
            )

            # After the downsampler application, information is added from control to base
            # Addition requires change in number of channels
            ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
        else:
            self.base_downsamplers = None
            self.ctrl_downsamplers = None

        self.base_resnets = nn.ModuleList(base_resnets)
        self.ctrl_resnets = nn.ModuleList(ctrl_resnets)
        self.base_attentions = nn.ModuleList(base_attentions) if has_crossattn else [None] * num_layers
        self.ctrl_attentions = nn.ModuleList(ctrl_attentions) if has_crossattn else [None] * num_layers
        self.base_to_ctrl = nn.ModuleList(base_to_ctrl)
        self.ctrl_to_base = nn.ModuleList(ctrl_to_base)

        self.gradient_checkpointing = False

    @classmethod
    def from_modules(cls, base_downblock: CrossAttnDownBlock2D, ctrl_downblock: DownBlockControlNetXSAdapter):
        # get params
        def get_first_cross_attention(block):
            return block.attentions[0].transformer_blocks[0].attn2

        base_in_channels = base_downblock.resnets[0].in_channels
        base_out_channels = base_downblock.resnets[0].out_channels
        ctrl_in_channels = (
            ctrl_downblock.resnets[0].in_channels - base_in_channels
        )  # base channels are concatted to ctrl channels in init
        ctrl_out_channels = ctrl_downblock.resnets[0].out_channels
        temb_channels = base_downblock.resnets[0].time_emb_proj.in_features
        num_groups = base_downblock.resnets[0].norm1.num_groups
        ctrl_num_groups = ctrl_downblock.resnets[0].norm1.num_groups
        if hasattr(base_downblock, "attentions"):
            has_crossattn = True
            transformer_layers_per_block = len(base_downblock.attentions[0].transformer_blocks)
            base_num_attention_heads = get_first_cross_attention(base_downblock).heads
            ctrl_num_attention_heads = get_first_cross_attention(ctrl_downblock).heads
            cross_attention_dim = get_first_cross_attention(base_downblock).cross_attention_dim
            upcast_attention = get_first_cross_attention(base_downblock).upcast_attention
            use_linear_projection = base_downblock.attentions[0].use_linear_projection
        else:
            has_crossattn = False
            transformer_layers_per_block = None
            base_num_attention_heads = None
            ctrl_num_attention_heads = None
            cross_attention_dim = None
            upcast_attention = None
            use_linear_projection = None
        add_downsample = base_downblock.downsamplers is not None

        # create model
        model = cls(
            base_in_channels=base_in_channels,
            base_out_channels=base_out_channels,
            ctrl_in_channels=ctrl_in_channels,
            ctrl_out_channels=ctrl_out_channels,
            temb_channels=temb_channels,
            norm_num_groups=num_groups,
            ctrl_max_norm_num_groups=ctrl_num_groups,
            has_crossattn=has_crossattn,
            transformer_layers_per_block=transformer_layers_per_block,
            base_num_attention_heads=base_num_attention_heads,
            ctrl_num_attention_heads=ctrl_num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            add_downsample=add_downsample,
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        # # load weights
        model.base_resnets.load_state_dict(base_downblock.resnets.state_dict())
        model.ctrl_resnets.load_state_dict(ctrl_downblock.resnets.state_dict())
        if has_crossattn:
            model.base_attentions.load_state_dict(base_downblock.attentions.state_dict())
            model.ctrl_attentions.load_state_dict(ctrl_downblock.attentions.state_dict())
        if add_downsample:
            model.base_downsamplers.load_state_dict(base_downblock.downsamplers[0].state_dict())
            model.ctrl_downsamplers.load_state_dict(ctrl_downblock.downsamplers.state_dict())
        model.base_to_ctrl.load_state_dict(ctrl_downblock.base_to_ctrl.state_dict())
        model.ctrl_to_base.load_state_dict(ctrl_downblock.ctrl_to_base.state_dict())

        return model

    def freeze_base_params(self) -> None:
        """Freeze the weights of the parts belonging to the base UNet2DConditionModel, and leave everything else unfrozen for fine
        tuning."""
        # Unfreeze everything
        for param in self.parameters():
            param.requires_grad = True

        # Freeze base part
        base_parts = [self.base_resnets]
        if isinstance(self.base_attentions, nn.ModuleList):  # attentions can be a list of Nones
            base_parts.append(self.base_attentions)
        if self.base_downsamplers is not None:
            base_parts.append(self.base_downsamplers)
        for part in base_parts:
            for param in part.parameters():
                param.requires_grad = False

    def forward(
        self,
        hidden_states_base: Tensor,
        temb: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        hidden_states_ctrl: Optional[Tensor] = None,
        conditioning_scale: Optional[float] = 1.0,
        attention_mask: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        apply_control: bool = True,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        h_base = hidden_states_base
        h_ctrl = hidden_states_ctrl

        base_output_states = ()
        ctrl_output_states = ()

        base_blocks = list(zip(self.base_resnets, self.base_attentions))
        ctrl_blocks = list(zip(self.ctrl_resnets, self.ctrl_attentions))

        for (b_res, b_attn), (c_res, c_attn), b2c, c2b in zip(
            base_blocks, ctrl_blocks, self.base_to_ctrl, self.ctrl_to_base
        ):
            # concat base -> ctrl
            if apply_control:
                h_ctrl = torch.cat([h_ctrl, b2c(h_base)], dim=1)

            # apply base subblock
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                h_base = self._gradient_checkpointing_func(b_res, h_base, temb)
            else:
                h_base = b_res(h_base, temb)

            if b_attn is not None:
                h_base = b_attn(
                    h_base,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            # apply ctrl subblock
            if apply_control:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    h_ctrl = self._gradient_checkpointing_func(c_res, h_ctrl, temb)
                else:
                    h_ctrl = c_res(h_ctrl, temb)
                if c_attn is not None:
                    h_ctrl = c_attn(
                        h_ctrl,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

            # add ctrl -> base
            if apply_control:
                h_base = h_base + c2b(h_ctrl) * conditioning_scale

            base_output_states = base_output_states + (h_base,)
            ctrl_output_states = ctrl_output_states + (h_ctrl,)

        if self.base_downsamplers is not None:  # if we have a base_downsampler, then also a ctrl_downsampler
            b2c = self.base_to_ctrl[-1]
            c2b = self.ctrl_to_base[-1]

            # concat base -> ctrl
            if apply_control:
                h_ctrl = torch.cat([h_ctrl, b2c(h_base)], dim=1)
            # apply base subblock
            h_base = self.base_downsamplers(h_base)
            # apply ctrl subblock
            if apply_control:
                h_ctrl = self.ctrl_downsamplers(h_ctrl)
            # add ctrl -> base
            if apply_control:
                h_base = h_base + c2b(h_ctrl) * conditioning_scale

            base_output_states = base_output_states + (h_base,)
            ctrl_output_states = ctrl_output_states + (h_ctrl,)

        return h_base, h_ctrl, base_output_states, ctrl_output_states


class ControlNetXSCrossAttnMidBlock2D(nn.Module):
    def __init__(
        self,
        base_channels: int,
        ctrl_channels: int,
        temb_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        ctrl_max_norm_num_groups: int = 32,
        transformer_layers_per_block: int = 1,
        base_num_attention_heads: Optional[int] = 1,
        ctrl_num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        upcast_attention: bool = False,
        use_linear_projection: Optional[bool] = True,
    ):
        super().__init__()

        # Before the midblock application, information is concatted from base to control.
        # Concat doesn't require change in number of channels
        self.base_to_ctrl = make_zero_conv(base_channels, base_channels)

        self.base_midblock = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=base_channels,
            temb_channels=temb_channels,
            resnet_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=base_num_attention_heads,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        self.ctrl_midblock = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=ctrl_channels + base_channels,
            out_channels=ctrl_channels,
            temb_channels=temb_channels,
            # number or norm groups must divide both in_channels and out_channels
            resnet_groups=find_largest_factor(
                gcd(ctrl_channels, ctrl_channels + base_channels), ctrl_max_norm_num_groups
            ),
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=ctrl_num_attention_heads,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        # After the midblock application, information is added from control to base
        # Addition requires change in number of channels
        self.ctrl_to_base = make_zero_conv(ctrl_channels, base_channels)

        self.gradient_checkpointing = False

    @classmethod
    def from_modules(
        cls,
        base_midblock: UNetMidBlock2DCrossAttn,
        ctrl_midblock: MidBlockControlNetXSAdapter,
    ):
        base_to_ctrl = ctrl_midblock.base_to_ctrl
        ctrl_to_base = ctrl_midblock.ctrl_to_base
        ctrl_midblock = ctrl_midblock.midblock

        # get params
        def get_first_cross_attention(midblock):
            return midblock.attentions[0].transformer_blocks[0].attn2

        base_channels = ctrl_to_base.out_channels
        ctrl_channels = ctrl_to_base.in_channels
        transformer_layers_per_block = len(base_midblock.attentions[0].transformer_blocks)
        temb_channels = base_midblock.resnets[0].time_emb_proj.in_features
        num_groups = base_midblock.resnets[0].norm1.num_groups
        ctrl_num_groups = ctrl_midblock.resnets[0].norm1.num_groups
        base_num_attention_heads = get_first_cross_attention(base_midblock).heads
        ctrl_num_attention_heads = get_first_cross_attention(ctrl_midblock).heads
        cross_attention_dim = get_first_cross_attention(base_midblock).cross_attention_dim
        upcast_attention = get_first_cross_attention(base_midblock).upcast_attention
        use_linear_projection = base_midblock.attentions[0].use_linear_projection

        # create model
        model = cls(
            base_channels=base_channels,
            ctrl_channels=ctrl_channels,
            temb_channels=temb_channels,
            norm_num_groups=num_groups,
            ctrl_max_norm_num_groups=ctrl_num_groups,
            transformer_layers_per_block=transformer_layers_per_block,
            base_num_attention_heads=base_num_attention_heads,
            ctrl_num_attention_heads=ctrl_num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        # load weights
        model.base_to_ctrl.load_state_dict(base_to_ctrl.state_dict())
        model.base_midblock.load_state_dict(base_midblock.state_dict())
        model.ctrl_midblock.load_state_dict(ctrl_midblock.state_dict())
        model.ctrl_to_base.load_state_dict(ctrl_to_base.state_dict())

        return model

    def freeze_base_params(self) -> None:
        """Freeze the weights of the parts belonging to the base UNet2DConditionModel, and leave everything else unfrozen for fine
        tuning."""
        # Unfreeze everything
        for param in self.parameters():
            param.requires_grad = True

        # Freeze base part
        for param in self.base_midblock.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states_base: Tensor,
        temb: Tensor,
        encoder_hidden_states: Tensor,
        hidden_states_ctrl: Optional[Tensor] = None,
        conditioning_scale: Optional[float] = 1.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        apply_control: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        h_base = hidden_states_base
        h_ctrl = hidden_states_ctrl

        joint_args = {
            "temb": temb,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
            "cross_attention_kwargs": cross_attention_kwargs,
            "encoder_attention_mask": encoder_attention_mask,
        }

        if apply_control:
            h_ctrl = torch.cat([h_ctrl, self.base_to_ctrl(h_base)], dim=1)  # concat base -> ctrl
        h_base = self.base_midblock(h_base, **joint_args)  # apply base mid block
        if apply_control:
            h_ctrl = self.ctrl_midblock(h_ctrl, **joint_args)  # apply ctrl mid block
            h_base = h_base + self.ctrl_to_base(h_ctrl) * conditioning_scale  # add ctrl -> base

        return h_base, h_ctrl


class ControlNetXSCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        ctrl_skip_channels: List[int],
        temb_channels: int,
        norm_num_groups: int = 32,
        resolution_idx: Optional[int] = None,
        has_crossattn=True,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1024,
        add_upsample: bool = True,
        upcast_attention: bool = False,
        use_linear_projection: Optional[bool] = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        ctrl_to_base = []

        num_layers = 3  # only support sd + sdxl

        self.has_cross_attention = has_crossattn
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )

            if has_crossattn:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        norm_num_groups=norm_num_groups,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions) if has_crossattn else [None] * num_layers
        self.ctrl_to_base = nn.ModuleList(ctrl_to_base)

        if add_upsample:
            self.upsamplers = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    @classmethod
    def from_modules(cls, base_upblock: CrossAttnUpBlock2D, ctrl_upblock: UpBlockControlNetXSAdapter):
        ctrl_to_base_skip_connections = ctrl_upblock.ctrl_to_base

        # get params
        def get_first_cross_attention(block):
            return block.attentions[0].transformer_blocks[0].attn2

        out_channels = base_upblock.resnets[0].out_channels
        in_channels = base_upblock.resnets[-1].in_channels - out_channels
        prev_output_channels = base_upblock.resnets[0].in_channels - out_channels
        ctrl_skip_channelss = [c.in_channels for c in ctrl_to_base_skip_connections]
        temb_channels = base_upblock.resnets[0].time_emb_proj.in_features
        num_groups = base_upblock.resnets[0].norm1.num_groups
        resolution_idx = base_upblock.resolution_idx
        if hasattr(base_upblock, "attentions"):
            has_crossattn = True
            transformer_layers_per_block = len(base_upblock.attentions[0].transformer_blocks)
            num_attention_heads = get_first_cross_attention(base_upblock).heads
            cross_attention_dim = get_first_cross_attention(base_upblock).cross_attention_dim
            upcast_attention = get_first_cross_attention(base_upblock).upcast_attention
            use_linear_projection = base_upblock.attentions[0].use_linear_projection
        else:
            has_crossattn = False
            transformer_layers_per_block = None
            num_attention_heads = None
            cross_attention_dim = None
            upcast_attention = None
            use_linear_projection = None
        add_upsample = base_upblock.upsamplers is not None

        # create model
        model = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channels,
            ctrl_skip_channels=ctrl_skip_channelss,
            temb_channels=temb_channels,
            norm_num_groups=num_groups,
            resolution_idx=resolution_idx,
            has_crossattn=has_crossattn,
            transformer_layers_per_block=transformer_layers_per_block,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            add_upsample=add_upsample,
            upcast_attention=upcast_attention,
            use_linear_projection=use_linear_projection,
        )

        # load weights
        model.resnets.load_state_dict(base_upblock.resnets.state_dict())
        if has_crossattn:
            model.attentions.load_state_dict(base_upblock.attentions.state_dict())
        if add_upsample:
            model.upsamplers.load_state_dict(base_upblock.upsamplers[0].state_dict())
        model.ctrl_to_base.load_state_dict(ctrl_to_base_skip_connections.state_dict())

        return model

    def freeze_base_params(self) -> None:
        """Freeze the weights of the parts belonging to the base UNet2DConditionModel, and leave everything else unfrozen for fine
        tuning."""
        # Unfreeze everything
        for param in self.parameters():
            param.requires_grad = True

        # Freeze base part
        base_parts = [self.resnets]
        if isinstance(self.attentions, nn.ModuleList):  # attentions can be a list of Nones
            base_parts.append(self.attentions)
        if self.upsamplers is not None:
            base_parts.append(self.upsamplers)
        for part in base_parts:
            for param in part.parameters():
                param.requires_grad = False

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple_base: Tuple[Tensor, ...],
        res_hidden_states_tuple_ctrl: Tuple[Tensor, ...],
        temb: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        conditioning_scale: Optional[float] = 1.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[Tensor] = None,
        upsample_size: Optional[int] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        apply_control: bool = True,
    ) -> Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        def maybe_apply_freeu_to_subblock(hidden_states, res_h_base):
            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                return apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_h_base,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )
            else:
                return hidden_states, res_h_base

        for resnet, attn, c2b, res_h_base, res_h_ctrl in zip(
            self.resnets,
            self.attentions,
            self.ctrl_to_base,
            reversed(res_hidden_states_tuple_base),
            reversed(res_hidden_states_tuple_ctrl),
        ):
            if apply_control:
                hidden_states += c2b(res_h_ctrl) * conditioning_scale

            hidden_states, res_h_base = maybe_apply_freeu_to_subblock(hidden_states, res_h_base)
            hidden_states = torch.cat([hidden_states, res_h_base], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            if attn is not None:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            hidden_states = self.upsamplers(hidden_states, upsample_size)

        return hidden_states


def make_zero_conv(in_channels, out_channels=None):
    return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def find_largest_factor(number, max_factor):
    factor = max_factor
    if factor >= number:
        return number
    while factor != 0:
        residual = number % factor
        if residual == 0:
            return factor
        factor -= 1
