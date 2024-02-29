# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, is_torch_version, logging
from .autoencoders import AutoencoderKL
from .embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from .modeling_utils import ModelMixin
from .unets.unet_2d_blocks import Downsample2D, ResnetBlock2D, Transformer2DModel, UNetMidBlock2DCrossAttn, Upsample2D
from .unets.unet_2d_condition import UNet2DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetXSOutput(BaseOutput):
    """
    The output of [`ControlNetXSModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The output of the `ControlNetXSModel`. Unlike `ControlNetOutput` this is NOT to be added to the base model
            output, but is already the final output.
    """

    sample: torch.FloatTensor = None


# copied from diffusers.models.controlnet.ControlNetConditioningEmbedding
class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetXSAddon(ModelMixin, ConfigMixin):
    r"""
    A `ControlNetXSAddon` model. To use it, pass it into a `ControlNetXSModel` (together with a `UNet2DConditionModel` base model).

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    Like `ControlNetXSModel`, `ControlNetXSAddon` is compatible with StableDiffusion and StableDiffusion-XL.
    It's default parameters are compatible with StableDiffusion.

    Parameters:
        conditioning_channels (`int`, defaults to 3):
            Number of channels of conditioning input (e.g. an image)
        conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
            The tuple of output channels for each block in the `controlnet_cond_embedding` layer.
        time_embedding_input_dim (`int`, defaults to 320):
            Dimension of input into time embedding. Needs to be same as in the base model.
        time_embedding_dim (`int`, defaults to 1280):
            Dimension of output from time embedding. Needs to be same as in the base model.
        learn_time_embedding (`bool`, defaults to `False`):
            Whether a time embedding should be learned. If yes, `ControlNetXSModel` will combine the time embeddings of the base model and the addon.
            If no, `ControlNetXSModel` will use the base model's time embedding.
        channels_base (`Dict[str, List[Tuple[int]]]`, defaults to `ControlNetXSAddon.gather_base_subblock_sizes((320,640,1280,1280))`):
            Channels of each subblock of the base model. Use `ControlNetXSAddon.gather_base_subblock_sizes` to obtain them.
        attention_head_dim (`list[int]`, defaults to `[4]`):
            The dimension of the attention heads.
        block_out_channels (`list[int]`, defaults to `[4, 8, 16, 16]`):
            The tuple of output channels for each block.
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
            Maximum number of groups in group normal. The actual number will the the largest divisor of the respective channels, that is <= max_norm_num_groups.
    """

    @staticmethod
    def gather_base_subblock_sizes(blocks_sizes: List[int]):
        """
        To create a correctly sized `ControlNetXSAddon`, we need to know
        the channels sizes of each base subblock.

        Parameters:
            blocks_sizes (`List[int]`):
                Channel sizes of each base block.
        """

        n_blocks = len(blocks_sizes)
        n_subblocks_per_block = 3

        down_out = []
        up_in = []

        # down_out
        for b in range(n_blocks):
            for i in range(n_subblocks_per_block):
                if b == n_blocks - 1 and i == 2:
                    # Last block has no downsampler, so there are only 2 subblocks instead of 3
                    continue

                # The input channels are changed by the first resnet, which is in the first subblock.
                if i == 0:
                    # Same input channels
                    down_out.append(blocks_sizes[max(b - 1, 0)])
                else:
                    # Changed input channels
                    down_out.append(blocks_sizes[b])

        down_out.append(blocks_sizes[-1])

        # up_in
        rev_blocks_sizes = list(reversed(blocks_sizes))
        for b in range(len(rev_blocks_sizes)):
            for i in range(n_subblocks_per_block):
                # The input channels are changed by the first resnet, which is in the first subblock.
                if i == 0:
                    # Same input channels
                    up_in.append(rev_blocks_sizes[max(b - 1, 0)])
                else:
                    # Changed input channels
                    up_in.append(rev_blocks_sizes[b])

        return {
            "down - out": down_out,
            "mid - out": blocks_sizes[-1],
            "up - in": up_in,
        }

    @classmethod
    def from_unet(
        cls,
        base_model: UNet2DConditionModel,
        size_ratio: Optional[float] = None,
        block_out_channels: Optional[List[int]] = None,
        num_attention_heads: Optional[List[int]] = None,
        learn_time_embedding: bool = False,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        r"""
        Instantiate a [`ControlNetXSAddon`] from a [`UNet2DConditionModel`].

        Parameters:
            base_model (`UNet2DConditionModel`):
                The UNet model we want to control. The dimensions of the ControlNetXSAddon will be adapted to it.
            size_ratio (float, *optional*, defaults to `None`):
                When given, block_out_channels is set to a fraction of the base model's block_out_channels.
                Either this or `block_out_channels` must be given.
            block_out_channels (`List[int]`, *optional*, defaults to `None`):
                Down blocks output channels in control model. Either this or `size_ratio` must be given.
            num_attention_heads (`List[int]`, *optional*, defaults to `None`):
                The dimension of the attention heads. The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
            learn_time_embedding (`bool`, defaults to `False`):
                Whether the `ControlNetXSAddon` should learn a time embedding.
            conditioning_embedding_out_channels (`Tuple[int]`, defaults to `(16, 32, 96, 256)`):
                The tuple of output channel for each block in the `controlnet_cond_embedding` layer.
        """

        # Check input
        fixed_size = block_out_channels is not None
        relative_size = size_ratio is not None
        if not (fixed_size ^ relative_size):
            raise ValueError(
                "Pass exactly one of `block_out_channels` (for absolute sizing) or `control_model_ratio` (for relative sizing)."
            )

        channels_base = ControlNetXSAddon.gather_base_subblock_sizes(base_model.config.block_out_channels)

        block_out_channels = [int(b * size_ratio) for b in base_model.config.block_out_channels]
        if num_attention_heads is None:
            # The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
            num_attention_heads = base_model.config.attention_head_dim

        max_norm_num_groups = base_model.config.norm_num_groups

        time_embedding_input_dim = base_model.time_embedding.linear_1.in_features
        time_embedding_dim = base_model.time_embedding.linear_1.out_features

        return ControlNetXSAddon(
            learn_time_embedding=learn_time_embedding,
            channels_base=channels_base,
            attention_head_dim=num_attention_heads,
            block_out_channels=block_out_channels,
            cross_attention_dim=base_model.config.cross_attention_dim,
            down_block_types=base_model.config.down_block_types,
            sample_size=base_model.config.sample_size,
            transformer_layers_per_block=base_model.config.transformer_layers_per_block,
            upcast_attention=base_model.config.upcast_attention,
            max_norm_num_groups=max_norm_num_groups,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            time_embedding_input_dim=time_embedding_input_dim,
            time_embedding_dim=time_embedding_dim,
        )

    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        time_embedding_input_dim: Optional[int] = 320,
        time_embedding_dim: Optional[int] = 1280,
        learn_time_embedding: bool = False,
        channels_base: Dict[str, List[Tuple[int]]] = {
            "down - out": [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280],
            "mid - out": 1280,
            "up - in": [1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320],
        },
        attention_head_dim: Union[int, Tuple[int]] = 4,
        block_out_channels: Tuple[int] = (4, 8, 16, 16),
        cross_attention_dim: int = 1024,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        sample_size: Optional[int] = 96,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        upcast_attention: bool = True,
        max_norm_num_groups: int = 32,
    ):
        super().__init__()

        self.sample_size = sample_size

        # `num_attention_heads` defaults to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = attention_head_dim

        # Check inputs
        if conditioning_channel_order not in ["rgb", "bgr"]:
            raise ValueError(f"unknown `conditioning_channel_order`: {conditioning_channel_order}")

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )
        elif isinstance(attention_head_dim, int):
            attention_head_dim = [attention_head_dim] * len(down_block_types)

        # input
        self.conv_in = nn.Conv2d(4, block_out_channels[0], kernel_size=3, padding=1)

        # time
        if learn_time_embedding:
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
            self.time_embedding = TimestepEmbedding(time_embedding_input_dim, time_embedding_dim)
        else:
            self.time_proj = None
            self.time_embedding = None

        self.time_embed_act = None

        self.down_subblocks = nn.ModuleList([])
        self.up_subblocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        subblock_counter = 0

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            use_crossattention = down_block_type == "CrossAttnDownBlock2D"

            self.down_subblocks.append(
                CrossAttnSubBlock2D(
                    has_crossattn=use_crossattention,
                    in_channels=input_channel + channels_base["down - out"][subblock_counter],
                    out_channels=output_channel,
                    temb_channels=time_embedding_dim,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    max_norm_num_groups=max_norm_num_groups,
                )
            )
            subblock_counter += 1
            self.down_subblocks.append(
                CrossAttnSubBlock2D(
                    has_crossattn=use_crossattention,
                    in_channels=output_channel + channels_base["down - out"][subblock_counter],
                    out_channels=output_channel,
                    temb_channels=time_embedding_dim,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    max_norm_num_groups=max_norm_num_groups,
                )
            )
            subblock_counter += 1
            if i < len(down_block_types) - 1:
                self.down_subblocks.append(
                    DownSubBlock2D(
                        in_channels=output_channel + channels_base["down - out"][subblock_counter],
                        out_channels=output_channel,
                    )
                )
                subblock_counter += 1

        # mid
        mid_in_channels = block_out_channels[-1] + channels_base["down - out"][subblock_counter]
        mid_out_channels = block_out_channels[-1]

        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_in_channels,
            out_channels=mid_out_channels,
            temb_channels=time_embedding_dim,
            resnet_eps=1e-05,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=find_largest_factor(mid_in_channels, max_norm_num_groups),
            resnet_groups_out=find_largest_factor(mid_out_channels, max_norm_num_groups),
            use_linear_projection=True,
            upcast_attention=upcast_attention,
        )

        # 3 - Gather Channel Sizes
        channels_ctrl = {
            "down - out": [self.conv_in.out_channels] + [s.out_channels for s in self.down_subblocks],
            "mid - out": self.down_subblocks[-1].out_channels,
        }

        # 4 - Build connections between base and control model
        # b2c = base -> ctrl ; c2b = ctrl -> base
        self.down_zero_convs_b2c = nn.ModuleList([])
        self.down_zero_convs_c2b = nn.ModuleList([])
        self.mid_zero_convs_c2b = nn.ModuleList([])
        self.up_zero_convs_c2b = nn.ModuleList([])

        # 4.1 - Connections from base encoder to ctrl encoder
        # As the information is concatted to ctrl, the channels sizes don't change.
        for c in channels_base["down - out"]:
            self.down_zero_convs_b2c.append(self._make_zero_conv(c, c))

        # 4.2 - Connections from ctrl encoder to base encoder
        # As the information is added to base, the out-channels need to match base.
        for ch_base, ch_ctrl in zip(channels_base["down - out"], channels_ctrl["down - out"]):
            self.down_zero_convs_c2b.append(self._make_zero_conv(ch_ctrl, ch_base))

        # 4.3 - Connections in mid block
        self.mid_zero_convs_c2b = self._make_zero_conv(channels_ctrl["mid - out"], channels_base["mid - out"])

        # 4.3 - Connections from ctrl encoder to base decoder
        skip_channels = reversed(channels_ctrl["down - out"])
        for s, i in zip(skip_channels, channels_base["up - in"]):
            self.up_zero_convs_c2b.append(self._make_zero_conv(s, i))

        # 5 - Create conditioning hint embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

    def forward(self, *args, **kwargs):
        raise ValueError(
            "A ControlNetXSAddonModel cannot be run by itself. Pass it into a ControlNetXSModel model instead."
        )

    def _make_zero_conv(self, in_channels, out_channels=None):
        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


class ControlNetXSModel(nn.Module):
    r"""
    A ControlNet-XS model

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    `ControlNetXSModel` is compatible with StableDiffusion and StableDiffusion-XL.
    It's default parameters are compatible with StableDiffusion.

    Parameters:
        base_model (`UNet2DConditionModel`):
            The base UNet to control.
        ctrl_addon (`ControlNetXSAddon`):
            The control addon.
        time_embedding_mix (`float`, defaults to 1.0):
            If 0, then only the base model's time embedding is used.
            If 1, then only the control model's time embedding is used.
            Otherwise, both are combined.
    """

    def __init__(
        self,
        base_model: UNet2DConditionModel,
        ctrl_addon: ControlNetXSAddon,
        time_embedding_mix: float = 1.0,
    ):
        super().__init__()

        if time_embedding_mix < 0 or time_embedding_mix > 1:
            raise ValueError("`time_embedding_mix` needs to be between 0 and 1.")
        if time_embedding_mix < 1 and not ctrl_addon.config.learn_time_embedding:
            raise ValueError(
                "To use `time_embedding_mix` < 1, initialize `ctrl_addon` with `learn_time_embedding = True`"
            )

        self.ctrl_addon = ctrl_addon
        self.base_model = base_model
        self.time_embedding_mix = time_embedding_mix

        # Decompose blocks of base model into subblocks
        self.base_down_subblocks = nn.ModuleList()
        self.base_up_subblocks = nn.ModuleList()

        for block in base_model.down_blocks:
            # Each ResNet / Attention pair is a subblock
            resnets = block.resnets
            attentions = block.attentions if hasattr(block, "attentions") else [None] * len(resnets)
            for r, a in zip(resnets, attentions):
                self.base_down_subblocks.append(CrossAttnSubBlock2D.from_modules(r, a))
            # Each Downsampler is a subblock
            if block.downsamplers is not None:
                if len(block.downsamplers) != 1:
                    raise ValueError(
                        "ControlNet-XS currently only supports StableDiffusion and StableDiffusion-XL."
                        "Therefore each down block of the base model should have only 1 downsampler (if any)."
                    )
                self.base_down_subblocks.append(DownSubBlock2D.from_modules(block.downsamplers[0]))

        for block in base_model.up_blocks:
            # Each ResNet / Attention / Upsampler triple is a subblock
            if block.upsamplers is not None:
                if len(block.upsamplers) != 1:
                    raise ValueError(
                        "ControlNet-XS currently only supports StableDiffusion and StableDiffusion-XL."
                        "Therefore each up block of the base model should have only 1 upsampler (if any)."
                    )
                upsampler = block.upsamplers[0]
            else:
                upsampler = None

            resnets = block.resnets
            attentions = block.attentions if hasattr(block, "attentions") else [None] * len(resnets)
            upsamplers = [None] * (len(resnets) - 1) + [upsampler]
            for r, a, u in zip(resnets, attentions, upsamplers):
                self.base_up_subblocks.append(CrossAttnUpSubBlock2D.from_modules(r, a, u))

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return self.base_model.device

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return self.base_model.dtype

    @torch.no_grad()
    def _check_if_vae_compatible(self, vae: AutoencoderKL):
        condition_downscale_factor = 2 ** (len(self.ctrl_addon.config.conditioning_embedding_out_channels) - 1)
        vae_downscale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        compatible = condition_downscale_factor == vae_downscale_factor
        return compatible, condition_downscale_factor, vae_downscale_factor

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
        do_control: bool = True,
    ) -> Union[ControlNetXSOutput, Tuple]:
        """
        The [`ControlNetXSModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.FloatTensor`):
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
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.
            do_control (`bool`, defaults to `True`):
                If `False`, the input is run only through the base model.

        Returns:
            [`~models.controlnetxs.ControlNetXSOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnetxs.ControlNetXSOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        if not do_control:
            return self.base_model(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict,
            )

        # check channel order
        if self.ctrl_addon.config.conditioning_channel_order == "bgr":
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
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.base_model.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.ctrl_addon.config.learn_time_embedding:
            ctrl_temb = self.ctrl_addon.time_embedding(t_emb, timestep_cond)
            base_temb = self.base_model.time_embedding(t_emb, timestep_cond)
            interpolation_param = self.time_embedding_mix**0.3

            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = self.base_model.time_embedding(t_emb)

        # added time & text embeddings
        aug_emb = None

        if self.base_model.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.base_model.config.class_embed_type == "timestep":
                class_labels = self.base_time_proj(class_labels)

            class_emb = self.base_model.class_embedding(class_labels).to(dtype=self.dtype)
            temb = temb + class_emb

        if self.base_model.config.addition_embed_type is None:
            pass
        elif self.base_model.config.addition_embed_type == "text_time":
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
            time_embeds = self.base_model.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(temb.dtype)
            aug_emb = self.base_model.add_embedding(add_embeds)
        else:
            raise ValueError(
                f"ControlNet-XS currently only supports StableDiffusion and StableDiffusion-XL, so addition_embed_type = {self.base_model.config.addition_embed_type} is currently not supported."
            )

        temb = temb + aug_emb if aug_emb is not None else temb

        # text embeddings
        cemb = encoder_hidden_states

        # Preparation
        guided_hint = self.ctrl_addon.controlnet_cond_embedding(controlnet_cond)

        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []

        # Cross Control
        # Let's first define variables to shorten notation
        base_down_subblocks = self.base_down_subblocks
        ctrl_down_subblocks = self.ctrl_addon.down_subblocks

        down_zero_convs_b2c = self.ctrl_addon.down_zero_convs_b2c
        down_zero_convs_c2b = self.ctrl_addon.down_zero_convs_c2b
        mid_zero_convs_c2b = self.ctrl_addon.mid_zero_convs_c2b
        up_zero_convs_c2b = self.ctrl_addon.up_zero_convs_c2b

        # 1 - conv in & down
        # The base -> ctrl connections are "delayed" by 1 subblock, because we want to "wait" to ensure the new information from the last ctrl -> base connection is also considered.
        # Therefore, the connections iterate over:
        #       ctrl -> base:   conv_in | subblock 1  |  ...  | subblock n
        #       base -> ctrl:           | subblock 1  |  ...  | subblock n | mid block

        h_base = self.base_model.conv_in(h_base)
        h_ctrl = self.ctrl_addon.conv_in(h_ctrl)
        if guided_hint is not None:
            h_ctrl += guided_hint
        h_base = h_base + down_zero_convs_c2b[0](h_ctrl) * conditioning_scale  # add ctrl -> base

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        for b, c, b2c, c2b in zip(
            base_down_subblocks,
            ctrl_down_subblocks,
            down_zero_convs_b2c[:-1],
            down_zero_convs_c2b[1:],
        ):
            if isinstance(b, CrossAttnSubBlock2D):
                additional_params = [temb, cemb, attention_mask, cross_attention_kwargs]
            else:
                additional_params = []

            h_ctrl = torch.cat([h_ctrl, b2c(h_base)], dim=1)  # concat base -> ctrl
            h_base = b(h_base, *additional_params)  # apply base subblock
            h_ctrl = c(h_ctrl, *additional_params)  # apply ctrl subblock
            h_base = h_base + c2b(h_ctrl) * conditioning_scale  # add ctrl -> base

            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)
        h_ctrl = torch.cat([h_ctrl, down_zero_convs_b2c[-1](h_base)], dim=1)  # concat base -> ctrl

        # 2 - mid
        h_base = self.base_model.mid_block(
            h_base, temb, cemb, attention_mask, cross_attention_kwargs
        )  # apply base subblock
        h_ctrl = self.ctrl_addon.mid_block(
            h_ctrl, temb, cemb, attention_mask, cross_attention_kwargs
        )  # apply ctrl subblock
        h_base = h_base + mid_zero_convs_c2b(h_ctrl) * conditioning_scale  # add ctrl -> base

        # 3 - up
        for b, c2b, skip_c, skip_b in zip(
            self.base_up_subblocks, up_zero_convs_c2b, reversed(hs_ctrl), reversed(hs_base)
        ):
            h_base = h_base + c2b(skip_c) * conditioning_scale  # add info from ctrl encoder
            h_base = torch.cat([h_base, skip_b], dim=1)  # concat info from base encoder+ctrl encoder
            h_base = b(h_base, temb, cemb, attention_mask, cross_attention_kwargs)

        h_base = self.base_model.conv_norm_out(h_base)
        h_base = self.base_model.conv_act(h_base)
        h_base = self.base_model.conv_out(h_base)

        if not return_dict:
            return h_base

        return ControlNetXSOutput(sample=h_base)


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


class CrossAttnSubBlock2D(nn.Module):
    def __init__(
        self,
        is_empty: bool = False,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = None,
        max_norm_num_groups: Optional[int] = 32,
        has_crossattn=False,
        transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,
        num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        upcast_attention: Optional[bool] = False,
    ):
        super().__init__()
        self.gradient_checkpointing = False

        if is_empty:
            # modules will be set manually, see `CrossAttnSubBlock2D.from_modules`
            return

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.resnet = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            groups=find_largest_factor(in_channels, max_factor=max_norm_num_groups),
            groups_out=find_largest_factor(out_channels, max_factor=max_norm_num_groups),
            eps=1e-5,
        )

        if has_crossattn:
            self.attention = Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block,
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=True,
                upcast_attention=upcast_attention,
                norm_num_groups=find_largest_factor(out_channels, max_factor=max_norm_num_groups),
            )
        else:
            self.attention = None

    @classmethod
    def from_modules(cls, resnet: ResnetBlock2D, attention: Optional[Transformer2DModel] = None):
        """Create empty subblock and set resnet and attention manually"""
        subblock = cls(is_empty=True)
        subblock.resnet = resnet
        subblock.attention = attention
        subblock.in_channels = resnet.in_channels
        subblock.out_channels = resnet.out_channels
        return subblock

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            if self.resnet is not None:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            if self.attention is not None:
                hidden_states = self.attention(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
        else:
            if self.resnet is not None:
                hidden_states = self.resnet(hidden_states, temb, scale=lora_scale)
            if self.attention is not None:
                hidden_states = self.attention(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

        return hidden_states


class DownSubBlock2D(nn.Module):
    def __init__(
        self,
        is_empty: bool = False,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.gradient_checkpointing = False

        if is_empty:
            # downsampler will be set manually, see `DownSubBlock2D.from_modules`
            return

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsampler = Downsample2D(in_channels, use_conv=True, out_channels=out_channels, name="op")

    @classmethod
    def from_modules(cls, downsampler: Downsample2D):
        """Create empty subblock and set downsampler manually"""
        subblock = cls(is_empty=True)
        subblock.downsampler = downsampler
        subblock.in_channels = downsampler.channels
        subblock.out_channels = downsampler.out_channels
        return subblock

    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        return self.downsampler(hidden_states)


class CrossAttnUpSubBlock2D(nn.Module):
    def __init__(self):
        """
        In the context of ControlNet-XS, `CrossAttnUpSubBlock2D` are only loaded from existing modules, and not created from scratch.
        Therefore, `__init__` is left almost empty.
        """
        super().__init__()
        self.gradient_checkpointing = False

    @classmethod
    def from_modules(
        cls,
        resnet: ResnetBlock2D,
        attention: Optional[Transformer2DModel] = None,
        upsampler: Optional[Upsample2D] = None,
    ):
        """Create empty subblock and set resnet, attention and upsampler manually"""
        subblock = cls()
        subblock.resnet = resnet
        subblock.attention = attention
        subblock.upsampler = upsampler
        subblock.in_channels = resnet.in_channels
        subblock.out_channels = resnet.out_channels
        return subblock

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            if self.attention is not None:
                hidden_states = self.attention(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            if self.upsampler is not None:
                hidden_states = self.upsampler(hidden_states)
        else:
            hidden_states = self.resnet(hidden_states, temb, scale=lora_scale)
            if self.attention is not None:
                hidden_states = self.attention(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            if self.upsampler is not None:
                hidden_states = self.upsampler(hidden_states)

        return hidden_states
