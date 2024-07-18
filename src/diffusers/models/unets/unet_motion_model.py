# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...configuration_utils import ConfigMixin, FrozenDict, register_to_config
from ...loaders import FromOriginalModelMixin, UNet2DConditionLoadersMixin
from ...utils import logging
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    AttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..transformers.transformer_temporal import TransformerTemporalModel
from .unet_2d_blocks import UNetMidBlock2DCrossAttn
from .unet_2d_condition import UNet2DConditionModel
from .unet_3d_blocks import (
    CrossAttnDownBlockMotion,
    CrossAttnUpBlockMotion,
    DownBlockMotion,
    UNetMidBlockCrossAttnMotion,
    UpBlockMotion,
    get_down_block,
    get_up_block,
)
from .unet_3d_condition import UNet3DConditionOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MotionModules(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layers_per_block: int = 2,
        transformer_layers_per_block: Union[int, Tuple[int]] = 8,
        num_attention_heads: Union[int, Tuple[int]] = 8,
        attention_bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_num_groups: int = 32,
        max_seq_length: int = 32,
    ):
        super().__init__()
        self.motion_modules = nn.ModuleList([])

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * layers_per_block
        elif len(transformer_layers_per_block) != layers_per_block:
            raise ValueError(
                f"The number of transformer layers per block must match the number of layers per block, "
                f"got {layers_per_block} and {len(transformer_layers_per_block)}"
            )

        for i in range(layers_per_block):
            self.motion_modules.append(
                TransformerTemporalModel(
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=in_channels // num_attention_heads,
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=max_seq_length,
                )
            )


class MotionAdapter(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        motion_layers_per_block: Union[int, Tuple[int]] = 2,
        motion_transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]] = 1,
        motion_mid_block_layers_per_block: int = 1,
        motion_transformer_layers_per_mid_block: Union[int, Tuple[int]] = 1,
        motion_num_attention_heads: Union[int, Tuple[int]] = 8,
        motion_norm_num_groups: int = 32,
        motion_max_seq_length: int = 32,
        use_motion_mid_block: bool = True,
        conv_in_channels: Optional[int] = None,
    ):
        """Container to store AnimateDiff Motion Modules

        Args:
            block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each UNet block.
            motion_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 2):
                The number of motion layers per UNet block.
            motion_transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple[int]]`, *optional*, defaults to 1):
                The number of transformer layers to use in each motion layer in each block.
            motion_mid_block_layers_per_block (`int`, *optional*, defaults to 1):
                The number of motion layers in the middle UNet block.
            motion_transformer_layers_per_mid_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
                The number of transformer layers to use in each motion layer in the middle block.
            motion_num_attention_heads (`int` or `Tuple[int]`, *optional*, defaults to 8):
                The number of heads to use in each attention layer of the motion module.
            motion_norm_num_groups (`int`, *optional*, defaults to 32):
                The number of groups to use in each group normalization layer of the motion module.
            motion_max_seq_length (`int`, *optional*, defaults to 32):
                The maximum sequence length to use in the motion module.
            use_motion_mid_block (`bool`, *optional*, defaults to True):
                Whether to use a motion module in the middle of the UNet.
        """

        super().__init__()
        down_blocks = []
        up_blocks = []

        if isinstance(motion_layers_per_block, int):
            motion_layers_per_block = (motion_layers_per_block,) * len(block_out_channels)
        elif len(motion_layers_per_block) != len(block_out_channels):
            raise ValueError(
                f"The number of motion layers per block must match the number of blocks, "
                f"got {len(block_out_channels)} and {len(motion_layers_per_block)}"
            )

        if isinstance(motion_transformer_layers_per_block, int):
            motion_transformer_layers_per_block = (motion_transformer_layers_per_block,) * len(block_out_channels)

        if isinstance(motion_transformer_layers_per_mid_block, int):
            motion_transformer_layers_per_mid_block = (
                motion_transformer_layers_per_mid_block,
            ) * motion_mid_block_layers_per_block
        elif len(motion_transformer_layers_per_mid_block) != motion_mid_block_layers_per_block:
            raise ValueError(
                f"The number of layers per mid block ({motion_mid_block_layers_per_block}) "
                f"must match the length of motion_transformer_layers_per_mid_block ({len(motion_transformer_layers_per_mid_block)})"
            )

        if isinstance(motion_num_attention_heads, int):
            motion_num_attention_heads = (motion_num_attention_heads,) * len(block_out_channels)
        elif len(motion_num_attention_heads) != len(block_out_channels):
            raise ValueError(
                f"The length of the attention head number tuple in the motion module must match the "
                f"number of block, got {len(motion_num_attention_heads)} and {len(block_out_channels)}"
            )

        if conv_in_channels:
            # input
            self.conv_in = nn.Conv2d(conv_in_channels, block_out_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = None

        for i, channel in enumerate(block_out_channels):
            output_channel = block_out_channels[i]
            down_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=None,
                    activation_fn="geglu",
                    attention_bias=False,
                    num_attention_heads=motion_num_attention_heads[i],
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=motion_layers_per_block[i],
                    transformer_layers_per_block=motion_transformer_layers_per_block[i],
                )
            )

        if use_motion_mid_block:
            self.mid_block = MotionModules(
                in_channels=block_out_channels[-1],
                norm_num_groups=motion_norm_num_groups,
                cross_attention_dim=None,
                activation_fn="geglu",
                attention_bias=False,
                num_attention_heads=motion_num_attention_heads[-1],
                max_seq_length=motion_max_seq_length,
                layers_per_block=motion_mid_block_layers_per_block,
                transformer_layers_per_block=motion_transformer_layers_per_mid_block,
            )
        else:
            self.mid_block = None

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        reversed_motion_layers_per_block = list(reversed(motion_layers_per_block))
        reversed_motion_transformer_layers_per_block = list(reversed(motion_transformer_layers_per_block))
        reversed_motion_num_attention_heads = list(reversed(motion_num_attention_heads))
        for i, channel in enumerate(reversed_block_out_channels):
            output_channel = reversed_block_out_channels[i]
            up_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=None,
                    activation_fn="geglu",
                    attention_bias=False,
                    num_attention_heads=reversed_motion_num_attention_heads[i],
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=reversed_motion_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_motion_transformer_layers_per_block[i],
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, sample):
        pass


class UNetMotionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    A modified conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a
    sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "DownBlockMotion",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlockMotion",
            "CrossAttnUpBlockMotion",
            "CrossAttnUpBlockMotion",
            "CrossAttnUpBlockMotion",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Union[int, Tuple[int], Tuple[Tuple]]] = None,
        temporal_transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_temporal_transformer_layers_per_block: Optional[Union[int, Tuple[int], Tuple[Tuple]]] = None,
        transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = None,
        temporal_transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = 1,
        use_linear_projection: bool = False,
        num_attention_heads: Union[int, Tuple[int, ...]] = 8,
        motion_max_seq_length: int = 32,
        motion_num_attention_heads: Union[int, Tuple[int, ...]] = 8,
        reverse_motion_num_attention_heads: Optional[Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]] = None,
        use_motion_mid_block: bool = True,
        mid_block_layers: int = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        projection_class_embeddings_input_dim: Optional[int] = None,
        time_cond_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

        if (
            isinstance(temporal_transformer_layers_per_block, list)
            and reverse_temporal_transformer_layers_per_block is None
        ):
            for layer_number_per_block in temporal_transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError(
                        "Must provide 'reverse_temporal_transformer_layers_per_block` if using asymmetrical motion module in UNet."
                    )

        # input
        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn=act_fn, cond_proj_dim=time_cond_proj_dim
        )

        if encoder_hid_dim_type is None:
            self.encoder_hid_proj = None

        if addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, True, 0)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # class embedding
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if isinstance(reverse_transformer_layers_per_block, int):
            reverse_transformer_layers_per_block = [reverse_transformer_layers_per_block] * len(down_block_types)

        if isinstance(temporal_transformer_layers_per_block, int):
            temporal_transformer_layers_per_block = [temporal_transformer_layers_per_block] * len(down_block_types)

        if isinstance(reverse_temporal_transformer_layers_per_block, int):
            reverse_temporal_transformer_layers_per_block = [reverse_temporal_transformer_layers_per_block] * len(
                down_block_types
            )

        if isinstance(motion_num_attention_heads, int):
            motion_num_attention_heads = (motion_num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                dual_cross_attention=False,
                temporal_num_attention_heads=motion_num_attention_heads[i],
                temporal_max_seq_length=motion_max_seq_length,
                transformer_layers_per_block=transformer_layers_per_block[i],
                temporal_transformer_layers_per_block=temporal_transformer_layers_per_block[i],
            )
            self.down_blocks.append(down_block)

        # mid
        if transformer_layers_per_mid_block is None:
            transformer_layers_per_mid_block = (
                transformer_layers_per_block[-1] if isinstance(transformer_layers_per_block[-1], int) else 1
            )

        if use_motion_mid_block:
            self.mid_block = UNetMidBlockCrossAttnMotion(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                num_layers=mid_block_layers,
                temporal_num_attention_heads=motion_num_attention_heads[-1],
                temporal_max_seq_length=motion_max_seq_length,
                transformer_layers_per_block=transformer_layers_per_mid_block,
                temporal_transformer_layers_per_block=temporal_transformer_layers_per_mid_block,
            )

        else:
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                num_layers=mid_block_layers,
                transformer_layers_per_block=transformer_layers_per_mid_block,
            )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_motion_num_attention_heads = list(reversed(motion_num_attention_heads))

        if reverse_transformer_layers_per_block is None:
            reverse_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        if reverse_temporal_transformer_layers_per_block is None:
            reverse_temporal_transformer_layers_per_block = list(reversed(temporal_transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                resolution_idx=i,
                use_linear_projection=use_linear_projection,
                temporal_num_attention_heads=reversed_motion_num_attention_heads[i],
                temporal_max_seq_length=motion_max_seq_length,
                transformer_layers_per_block=reverse_transformer_layers_per_block[i],
                temporal_transformer_layers_per_block=reverse_temporal_transformer_layers_per_block[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

    @classmethod
    def from_unet2d(
        cls,
        unet: UNet2DConditionModel,
        motion_adapter: Optional[MotionAdapter] = None,
        load_weights: bool = True,
    ):
        has_motion_adapter = motion_adapter is not None

        if has_motion_adapter:
            motion_adapter.to(device=unet.device)

            # check compatibility of number of blocks
            if len(unet.config["down_block_types"]) != len(motion_adapter.config["block_out_channels"]):
                raise ValueError("Incompatible Motion Adapter, got different number of blocks")

            # check layers compatibility for each block
            if isinstance(unet.config["layers_per_block"], int):
                expanded_layers_per_block = [unet.config["layers_per_block"]] * len(unet.config["down_block_types"])
            else:
                expanded_layers_per_block = list(unet.config["layers_per_block"])
            if isinstance(motion_adapter.config["motion_layers_per_block"], int):
                expanded_adapter_layers_per_block = [motion_adapter.config["motion_layers_per_block"]] * len(
                    motion_adapter.config["block_out_channels"]
                )
            else:
                expanded_adapter_layers_per_block = list(motion_adapter.config["motion_layers_per_block"])
            if expanded_layers_per_block != expanded_adapter_layers_per_block:
                raise ValueError("Incompatible Motion Adapter, got different number of layers per block")

        # based on https://github.com/guoyww/AnimateDiff/blob/895f3220c06318ea0760131ec70408b466c49333/animatediff/models/unet.py#L459
        config = dict(unet.config)
        config["_class_name"] = cls.__name__

        down_blocks = []
        for down_blocks_type in config["down_block_types"]:
            if "CrossAttn" in down_blocks_type:
                down_blocks.append("CrossAttnDownBlockMotion")
            else:
                down_blocks.append("DownBlockMotion")
        config["down_block_types"] = down_blocks

        up_blocks = []
        for down_blocks_type in config["up_block_types"]:
            if "CrossAttn" in down_blocks_type:
                up_blocks.append("CrossAttnUpBlockMotion")
            else:
                up_blocks.append("UpBlockMotion")
        config["up_block_types"] = up_blocks

        if has_motion_adapter:
            config["motion_num_attention_heads"] = motion_adapter.config["motion_num_attention_heads"]
            config["motion_max_seq_length"] = motion_adapter.config["motion_max_seq_length"]
            config["use_motion_mid_block"] = motion_adapter.config["use_motion_mid_block"]
            config["layers_per_block"] = motion_adapter.config["motion_layers_per_block"]
            config["temporal_transformer_layers_per_mid_block"] = motion_adapter.config[
                "motion_transformer_layers_per_mid_block"
            ]
            config["temporal_transformer_layers_per_block"] = motion_adapter.config[
                "motion_transformer_layers_per_block"
            ]
            config["motion_num_attention_heads"] = motion_adapter.config["motion_num_attention_heads"]

            # For PIA UNets we need to set the number input channels to 9
            if motion_adapter.config["conv_in_channels"]:
                config["in_channels"] = motion_adapter.config["conv_in_channels"]

        # Need this for backwards compatibility with UNet2DConditionModel checkpoints
        if not config.get("num_attention_heads"):
            config["num_attention_heads"] = config["attention_head_dim"]

        expected_kwargs, optional_kwargs = cls._get_signature_keys(cls)
        config = FrozenDict({k: config.get(k) for k in config if k in expected_kwargs or k in optional_kwargs})
        config["_class_name"] = cls.__name__
        model = cls.from_config(config)

        if not load_weights:
            return model

        # Logic for loading PIA UNets which allow the first 4 channels to be any UNet2DConditionModel conv_in weight
        # while the last 5 channels must be PIA conv_in weights.
        if has_motion_adapter and motion_adapter.config["conv_in_channels"]:
            model.conv_in = motion_adapter.conv_in
            updated_conv_in_weight = torch.cat(
                [unet.conv_in.weight, motion_adapter.conv_in.weight[:, 4:, :, :]], dim=1
            )
            model.conv_in.load_state_dict({"weight": updated_conv_in_weight, "bias": unet.conv_in.bias})
        else:
            model.conv_in.load_state_dict(unet.conv_in.state_dict())

        model.time_proj.load_state_dict(unet.time_proj.state_dict())
        model.time_embedding.load_state_dict(unet.time_embedding.state_dict())

        if any(
            isinstance(proc, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0))
            for proc in unet.attn_processors.values()
        ):
            attn_procs = {}
            for name, processor in unet.attn_processors.items():
                if name.endswith("attn1.processor"):
                    attn_processor_class = (
                        AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                    )
                    attn_procs[name] = attn_processor_class()
                else:
                    attn_processor_class = (
                        IPAdapterAttnProcessor2_0
                        if hasattr(F, "scaled_dot_product_attention")
                        else IPAdapterAttnProcessor
                    )
                    attn_procs[name] = attn_processor_class(
                        hidden_size=processor.hidden_size,
                        cross_attention_dim=processor.cross_attention_dim,
                        scale=processor.scale,
                        num_tokens=processor.num_tokens,
                    )
            for name, processor in model.attn_processors.items():
                if name not in attn_procs:
                    attn_procs[name] = processor.__class__()
            model.set_attn_processor(attn_procs)
            model.config.encoder_hid_dim_type = "ip_image_proj"
            model.encoder_hid_proj = unet.encoder_hid_proj

        for i, down_block in enumerate(unet.down_blocks):
            model.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
            if hasattr(model.down_blocks[i], "attentions"):
                model.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
            if model.down_blocks[i].downsamplers:
                model.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())

        for i, up_block in enumerate(unet.up_blocks):
            model.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
            if hasattr(model.up_blocks[i], "attentions"):
                model.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
            if model.up_blocks[i].upsamplers:
                model.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())

        model.mid_block.resnets.load_state_dict(unet.mid_block.resnets.state_dict())
        model.mid_block.attentions.load_state_dict(unet.mid_block.attentions.state_dict())

        if unet.conv_norm_out is not None:
            model.conv_norm_out.load_state_dict(unet.conv_norm_out.state_dict())
        if unet.conv_act is not None:
            model.conv_act.load_state_dict(unet.conv_act.state_dict())
        model.conv_out.load_state_dict(unet.conv_out.state_dict())

        if has_motion_adapter:
            model.load_motion_modules(motion_adapter)

        # ensure that the Motion UNet is the same dtype as the UNet2DConditionModel
        model.to(unet.dtype)

        return model

    def freeze_unet2d_params(self) -> None:
        """Freeze the weights of just the UNet2DConditionModel, and leave the motion modules
        unfrozen for fine tuning.
        """
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze Motion Modules
        for down_block in self.down_blocks:
            motion_modules = down_block.motion_modules
            for param in motion_modules.parameters():
                param.requires_grad = True

        for up_block in self.up_blocks:
            motion_modules = up_block.motion_modules
            for param in motion_modules.parameters():
                param.requires_grad = True

        if hasattr(self.mid_block, "motion_modules"):
            motion_modules = self.mid_block.motion_modules
            for param in motion_modules.parameters():
                param.requires_grad = True

    def load_motion_modules(self, motion_adapter: Optional[MotionAdapter]) -> None:
        for i, down_block in enumerate(motion_adapter.down_blocks):
            self.down_blocks[i].motion_modules.load_state_dict(down_block.motion_modules.state_dict())
        for i, up_block in enumerate(motion_adapter.up_blocks):
            self.up_blocks[i].motion_modules.load_state_dict(up_block.motion_modules.state_dict())

        # to support older motion modules that don't have a mid_block
        if hasattr(self.mid_block, "motion_modules"):
            self.mid_block.motion_modules.load_state_dict(motion_adapter.mid_block.motion_modules.state_dict())

    def save_motion_modules(
        self,
        save_directory: str,
        is_main_process: bool = True,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        state_dict = self.state_dict()

        # Extract all motion modules
        motion_state_dict = {}
        for k, v in state_dict.items():
            if "motion_modules" in k:
                motion_state_dict[k] = v

        adapter = MotionAdapter(
            block_out_channels=self.config["block_out_channels"],
            motion_layers_per_block=self.config["layers_per_block"],
            motion_norm_num_groups=self.config["norm_num_groups"],
            motion_num_attention_heads=self.config["motion_num_attention_heads"],
            motion_max_seq_length=self.config["motion_max_seq_length"],
            use_motion_mid_block=self.config["use_motion_mid_block"],
        )
        adapter.load_state_dict(motion_state_dict)
        adapter.save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            safe_serialization=safe_serialization,
            variant=variant,
            push_to_hub=push_to_hub,
            **kwargs,
        )

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

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self) -> None:
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self) -> None:
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

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlockMotion, DownBlockMotion, CrossAttnUpBlockMotion, UpBlockMotion)):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float) -> None:
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

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
    def disable_freeu(self) -> None:
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

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple[torch.Tensor]]:
        r"""
        The [`UNetMotionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

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
        num_frames = sample.shape[2]
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.config.addition_embed_type == "text_time":
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
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)

        emb = emb if aug_emb is None else emb + aug_emb
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds)
            image_embeds = [image_embed.repeat_interleave(repeats=num_frames, dim=0) for image_embed in image_embeds]
            encoder_hidden_states = (encoder_hidden_states, image_embeds)

        # 2. pre-process
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            # To support older versions of motion modules that don't have a mid_block
            if hasattr(self.mid_block, "motion_modules"):
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
