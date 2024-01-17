import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ..umer_debug_logger import udl
from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, is_torch_version, logging
from .autoencoders import AutoencoderKL
from .embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from .modeling_utils import ModelMixin
from .unet_2d_blocks import Downsample2D, ResnetBlock2D, Transformer2DModel, UNetMidBlock2DCrossAttn, Upsample2D
from .unet_2d_condition import UNet2DConditionModel


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

    @staticmethod
    def gather_base_subblock_sizes(blocks_sizes: List[int]):
        """todo - comment"""

        n_blocks = len(blocks_sizes)
        n_subblocks_per_block = 3

        down_out = []
        up_in = []

        # down_out
        for b in range(n_blocks):
            for i in range(n_subblocks_per_block):
                if b==n_blocks-1 and i==2:
                    # last block has now downsampler, so has only 2 subblocks instead of 3
                    continue
                if i==0:
                    # first subblock has same input channels as in last block,
                    # because channels are changed by the first resnet, which is the first subblock
                    down_out.append(blocks_sizes[max(b-1,0)])
                else:
                    down_out.append(blocks_sizes[b])
        down_out.append(blocks_sizes[-1])

        # up_in
        rev_blocks_sizes = list(reversed(blocks_sizes))
        for b in range(len(rev_blocks_sizes)):
            for i in range(n_subblocks_per_block):
                if i==0:
                    up_in.append(rev_blocks_sizes[max(b-1,0)])
                else:
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
    ):
        # todo - comment

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
            num_attention_heads = base_model.config.num_attention_heads

        norm_num_groups = math.gcd(*block_out_channels)

        return ControlNetXSAddon(
            learn_time_embedding=learn_time_embedding,
            channels_base=channels_base,
            addition_embed_type=base_model.config.addition_embed_type,
            addition_time_embed_dim=base_model.config.addition_time_embed_dim,
            attention_head_dim=num_attention_heads,
            block_out_channels=block_out_channels,
            cross_attention_dim=base_model.config.cross_attention_dim,
            down_block_types=base_model.config.down_block_types,
            projection_class_embeddings_input_dim=base_model.config.projection_class_embeddings_input_dim,
            sample_size=base_model.config.sample_size,
            transformer_layers_per_block=base_model.config.transformer_layers_per_block,
            upcast_attention=base_model.config.upcast_attention,
            norm_num_groups=norm_num_groups,
        )

    @register_to_config
    def __init__(
        self,
        conditioning_channel_order: str = "rgb",
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        time_embedding_input_dim: int = 320,
        time_embedding_dim: int = 1280,
        learn_time_embedding: bool = False,
        channels_base: Dict[str, List[Tuple[int]]] = {
            "down - out": [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280],
            "mid - out": 1280,
            "up - in": [1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320],
        },
        addition_embed_type=None,
        addition_time_embed_dim=None,
        attention_head_dim=[4],
        block_out_channels=[4, 8, 16, 16],
        cross_attention_dim=1024,
        down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
        projection_class_embeddings_input_dim=None,
        sample_size=96,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        upcast_attention=True,
        norm_num_groups=32,
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
        # todo - other checks

        # input
        self.conv_in = nn.Conv2d(4, block_out_channels[0], kernel_size=3, padding=1)

        # time
        if learn_time_embedding:
            time_embedding_dim = time_embedding_dim or block_out_channels[0] * 4
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
            self.time_embedding = TimestepEmbedding(time_embedding_input_dim, time_embedding_dim)
        else:
            self.time_proj = None
            self.time_embedding = None

        if addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embedding_dim)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

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
                    in_channels=input_channel + channels_base['down - out'][subblock_counter],
                    out_channels=output_channel,
                    temb_channels=time_embedding_dim,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    norm_num_groups=norm_num_groups,
                )
            )
            subblock_counter += 1
            self.down_subblocks.append(
                CrossAttnSubBlock2D(
                    has_crossattn=use_crossattention,
                    in_channels=output_channel + channels_base['down - out'][subblock_counter],
                    out_channels=output_channel,
                    temb_channels=time_embedding_dim,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    norm_num_groups=norm_num_groups,
                )
            )
            subblock_counter += 1
            if i < len(down_block_types) - 1:
                self.down_subblocks.append(
                    DownSubBlock2D(
                        in_channels=output_channel + channels_base['down - out'][subblock_counter],
                        out_channels=output_channel,
                    )
                )
                subblock_counter += 1

        # mid
        mid_in_channels = block_out_channels[-1] + channels_base['down - out'][subblock_counter]
        mid_out_channels = block_out_channels[-1]

        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_in_channels,
            out_channels=mid_out_channels,
            temb_channels=time_embedding_dim,
            resnet_eps=1e-05,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=find_largest_factor(mid_in_channels, norm_num_groups),
            resnet_groups_out=find_largest_factor(mid_out_channels, norm_num_groups),
            use_linear_projection=True,
            upcast_attention=upcast_attention,
        )

        # 3 - Gather Channel Sizes
        conditioning_embedding_out_channels
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
        # todo - better comment
        # Information is passed from base to ctrl _before_ each subblock. We therefore use the 'in' channels.
        # As the information is concatted in ctrl, we don't need to change channel sizes. So channels in = channels out.
        for c in channels_base["down - out"]: # change down - in to down - out
            self.down_zero_convs_b2c.append(self._make_zero_conv(c, c))

        # 4.2 - Connections from ctrl encoder to base encoder
        # Information is passed from ctrl to  base _after_ each subblock. We therefore use the 'out' channels.
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

    @torch.no_grad()
    def _check_if_vae_compatible(self, vae: AutoencoderKL):
        condition_downscale_factor = 2 ** (len(self.config.conditioning_embedding_out_channels) - 1)
        vae_downscale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        compatible = condition_downscale_factor == vae_downscale_factor
        return compatible, condition_downscale_factor, vae_downscale_factor

    def _make_zero_conv(self, in_channels, out_channels=None):
        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


class ControlNetXSModel(ModelMixin, ConfigMixin):
    r"""
    A ControlNet-XS model

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    Most of parameters for this model are passed into the [`UNet2DConditionModel`] it creates. Check the documentation
    of [`UNet2DConditionModel`] for them.

    Parameters:
        conditioning_channels (`int`, defaults to 3):
            Number of channels of conditioning input (e.g. an image)
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `controlnet_cond_embedding` layer.
        time_embedding_input_dim (`int`, defaults to 320):
            Dimension of input into time embedding. Needs to be same as in the base model.
        time_embedding_dim (`int`, defaults to 1280):
            Dimension of output from time embedding. Needs to be same as in the base model.
        learn_embedding (`bool`, defaults to `False`):
            Whether to use time embedding of the control model. If yes, the time embedding is a linear interpolation of
            the time embeddings of the control and base model with interpolation parameter `time_embedding_mix**3`.
        time_embedding_mix (`float`, defaults to 1.0):
            Linear interpolation parameter used if `learn_embedding` is `True`. A value of 1.0 means only the
            control model's time embedding will be used. A value of 0.0 means only the base model's time embedding will be used.
        channels_base (`Dict[str, List[Tuple[int]]]`):
            Channel sizes of each subblock of base model. Use `gather_subblock_sizes` on your base model to compute it.
    """

    @classmethod
    def init_original(cls, base_model: UNet2DConditionModel, is_sdxl=True):
        """
        Create a ControlNetXS model with the same parameters as in the original paper (https://github.com/vislearn/ControlNet-XS).

        Parameters:
            base_model (`UNet2DConditionModel`):
                Base UNet model. Needs to be either StableDiffusion or StableDiffusion-XL.
            is_sdxl (`bool`, defaults to `True`):
                Whether passed `base_model` is a StableDiffusion-XL model.
        """

        def get_dim_attn_heads(base_model: UNet2DConditionModel, size_ratio: float, num_attn_heads: int):
            """
            Currently, diffusers can only set the dimension of attention heads (see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why).
            The original ControlNet-XS model, however, define the number of attention heads.
            That's why compute the dimensions needed to get the correct number of attention heads.
            """
            block_out_channels = [int(size_ratio * c) for c in base_model.config.block_out_channels]
            dim_attn_heads = [math.ceil(c / num_attn_heads) for c in block_out_channels]
            return dim_attn_heads

        if is_sdxl:
            time_embedding_mix = 0.95
            controlnet_addon = ControlNetXSAddon.from_unet(
                base_model,
                learn_time_embedding=True,
                size_ratio=0.1,
                num_attention_heads=get_dim_attn_heads(base_model, 0.1, 64),
            )
        else:
            time_embedding_mix = 1.0
            controlnet_addon = ControlNetXSAddon.from_unet(
                base_model,
                learn_time_embedding=True,
                size_ratio=0.0125,
                num_attention_heads=get_dim_attn_heads(base_model, 0.0125, 8),
            )

        return cls(base_model=base_model, ctrl_model=controlnet_addon, time_embedding_mix=time_embedding_mix)

    @register_to_config
    def __init__(
        self,
        base_model: UNet2DConditionModel,
        ctrl_model: ControlNetXSAddon,
        time_embedding_mix: float = 1.0,
    ):
        super().__init__()

        # 1 - Save options
        self.use_ctrl_time_embedding = ctrl_model.config.learn_time_embedding
        self.conditioning_channel_order = ctrl_model.config.conditioning_channel_order
        self.class_embed_type = base_model.config.class_embed_type

        # 2 - Save control model parts
        self.ctrl_time_embedding = ctrl_model.time_embedding
        self.ctrl_conv_in = ctrl_model.conv_in
        self.ctrl_controlnet_cond_embedding = ctrl_model.controlnet_cond_embedding
        self.ctrl_down_subblocks = ctrl_model.down_subblocks
        self.ctrl_mid_block = ctrl_model.mid_block

        # 3 - Save connections
        self.down_zero_convs_b2c = ctrl_model.down_zero_convs_b2c
        self.down_zero_convs_c2b = ctrl_model.down_zero_convs_c2b
        self.mid_zero_convs_c2b = ctrl_model.mid_zero_convs_c2b
        self.up_zero_convs_c2b = ctrl_model.up_zero_convs_c2b

        # 4 - Save base model parts
        self.base_time_proj = base_model.time_proj
        self.base_time_embedding = base_model.time_embedding
        self.base_class_embedding = base_model.class_embedding
        self.base_addition_embed_type = base_model.config.addition_embed_type
        self.base_conv_in = base_model.conv_in
        self.base_down_subblocks = nn.ModuleList()
        self.base_mid_block = base_model.mid_block
        self.base_up_subblocks = nn.ModuleList()

        # 4.1 - SDXL specific components
        if hasattr(base_model, "add_time_proj"):
            self.base_add_time_proj = base_model.add_time_proj
        if hasattr(base_model, "add_embedding"):
            self.base_add_embedding = base_model.add_embedding

        # 4.2 - Decompose blocks of base model into subblocks
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

        self.base_conv_norm_out = base_model.conv_norm_out
        self.base_conv_act = base_model.conv_act
        self.base_conv_out = base_model.conv_out

        self.time_embedding_mix = time_embedding_mix

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
    ) -> Union[ControlNetXSOutput, Tuple]:
        """
        The [`ControlNetModel`] forward method.

        Args:
            base_model (`UNet2DConditionModel`):
                The base unet model we want to control.
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
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnetxs.ControlNetXSOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnetxs.ControlNetXSOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        # check channel order
        if self.conditioning_channel_order == "bgr":
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

        sample, timesteps, encoder_hidden_states, controlnet_cond = udl.do_input_action(
            x=sample,
            t=timesteps,
            xcross=encoder_hidden_states, 
            hint=controlnet_cond
        )

        udl.log_if('sample', sample, udl.SUBBLOCK)
        udl.log_if('timestep', timesteps, udl.SUBBLOCK)
        udl.log_if('encoder_hidden_states', encoder_hidden_states, udl.SUBBLOCK)
        udl.log_if('controlnet_cond', controlnet_cond, udl.SUBBLOCK)

        t_emb = self.base_time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.use_ctrl_time_embedding:
            ctrl_temb = self.ctrl_time_embedding(t_emb, timestep_cond)
            base_temb = self.base_time_embedding(t_emb, timestep_cond)
            interpolation_param = self.config.time_embedding_mix**0.3

            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = self.base_time_embedding(t_emb)

        # added time & text embeddings
        aug_emb = None

        if self.base_class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.class_embed_type == "timestep":
                class_labels = self.base_time_proj(class_labels)

            class_emb = self.base_class_embedding(class_labels).to(dtype=self.dtype)
            temb = temb + class_emb

        if self.base_addition_embed_type is None:
            pass
        elif self.base_addition_embed_type == "text_time":
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
            raise NotImplementedError()

        temb = temb + aug_emb if aug_emb is not None else temb

        # text embeddings
        cemb = encoder_hidden_states

        # Preparation
        guided_hint = self.ctrl_controlnet_cond_embedding(controlnet_cond)

        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []

        udl.log_if('h_ctrl', h_ctrl, udl.SUBBLOCK)
        udl.log_if('h_base', h_base, udl.SUBBLOCK)

        # Cross Control
        # 1 - conv in & down
        # The base -> ctrl connections are 'delayed' by 1 subblock, because we want to 'wait' to ensure the new information from the last  ctrl -> base connection is also considered
        # Therefore, the connections iterate over:
        #       ctrl -> base:   conv_in | subblock 1  |  ...  | subblock n
        #       base -> ctrl:           | subblock 1  |  ...  | subblock n | mid block

        h_base = self.base_conv_in(h_base)
        udl.log_if('base', h_base, udl.SUBBLOCK)
        h_ctrl = self.ctrl_conv_in(h_ctrl)
        udl.log_if('ctrl', h_ctrl, udl.SUBBLOCK)
        if guided_hint is not None:
            h_ctrl += guided_hint
        h_base = h_base + self.down_zero_convs_c2b[0](h_ctrl) * conditioning_scale  # add ctrl -> base
        udl.log_if('add c2b', h_base, udl.SUBBLOCK)

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        for b, c, b2c, c2b in zip(
            self.base_down_subblocks,
            self.ctrl_down_subblocks,
            self.down_zero_convs_b2c[:-1],
            self.down_zero_convs_c2b[1:],
        ):
            if isinstance(b, CrossAttnSubBlock2D):
                additional_params = [temb, cemb, attention_mask, cross_attention_kwargs]
            else:
                additional_params = []

            h_ctrl = torch.cat([h_ctrl, b2c(h_base)], dim=1)  # concat base -> ctrl
            udl.log_if('concat b2c', h_ctrl, udl.SUBBLOCK)

            h_base = b(h_base, *additional_params)  # apply base subblock
            udl.log_if('base', h_base, udl.SUBBLOCK)

            h_ctrl = c(h_ctrl, *additional_params)  # apply ctrl subblock
            udl.log_if('ctrl', h_ctrl, udl.SUBBLOCK)

            h_base = h_base + c2b(h_ctrl) * conditioning_scale  # add ctrl -> base
            udl.log_if('add c2b', h_base, udl.SUBBLOCK)

            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)
        h_ctrl = torch.cat([h_ctrl, self.down_zero_convs_b2c[-1](h_base)], dim=1)  # concat base -> ctrl 
        udl.log_if('concat b2c', h_ctrl, udl.SUBBLOCK)

        # 2 - mid
        h_base = self.base_mid_block(h_base, temb, cemb, attention_mask, cross_attention_kwargs)  # apply base subblock
        udl.log_if('base', h_base, udl.SUBBLOCK)
        
        h_ctrl = self.ctrl_mid_block(h_ctrl, temb, cemb, attention_mask, cross_attention_kwargs)  # apply ctrl subblock
        udl.log_if('ctrl', h_ctrl, udl.SUBBLOCK)

        h_base = h_base + self.mid_zero_convs_c2b(h_ctrl) * conditioning_scale  # add ctrl -> base
        udl.log_if('add c2b', h_base, udl.SUBBLOCK)

        # 3 - up
        for b, c2b, skip_c, skip_b in zip(
            self.base_up_subblocks, self.up_zero_convs_c2b, reversed(hs_ctrl), reversed(hs_base)
        ):
            h_base = h_base + c2b(skip_c) * conditioning_scale  # add info from ctrl encoder
            udl.log_if('add c2b', h_base, udl.SUBBLOCK)
    
            h_base = torch.cat([h_base, skip_b], dim=1)  # concat info from base encoder+ctrl encoder
            h_base = b(h_base, temb, cemb, attention_mask, cross_attention_kwargs)
            udl.log_if('base', h_base, udl.SUBBLOCK)

        h_base = self.base_conv_norm_out(h_base)
        h_base = self.base_conv_act(h_base)
        h_base = self.base_conv_out(h_base)
        udl.log_if('conv_out', h_base, udl.SUBBLOCK)

        udl.stop_if(udl.SUBBLOCK, 'It is done, my dude. Let us look at these tensors.')

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
        norm_num_groups: Optional[int] = 32,
        has_crossattn=False,
        transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,
        num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        upcast_attention: Optional[bool] = False,
    ):
        super().__init__()
        self.gradient_checkpointing = False

        if is_empty:
            # todo umer: comment
            return

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.resnet = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            groups=find_largest_factor(in_channels, start=norm_num_groups),
            groups_out=find_largest_factor(out_channels, start=norm_num_groups),
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
                norm_num_groups=find_largest_factor(out_channels, start=norm_num_groups),
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
            # todo umer: comment
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
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            # todo: gradient ckptin?
            hidden_states = self.downsampler(hidden_states)
        else:
            hidden_states = self.downsampler(hidden_states)

        return hidden_states


class CrossAttnUpSubBlock2D(nn.Module):
    def __init__(self):
        """todo doc - init emtpty as only from_modules will be used"""
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
