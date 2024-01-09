import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import GroupNorm

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging, is_torch_version
from .attention_processor import (
    AttentionProcessor,
)
from .autoencoders import AutoencoderKL
from .lora import LoRACompatibleConv
from .embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from .modeling_utils import ModelMixin
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    Downsample2D,
    ResnetBlock2D,
    Transformer2DModel,
    UNetMidBlock2DCrossAttn,
)
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
    @classmethod
    def init_original(cls, sd_type):
        kwargs = {}
        if sd_type == "sdxl":
            kwargs.update({
                'addition_embed_type': "text_time",
                'addition_time_embed_dim': 256,
                'attention_head_dim': [5, 10, 20],
                'block_out_channels': [320, 640, 1280],
                'cross_attention_dim': 2048,
                'down_block_types': ['DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D'],
                'projection_class_embeddings_input_dim': 2816,
                'sample_size': 128,
                'transformer_layers_per_block': [1, 2, 10],
                'up_block_types': ['CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D'],
                'upcast_attention': None,
            })
        elif sd_type == "sd":
            kwargs.update({
                'addition_embed_type': None,
                'addition_time_embed_dim': None,
                'attention_head_dim': [5, 10, 20, 20],
                'block_out_channels': [320, 640, 1280, 1280],
                'cross_attention_dim': 1024,
                'down_block_types': ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
                'projection_class_embeddings_input_dim': None,
                'sample_size': 96,
                'transformer_layers_per_block': 1,
                'up_block_types': ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'],
                'upcast_attention': True
            })
        else:
            raise ValueError("`sd_type` needs to either 'sd' or 'sdxl'")

        return ControlNetXSAddon(**kwargs)

    @register_to_config
    def __init__(
        self,
        channels_from_base_model: List[int],
        time_embedding_input_dim: int = 320,
        time_embedding_dim: int = 1280,
        time_embedding_mix: float = 1.0,
        learn_embedding: bool = False,
        base_model_channel_sizes: Dict[str, List[Tuple[int]]] = {
            "down": [
                (4, 320),
                (320, 320),
                (320, 320),
                (320, 320),
                (320, 640),
                (640, 640),
                (640, 640),
                (640, 1280),
                (1280, 1280),
            ],
            "mid": [(1280, 1280)],
            "up": [
                (2560, 1280),
                (2560, 1280),
                (1920, 1280),
                (1920, 640),
                (1280, 640),
                (960, 640),
                (960, 320),
                (640, 320),
                (640, 320),
            ],
        },
        addition_embed_type = None,
        addition_time_embed_dim = None,
        attention_head_dim = [5, 10, 20, 20], 
        block_out_channels = [320, 640, 1280, 1280],
        cross_attention_dim = 1024,
        down_block_types = ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
        projection_class_embeddings_input_dim = None,
        sample_size = 96,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        upcast_attention = True,
    ):
        super().__init__()

        # todo:
        # replace model surgery
		#	- 2.2 Allow for information infusion from base model
		#	- 2.3 Make group norms work with modified channel sizes
        # add connections

        self.sample_size = sample_size

        # `num_attention_heads` defaults to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = attention_head_dim

        # Check inputs
        # todo

        # input
        self.conv_in = nn.Conv2d(4, block_out_channels[0], kernel_size=3, padding=1)

        # time
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)

        # note umer: here `time_embedding_input_dim` is used, so time info can be received from base model
        self.time_embedding = TimestepEmbedding(time_embedding_input_dim, time_embed_dim)

        self.encoder_hid_proj = None

        # class embedding
        self.class_embedding = None

        if addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        self.time_embed_act = None
        
        self.down_subblocks = nn.ModuleList([])
        self.up_subblocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            use_crossattention = down_block_type == "CrossAttnDownBlock2D"

            self.down_subblocks.append(DownSubBlock2D(
                has_resnet=True,
                has_crossattn=use_crossattention,
                in_channels=input_channel + 0, # todo add channels from base model
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                transformer_layers_per_block=transformer_layers_per_block[i],
                num_attention_heads=num_attention_heads[i],
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention
            ))
            self.down_subblocks.append(DownSubBlock2D(
                has_resnet=True,
                has_crossattn=use_crossattention,
                in_channels=output_channel + 0, # todo add channels from base model
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                transformer_layers_per_block=transformer_layers_per_block[i],
                num_attention_heads=num_attention_heads[i],
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention
            ))
            self.down_subblocks.append(DownSubBlock2D(
                has_downsampler=True,
                in_channels=output_channel + 0, # todo add channels from base model
                out_channels=output_channel,
            ))

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            dropout=0.0,
            resnet_eps=1e-05,
            resnet_act_fn="silu",
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=32,
            dual_cross_attention=False,
            use_linear_projection=True,
            upcast_attention=upcast_attention,
            attention_type="default",
        )

        # todo: connections
        # 3 - Gather Channel Sizes
        self.ch_inout_ctrl = ControlNetXSModel._gather_subblock_sizes(self.control_model, base_or_control="control")
        self.ch_inout_base = base_model_channel_sizes

        # 4 - Build connections between base and control model
        self.down_zero_convs_out = nn.ModuleList([])
        self.down_zero_convs_in = nn.ModuleList([])
        self.middle_block_out = nn.ModuleList([])
        self.middle_block_in = nn.ModuleList([])
        self.up_zero_convs_out = nn.ModuleList([])
        self.up_zero_convs_in = nn.ModuleList([])

        for ch_io_base in self.ch_inout_base["down"]:
            self.down_zero_convs_in.append(self._make_zero_conv(in_channels=ch_io_base[1], out_channels=ch_io_base[1]))
        for i in range(len(self.ch_inout_ctrl["down"])):
            self.down_zero_convs_out.append(
                self._make_zero_conv(self.ch_inout_ctrl["down"][i][1], self.ch_inout_base["down"][i][1])
            )

        self.middle_block_out = self._make_zero_conv(
            self.ch_inout_ctrl["mid"][-1][1], self.ch_inout_base["mid"][-1][1]
        )

        self.up_zero_convs_out.append(
            self._make_zero_conv(self.ch_inout_ctrl["down"][-1][1], self.ch_inout_base["mid"][-1][1])
        )
        for i in range(1, len(self.ch_inout_ctrl["down"])):
            self.up_zero_convs_out.append(
                self._make_zero_conv(self.ch_inout_ctrl["down"][-(i + 1)][1], self.ch_inout_base["up"][i - 1][1])
            )


    def forward(self, sample, encoder_hidden_states, added_cond_kwargs = {}):
        #raise ValueError("A ControlNetXSAddonModel cannot be run by itself. Pass it into a ControlNetXSModel model instead.")

        timestep = 980
        cross_attention_kwargs = {}
        timestep_cond = None

        # # # unet.forward for testing

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

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
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
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
            encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=1.0)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        return sample


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
        base_model_channel_sizes (`Dict[str, List[Tuple[int]]]`):
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
            return ControlNetXSModel.from_unet(
                base_model,
                time_embedding_mix=0.95,
                learn_embedding=True,
                size_ratio=0.1,
                conditioning_embedding_out_channels=(16, 32, 96, 256),
                num_attention_heads=get_dim_attn_heads(base_model, 0.1, 64),
            )
        else:
            return ControlNetXSModel.from_unet(
                base_model,
                time_embedding_mix=1.0,
                learn_embedding=True,
                size_ratio=0.0125,
                conditioning_embedding_out_channels=(16, 32, 96, 256),
                num_attention_heads=get_dim_attn_heads(base_model, 0.0125, 8),
            )

    @classmethod
    def _gather_subblock_sizes(cls, unet: UNet2DConditionModel, base_or_control: str):
        """To create correctly sized connections between base and control model, we need to know
        the input and output channels of each subblock.

        Parameters:
            unet (`UNet2DConditionModel`):
                Unet of which the subblock channels sizes are to be gathered.
            base_or_control (`str`):
                Needs to be either "base" or "control". If "base", decoder is also considered.
        """
        if base_or_control not in ["base", "control"]:
            raise ValueError("`base_or_control` needs to be either `base` or `control`")

        channel_sizes = {"down": [], "mid": [], "up": []}

        # input convolution
        channel_sizes["down"].append((unet.conv_in.in_channels, unet.conv_in.out_channels))

        # encoder blocks
        for module in unet.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    channel_sizes["down"].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    channel_sizes["down"].append(
                        (module.downsamplers[0].channels, module.downsamplers[0].out_channels)
                    )
            else:
                raise ValueError(f"Encountered unknown module of type {type(module)} while creating ControlNet-XS.")

        # middle block
        channel_sizes["mid"].append((unet.mid_block.resnets[0].in_channels, unet.mid_block.resnets[0].out_channels))

        # decoder blocks
        if base_or_control == "base":
            for module in unet.up_blocks:
                if isinstance(module, (CrossAttnUpBlock2D, UpBlock2D)):
                    for r in module.resnets:
                        channel_sizes["up"].append((r.in_channels, r.out_channels))
                else:
                    raise ValueError(
                        f"Encountered unknown module of type {type(module)} while creating ControlNet-XS."
                    )

        return channel_sizes

    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        controlnet_conditioning_channel_order: str = "rgb",
        time_embedding_input_dim: int = 320,
        time_embedding_dim: int = 1280,
        time_embedding_mix: float = 1.0,
        learn_embedding: bool = False,
        base_model_channel_sizes: Dict[str, List[Tuple[int]]] = {
            "down": [
                (4, 320),
                (320, 320),
                (320, 320),
                (320, 320),
                (320, 640),
                (640, 640),
                (640, 640),
                (640, 1280),
                (1280, 1280),
            ],
            "mid": [(1280, 1280)],
            "up": [
                (2560, 1280),
                (2560, 1280),
                (1920, 1280),
                (1920, 640),
                (1280, 640),
                (960, 640),
                (960, 320),
                (640, 320),
                (640, 320),
            ],
        },
        sample_size: Optional[int] = None,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = 8,
        upcast_attention: bool = False,
    ):
        super().__init__()

        # 1 - Create control unet
        self.control_model = UNet2DConditionModel(
            sample_size=sample_size,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            attention_head_dim=num_attention_heads,
            use_linear_projection=True,
            upcast_attention=upcast_attention,
            time_embedding_dim=time_embedding_dim,
        )

        # 5 - Create conditioning hint embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        controlnet_conditioning_channel_order: str = "rgb",
        learn_embedding: bool = False,
        time_embedding_mix: float = 1.0,
        block_out_channels: Optional[Tuple[int]] = None,
        size_ratio: Optional[float] = None,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = 8,
        norm_num_groups: Optional[int] = None,
    ):
        r"""
        Instantiate a [`ControlNetXSModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model we want to control. The dimensions of the ControlNetXSModel will be adapted to it.
            conditioning_channels (`int`, defaults to 3):
                Number of channels of conditioning input (e.g. an image)
            conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
                The tuple of output channel for each block in the `controlnet_cond_embedding` layer.
            controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
                The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
            learn_embedding (`bool`, defaults to `False`):
                Wether to use time embedding of the control model. If yes, the time embedding is a linear interpolation
                of the time embeddings of the control and base model with interpolation parameter
                `time_embedding_mix**3`.
            time_embedding_mix (`float`, defaults to 1.0):
                Linear interpolation parameter used if `learn_embedding` is `True`.
            block_out_channels (`Tuple[int]`, *optional*):
                Down blocks output channels in control model. Either this or `size_ratio` must be given.
            size_ratio (float, *optional*):
                When given, block_out_channels is set to a relative fraction of the base model's block_out_channels.
                Either this or `block_out_channels` must be given.
            num_attention_heads (`Union[int, Tuple[int]]`, *optional*):
                The dimension of the attention heads. The naming seems a bit confusing and it is, see https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 for why.
            norm_num_groups (int, *optional*, defaults to `None`):
                The number of groups to use for the normalization of the control unet. If `None`,
                `int(unet.config.norm_num_groups * size_ratio)` is taken.
        """

        # Check input
        fixed_size = block_out_channels is not None
        relative_size = size_ratio is not None
        if not (fixed_size ^ relative_size):
            raise ValueError(
                "Pass exactly one of `block_out_channels` (for absolute sizing) or `control_model_ratio` (for relative sizing)."
            )

        # Create model
        if block_out_channels is None:
            block_out_channels = [int(size_ratio * c) for c in unet.config.block_out_channels]

        # Check that attention heads and group norms match channel sizes
        # - attention heads
        def attn_heads_match_channel_sizes(attn_heads, channel_sizes):
            if isinstance(attn_heads, (tuple, list)):
                return all(c % a == 0 for a, c in zip(attn_heads, channel_sizes))
            else:
                return all(c % attn_heads == 0 for c in channel_sizes)

        num_attention_heads = num_attention_heads or unet.config.attention_head_dim
        if not attn_heads_match_channel_sizes(num_attention_heads, block_out_channels):
            raise ValueError(
                f"The dimension of attention heads ({num_attention_heads}) must divide `block_out_channels` ({block_out_channels}). If you didn't set `num_attention_heads` the default settings don't match your model. Set `num_attention_heads` manually."
            )

        # - group norms
        def group_norms_match_channel_sizes(num_groups, channel_sizes):
            return all(c % num_groups == 0 for c in channel_sizes)

        if norm_num_groups is None:
            if group_norms_match_channel_sizes(unet.config.norm_num_groups, block_out_channels):
                norm_num_groups = unet.config.norm_num_groups
            else:
                norm_num_groups = min(block_out_channels)

                if group_norms_match_channel_sizes(norm_num_groups, block_out_channels):
                    print(
                        f"`norm_num_groups` was set to `min(block_out_channels)` (={norm_num_groups}) so it divides all block_out_channels` ({block_out_channels}). Set it explicitly to remove this information."
                    )
                else:
                    raise ValueError(
                        f"`block_out_channels` ({block_out_channels}) don't match the base models `norm_num_groups` ({unet.config.norm_num_groups}). Setting `norm_num_groups` to `min(block_out_channels)` ({norm_num_groups}) didn't fix this. Pass `norm_num_groups` explicitly so it divides all block_out_channels."
                    )

        def get_time_emb_input_dim(unet: UNet2DConditionModel):
            return unet.time_embedding.linear_1.in_features

        def get_time_emb_dim(unet: UNet2DConditionModel):
            return unet.time_embedding.linear_2.out_features

        # Clone params from base unet if
        #    (i)   it's required to build SD or SDXL, and
        #    (ii)  it's not used for the time embedding (as time embedding of control model is never used), and
        #    (iii) it's not set further below anyway
        to_keep = [
            "cross_attention_dim",
            "down_block_types",
            "sample_size",
            "transformer_layers_per_block",
            "up_block_types",
            "upcast_attention",
        ]
        kwargs = {k: v for k, v in dict(unet.config).items() if k in to_keep}
        kwargs.update(block_out_channels=block_out_channels)
        kwargs.update(num_attention_heads=num_attention_heads)
        kwargs.update(norm_num_groups=norm_num_groups)

        # Add controlnetxs-specific params
        kwargs.update(
            conditioning_channels=conditioning_channels,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            time_embedding_input_dim=get_time_emb_input_dim(unet),
            time_embedding_dim=get_time_emb_dim(unet),
            time_embedding_mix=time_embedding_mix,
            learn_embedding=learn_embedding,
            base_model_channel_sizes=ControlNetXSModel._gather_subblock_sizes(unet, base_or_control="base"),
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        return cls(**kwargs)

    def forward(
        self,
        base_model: UNet2DConditionModel,
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
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # scale control strength
        n_connections = len(self.down_zero_convs_out) + 1 + len(self.up_zero_convs_out)
        scale_list = torch.full((n_connections,), conditioning_scale)

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

        t_emb = base_model.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.learn_embedding:
            ctrl_temb = self.control_model.time_embedding(t_emb, timestep_cond)
            base_temb = base_model.time_embedding(t_emb, timestep_cond)
            interpolation_param = self.config.time_embedding_mix**0.3

            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = base_model.time_embedding(t_emb)

        # added time & text embeddings
        aug_emb = None

        if base_model.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if base_model.config.class_embed_type == "timestep":
                class_labels = base_model.time_proj(class_labels)

            class_emb = base_model.class_embedding(class_labels).to(dtype=self.dtype)
            temb = temb + class_emb

        if base_model.config.addition_embed_type is not None:
            if base_model.config.addition_embed_type == "text":
                aug_emb = base_model.add_embedding(encoder_hidden_states)
            elif base_model.config.addition_embed_type == "text_image":
                raise NotImplementedError()
            elif base_model.config.addition_embed_type == "text_time":
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
                time_embeds = base_model.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(temb.dtype)
                aug_emb = base_model.add_embedding(add_embeds)
            elif base_model.config.addition_embed_type == "image":
                raise NotImplementedError()
            elif base_model.config.addition_embed_type == "image_hint":
                raise NotImplementedError()

        temb = temb + aug_emb if aug_emb is not None else temb

        # text embeddings
        cemb = encoder_hidden_states

        # Preparation
        guided_hint = self.controlnet_cond_embedding(controlnet_cond)

        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []
        it_down_convs_in, it_down_convs_out, it_dec_convs_in, it_up_convs_out = map(
            iter, (self.down_zero_convs_in, self.down_zero_convs_out, self.up_zero_convs_in, self.up_zero_convs_out)
        )
        scales = iter(scale_list)

        base_down_subblocks = to_sub_blocks(base_model.down_blocks)
        ctrl_down_subblocks = to_sub_blocks(self.control_model.down_blocks)
        base_mid_subblocks = to_sub_blocks([base_model.mid_block])
        ctrl_mid_subblocks = to_sub_blocks([self.control_model.mid_block])
        base_up_subblocks = to_sub_blocks(base_model.up_blocks)

        # Cross Control
        # 0 - conv in
        h_base = base_model.conv_in(h_base)
        h_ctrl = self.control_model.conv_in(h_ctrl)
        if guided_hint is not None:
            h_ctrl += guided_hint
        h_base = h_base + next(it_down_convs_out)(h_ctrl) * next(scales)  # D - add ctrl -> base

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        # 1 - down
        for m_base, m_ctrl in zip(base_down_subblocks, ctrl_down_subblocks):
            h_ctrl = torch.cat([h_ctrl, next(it_down_convs_in)(h_base)], dim=1)  # A - concat base -> ctrl
            h_base = m_base(h_base, temb, cemb, attention_mask, cross_attention_kwargs)  # B - apply base subblock
            h_ctrl = m_ctrl(h_ctrl, temb, cemb, attention_mask, cross_attention_kwargs)  # C - apply ctrl subblock
            h_base = h_base + next(it_down_convs_out)(h_ctrl) * next(scales)  # D - add ctrl -> base
            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)

        # 2 - mid
        h_ctrl = torch.cat([h_ctrl, next(it_down_convs_in)(h_base)], dim=1)  # A - concat base -> ctrl
        for m_base, m_ctrl in zip(base_mid_subblocks, ctrl_mid_subblocks):
            h_base = m_base(h_base, temb, cemb, attention_mask, cross_attention_kwargs)  # B - apply base subblock
            h_ctrl = m_ctrl(h_ctrl, temb, cemb, attention_mask, cross_attention_kwargs)  # C - apply ctrl subblock
        h_base = h_base + self.middle_block_out(h_ctrl) * next(scales)  # D - add ctrl -> base

        # 3 - up
        for i, m_base in enumerate(base_up_subblocks):
            h_base = h_base + next(it_up_convs_out)(hs_ctrl.pop()) * next(scales)  # add info from ctrl encoder
            h_base = torch.cat([h_base, hs_base.pop()], dim=1)  # concat info from base encoder+ctrl encoder
            h_base = m_base(h_base, temb, cemb, attention_mask, cross_attention_kwargs)

        h_base = base_model.conv_norm_out(h_base)
        h_base = base_model.conv_act(h_base)
        h_base = base_model.conv_out(h_base)

        if not return_dict:
            return h_base

        return ControlNetXSOutput(sample=h_base)

    def _make_zero_conv(self, in_channels, out_channels=None):
        # keep running track of channels sizes
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


def increase_block_input_in_mid_resnet(unet: UNet2DConditionModel, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    m = unet.mid_block.resnets[0]
    old_norm1, old_conv1 = m.norm1, m.conv1
    # norm
    norm_args = "num_groups num_channels eps affine".split(" ")
    for a in norm_args:
        assert hasattr(old_norm1, a)
    norm_kwargs = {a: getattr(old_norm1, a) for a in norm_args}
    norm_kwargs["num_channels"] += by  # surgery done here
    # conv1
    conv1_args = (
        "in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer".split(" ")
    )
    for a in conv1_args:
        assert hasattr(old_conv1, a)
    conv1_kwargs = {a: getattr(old_conv1, a) for a in conv1_args}
    conv1_kwargs["bias"] = "bias" in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs["in_channels"] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work)
    conv_shortcut_args_kwargs = {
        "in_channels": conv1_kwargs["in_channels"],
        "out_channels": conv1_kwargs["out_channels"],
        # default arguments from resnet.__init__
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "bias": True,
    }
    # swap old with new modules
    unet.mid_block.resnets[0].norm1 = GroupNorm(**norm_kwargs)
    unet.mid_block.resnets[0].conv1 = LoRACompatibleConv(**conv1_kwargs)
    unet.mid_block.resnets[0].conv_shortcut = LoRACompatibleConv(**conv_shortcut_args_kwargs)
    unet.mid_block.resnets[0].in_channels += by  # surgery done here


def adjust_group_norms(unet: UNet2DConditionModel, max_num_group: int = 32):
    def find_denominator(number, start):
        if start >= number:
            return number
        while start != 0:
            residual = number % start
            if residual == 0:
                return start
            start -= 1

    for block in [*unet.down_blocks, unet.mid_block]:
        # resnets
        for r in block.resnets:
            if r.norm1.num_groups < max_num_group:
                r.norm1.num_groups = find_denominator(r.norm1.num_channels, start=max_num_group)

            if r.norm2.num_groups < max_num_group:
                r.norm2.num_groups = find_denominator(r.norm2.num_channels, start=max_num_group)

        # transformers
        if hasattr(block, "attentions"):
            for a in block.attentions:
                if a.norm.num_groups < max_num_group:
                    a.norm.num_groups = find_denominator(a.norm.num_channels, start=max_num_group)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module



class DownSubBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = None,
        transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,
        num_attention_heads: Optional[int] = 1,
        cross_attention_dim: Optional[int] = 1024,
        upcast_attention: Optional[bool] = False,
        has_resnet = False,
        has_crossattn = False,
        has_downsampler = False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if has_resnet:
            self.resnet = ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=1e-5,
            )
        else:
            self.resnet = None

        if has_crossattn:
            self.attention = Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block,
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=True,
                upcast_attention=upcast_attention,
            )
        else:
            self.attention = None

        if has_downsampler:
            self.downsampler = Downsample2D(out_channels, use_conv=True, out_channels=out_channels, name="op")
        else:
            self.downsampler = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        # todo

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
            hidden_states = self.attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = self.resnet(hidden_states, temb, scale=lora_scale)
            hidden_states = self.attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        return hidden_states