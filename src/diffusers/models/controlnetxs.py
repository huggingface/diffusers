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
from typing import Any, Dict, List, Optional, Union, Tuple

from itertools import zip_longest

import math
import torch
from torch import nn
from torch.nn.modules.normalization import GroupNorm
import torch.utils.checkpoint

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import UNet2DConditionLoadersMixin
from ..utils import BaseOutput, logging
from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from .embeddings import Timesteps
from .modeling_utils import ModelMixin
from .lora import LoRACompatibleConv
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    CrossAttnUpBlock2D,
    UpBlock2D,
    ResnetBlock2D,
    Transformer2DModel,
    Downsample2D,
    Upsample2D,
)
from .unet_2d_condition import UNet2DConditionModel
from ..umer_debug_logger import udl


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class ControlNetXSOutput(BaseOutput):
    # todo: docstring
    sample: torch.FloatTensor = None


class ControlNetXSModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    """A ControlNet-XS model."""

    # to delete later
    @classmethod
    def create_as_in_paper(cls, base_model: UNet2DConditionModel):

        def get_time_emb_dim(unet: UNet2DConditionModel): return unet.time_embedding.linear_2.out_features
        def get_time_emb_input_dim(unet: UNet2DConditionModel):return unet.time_embedding.linear_1.in_features

        base_model_channel_sizes = ControlNetXSModel.gather_base_model_sizes(base_model, base_or_control='base')

        control_model_ratio = 0.1

        block_out_channels = [int(c*control_model_ratio)for c in base_model.config.block_out_channels]
        dim_attention_heads = 64
        num_attention_heads = [math.ceil(c/dim_attention_heads) for c in block_out_channels]

        cnxs_model = cls(
            conditioning_channels=3,
            block_out_channels=block_out_channels,
            down_block_types=base_model.config.down_block_types,
            up_block_types=base_model.config.up_block_types,
            time_embedding_dim=get_time_emb_dim(base_model),
            time_embedding_input_dim=get_time_emb_input_dim(base_model),
            layers_per_block=base_model.config.layers_per_block,
            transformer_layers_per_block=base_model.config.transformer_layers_per_block,
            cross_attention_dim=base_model.config.cross_attention_dim,
            learn_embedding=True,
            base_model_channel_sizes=base_model_channel_sizes,
            addition_embed_type=base_model.config.addition_embed_type,
            num_attention_heads=num_attention_heads,
        )
        cnxs_model.base_model = base_model
        return cnxs_model

    @classmethod
    def gather_base_model_sizes(cls, unet: UNet2DConditionModel, base_or_control):
        if base_or_control not in ['base', 'control']:
            raise ValueError(f"`base_or_control` needs to be either `base` or `control`")

        channel_sizes = {'enc': [], 'mid': [], 'dec': []}

        # input convolution
        channel_sizes['enc'].append((unet.conv_in.in_channels, unet.conv_in.out_channels))

        # encoder blocks
        for module in unet.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    channel_sizes['enc'].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    channel_sizes['enc'].append((module.downsamplers[0].channels, module.downsamplers[0].out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')

        # middle block
        channel_sizes['mid'].append((unet.mid_block.resnets[0].in_channels, unet.mid_block.resnets[0].out_channels))

        # decoder blocks
        if base_or_control == 'base':
            for module in unet.up_blocks:
                if isinstance(module, (CrossAttnUpBlock2D, UpBlock2D)):
                    for r in module.resnets:
                        channel_sizes['dec'].append((r.in_channels, r.out_channels))
                else:
                   raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')

        return channel_sizes

    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str]=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
        up_block_types: Tuple[str]=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int]=(32,64,128),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        time_embedding_dim=1280,
        time_embedding_input_dim=320,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int]]=(0,2,10),
        base_model_channel_sizes: Dict[str, List[Tuple[int]]]={
            'enc': [(4, 320), (320, 320), (320, 320), (320, 320), (320, 640), (640, 640), (640, 640), (640, 1280), (1280, 1280)],
            'mid': [(1280, 1280)],
            'dec': [(2560, 1280), (2560, 1280), (1920, 1280), (1920, 640), (1280, 640), (960, 640), (960, 320), (640, 320), (640, 320)]
        },
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        global_pool_conditions: bool = False,
        time_control_scale:float=1.0,
        learn_embedding: bool =False,
        addition_embed_type: Optional[str] = None,
    ):
        super().__init__()

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # 1 - Create control unet
        self.control_model = UNet2DConditionModel(
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            time_embedding_dim=time_embedding_dim,
            layers_per_block=layers_per_block,
            transformer_layers_per_block=transformer_layers_per_block,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=num_attention_heads, 
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
        )

        # 2 - Do model surgery on control model
        # 2.1 - Allow to use the same time information as the base model
        adjust_time_input_dim(self.control_model, time_embedding_input_dim)
 
        # 2.2 - Allow for information infusion from base model
        base_block_out_channels = [sz[1] for sz in base_model_channel_sizes['enc'] if sz[0] != sz[1]]

        extra_channels = list(zip(
            base_block_out_channels[0:1] + base_block_out_channels[:-1],
            base_block_out_channels
        ))
        for i, (e1, e2) in enumerate(extra_channels):
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=0, by=e1)
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=1, by=e2)
            if self.control_model.down_blocks[i].downsamplers: increase_block_input_in_encoder_downsampler(self.control_model, block_no=i, by=e2)
        increase_block_input_in_mid_resnet(self.control_model, by=base_block_out_channels[-1])

        # 3 - Gather Channel Sizes
        self.ch_inout_ctrl = ControlNetXSModel.gather_base_model_sizes(self.control_model, base_or_control='control')
        self.ch_inout_base = base_model_channel_sizes

        # 4 - Build connections between base and control model
        self.enc_zero_convs_out = nn.ModuleList([])
        self.enc_zero_convs_in = nn.ModuleList([])
        self.middle_block_out = nn.ModuleList([])
        self.middle_block_in = nn.ModuleList([])
        self.dec_zero_convs_out = nn.ModuleList([])
        self.dec_zero_convs_in = nn.ModuleList([])

        for ch_io_base in self.ch_inout_base['enc']:
            self.enc_zero_convs_in.append(self.make_zero_conv(
                in_channels=ch_io_base[1], out_channels=ch_io_base[1])
            )
        for i in range(len(self.ch_inout_ctrl['enc'])):
            self.enc_zero_convs_out.append(
                self.make_zero_conv(self.ch_inout_ctrl['enc'][i][1], self.ch_inout_base['enc'][i][1])
            )       
 
        self.middle_block_out = self.make_zero_conv(self.ch_inout_ctrl['mid'][-1][1], self.ch_inout_base['mid'][-1][1])
        
        self.dec_zero_convs_out.append(
            self.make_zero_conv(self.ch_inout_ctrl['enc'][-1][1], self.ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(self.ch_inout_ctrl['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(self.ch_inout_ctrl['enc'][-(i + 1)][1], self.ch_inout_base['dec'][i - 1][1])
            )

        # 5 - Create conditioning hint embedding
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, block_out_channels[0], 3, padding=1))
        )
    
        # 6 - Create time embedding
        self.time_proj = Timesteps(time_embedding_input_dim, flip_sin_to_cos, freq_shift)

        # In the mininal implementation setting, we only need the control model up to the mid block
        del self.control_model.up_blocks
        del self.control_model.conv_norm_out
        del self.control_model.conv_out

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        block_out_channels: Optional[Tuple[int]] = None,
        control_size: Optional[float] = 0.1
    ):
        r"""
        Instantiate a [`ControlNetXSModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model whose configuration are copief to the [`ControlNetXSModel`].
        """

        fixed_size = block_out_channels is not None
        relative_size = control_size is not None

        if not (fixed_size ^ relative_size):
            raise ValueError("Exactly one of `block_out_channels` (for absolute sizing) or `control_size` (for relative sizing) must be given to create a controlnetxs model from a unet.")

        if block_out_channels is None:
            block_out_channels = [control_size*c for c in unet.config.block_out_channels]

        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        base_model_channel_sizes = ControlNetXSModel.gather_base_model_sizes(unet, base_or_control='base')

        controlnet = cls(
            base_model_channel_sizes=base_model_channel_sizes,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
        )

        return controlnet

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
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
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
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
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
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

        self.set_attn_processor(processor, _remove_lora=True)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    # Copied from diffusers.models.controlnet.ControlNetModel._set_gradient_checkpointing
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

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
        guess_mode: bool = False, # todo: understand and implement if required
        return_dict: bool = True,
    ) -> Union[ControlNetXSOutput, Tuple]:
        if self.base_model is None:
            raise RuntimeError("To use `forward`, first set the base model for this ControlNetXSModel via `cnxs_model.base_model = the_base_model`")

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
        n_connections = len(self.enc_zero_convs_out) + 1 + len(self.dec_zero_convs_out)
        scale_list = torch.full((n_connections,), conditioning_scale)
   
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps=timestep
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

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.learn_embedding:
            temb = self.control_model.time_embedding(t_emb) * self.config.time_control_scale ** 0.3 + self.base_model.time_embedding(t_emb) * (1 - self.config.time_control_scale ** 0.3)
        else:
            temb = self.base_model.time_embedding(t_emb)

        # added time & text embeddings
        aug_emb = None
        if self.config.addition_embed_type == "text":
            aug_emb = self.base_model.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            raise NotImplementedError()
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
            time_embeds = self.base_model.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(temb.dtype)
            aug_emb = self.base_model.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            raise NotImplementedError()
        elif self.config.addition_embed_type == "image_hint":
            raise NotImplementedError()

        temb = temb + aug_emb if aug_emb is not None else temb

        # text embeddings
        cemb = encoder_hidden_states

        # Preparation
        guided_hint = self.input_hint_block(controlnet_cond)

        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []
        it_enc_convs_in, it_enc_convs_out, it_dec_convs_in, it_dec_convs_out = map(iter, (self.enc_zero_convs_in, self.enc_zero_convs_out, self.dec_zero_convs_in, self.dec_zero_convs_out))
        scales = iter(scale_list)

        base_down_subblocks = to_sub_blocks(self.base_model.down_blocks)
        ctrl_down_subblocks = to_sub_blocks(self.control_model.down_blocks)
        base_mid_subblocks = to_sub_blocks([self.base_model.mid_block])
        ctrl_mid_subblocks = to_sub_blocks([self.control_model.mid_block])
        base_up_subblocks = to_sub_blocks(self.base_model.up_blocks)

        # Cross Control
        # 0 - conv in
        h_base = self.base_model.conv_in(h_base)
        h_ctrl = self.control_model.conv_in(h_ctrl)
        if guided_hint is not None: h_ctrl += guided_hint
        h_base = h_base + next(it_enc_convs_out)(h_ctrl) * next(scales)

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        # 1 - down
        for m_base, m_ctrl  in zip(base_down_subblocks, ctrl_down_subblocks):
            h_ctrl = torch.cat([h_ctrl, next(it_enc_convs_in)(h_base)], dim=1)  # A - concat base -> ctrl
            h_base = m_base(                                                    # B - apply base subblock
                h_base, temb, cemb,
                attention_mask, cross_attention_kwargs
            )                                 
            h_ctrl = m_ctrl(                                                    # C - apply ctrl subblock
                h_ctrl, temb, cemb,
                attention_mask, cross_attention_kwargs
            )                             
            h_base = h_base + next(it_enc_convs_out)(h_ctrl) * next(scales)     # D - add ctrl -> base

            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)

        # 2 - mid
        h_ctrl = torch.cat([h_ctrl, next(it_enc_convs_in)(h_base)], dim=1)      # A - concat base -> ctrl
        for m_base, m_ctrl in zip(base_mid_subblocks, ctrl_mid_subblocks):
            h_base = m_base(                                                    # B - apply base subblock
                h_base, temb, cemb,
                attention_mask, cross_attention_kwargs
            )  
            h_ctrl  = m_ctrl(                                                   # C - apply ctrl subblock
                h_ctrl, temb, cemb,
                attention_mask, cross_attention_kwargs
            )  
        h_base = h_base + self.middle_block_out(h_ctrl) * next(scales)          # D - add ctrl -> base
 
        # 3 - up
        for m_base in base_up_subblocks:
            h_base = h_base + next(it_dec_convs_out)(hs_ctrl.pop()) * next(scales)  # add info from ctrl encoder 
            h_base = torch.cat([h_base, hs_base.pop()], dim=1)                      # concat info from base encoder+ctrl encoder
            h_base = m_base(                                                   
                h_base, temb, cemb,
                attention_mask, cross_attention_kwargs
            )  

        h_base = self.base_model.conv_norm_out(h_base)
        h_base = self.base_model.conv_act(h_base)
        h_base = self.base_model.conv_out(h_base)

        if not return_dict:
            return h_base
        
        return ControlNetXSOutput(sample=h_base)

    def make_zero_conv(self, in_channels, out_channels=None):
        # keep running track # todo: better comment
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


class EmbedSequential(nn.ModuleList):
    """Sequential module passing embeddings (time and conditioning) to children if they support it."""
    def __init__(self,ms,*args,**kwargs):
        if not is_iterable(ms): ms = [ms]
        super().__init__(ms,*args,**kwargs)
    
    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cemb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for m in self:
            if isinstance(m,ResnetBlock2D):
                x = m(x,temb)
            elif isinstance(m,Transformer2DModel):
                x = m(x,cemb,attention_mask=attention_mask,cross_attention_kwargs=cross_attention_kwargs).sample
            elif isinstance(m,Downsample2D):
                x = m(x)
            elif isinstance(m,Upsample2D):
                x = m(x)
            else:
                raise ValueError(f'Type of m is {type(m)} but should be `ResnetBlock2D`, `Transformer2DModel`,  `Downsample2D` or `Upsample2D`')

        return x


def adjust_time_input_dim(unet: UNet2DConditionModel, dim: int):
    time_emb = unet.time_embedding
    time_emb.linear_1 = nn.Linear(dim, time_emb.linear_1.out_features)


def increase_block_input_in_encoder_resnet(unet:UNet2DConditionModel, block_no, resnet_idx, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    r=unet.down_blocks[block_no].resnets[resnet_idx]
    old_norm1, old_conv1, old_conv_shortcut = r.norm1,r.conv1,r.conv_shortcut
    # norm
    norm_args = 'num_groups num_channels eps affine'.split(' ')
    for a in norm_args: assert hasattr(old_norm1, a)
    norm_kwargs = { a: getattr(old_norm1, a) for a in norm_args }
    norm_kwargs['num_channels'] += by  # surgery done here
    # conv1
    conv1_args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in conv1_args: assert hasattr(old_conv1, a)
    conv1_kwargs = { a: getattr(old_conv1, a) for a in conv1_args }
    conv1_kwargs['bias'] = 'bias' in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs['in_channels'] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work) 
    conv_shortcut_args_kwargs = { 
        'in_channels': conv1_kwargs['in_channels'],
        'out_channels': conv1_kwargs['out_channels'],
        # default arguments from resnet.__init__
        'kernel_size':1, 
        'stride':1, 
        'padding':0,
        'bias':True
    }
    # swap old with new modules
    unet.down_blocks[block_no].resnets[resnet_idx].norm1 = GroupNorm(**norm_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].conv1 = LoRACompatibleConv(**conv1_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].conv_shortcut = LoRACompatibleConv(**conv_shortcut_args_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].in_channels += by  # surgery done here


def increase_block_input_in_encoder_downsampler(unet:UNet2DConditionModel, block_no, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    old_down=unet.down_blocks[block_no].downsamplers[0].conv
    # conv1
    args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in args: assert hasattr(old_down, a)
    kwargs = { a: getattr(old_down, a) for a in args}
    kwargs['bias'] = 'bias' in kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    kwargs['in_channels'] += by  # surgery done here
    # swap old with new modules
    unet.down_blocks[block_no].downsamplers[0].conv = LoRACompatibleConv(**kwargs)
    unet.down_blocks[block_no].downsamplers[0].channels += by  # surgery done here


def increase_block_input_in_mid_resnet(unet:UNet2DConditionModel, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    m=unet.mid_block.resnets[0]
    old_norm1, old_conv1, old_conv_shortcut = m.norm1,m.conv1,m.conv_shortcut
    # norm
    norm_args = 'num_groups num_channels eps affine'.split(' ')
    for a in norm_args: assert hasattr(old_norm1, a)
    norm_kwargs = { a: getattr(old_norm1, a) for a in norm_args }
    norm_kwargs['num_channels'] += by  # surgery done here
    # conv1
    conv1_args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in conv1_args: assert hasattr(old_conv1, a)
    conv1_kwargs = { a: getattr(old_conv1, a) for a in conv1_args }
    conv1_kwargs['bias'] = 'bias' in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs['in_channels'] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work) 
    conv_shortcut_args_kwargs = { 
        'in_channels': conv1_kwargs['in_channels'],
        'out_channels': conv1_kwargs['out_channels'],
        # default arguments from resnet.__init__
        'kernel_size':1, 
        'stride':1, 
        'padding':0,
        'bias':True
    }
    # swap old with new modules
    unet.mid_block.resnets[0].norm1 = GroupNorm(**norm_kwargs)
    unet.mid_block.resnets[0].conv1 = LoRACompatibleConv(**conv1_kwargs)
    unet.mid_block.resnets[0].conv_shortcut = LoRACompatibleConv(**conv_shortcut_args_kwargs)
    unet.mid_block.resnets[0].in_channels += by  # surgery done here


def is_iterable(o):
    if isinstance(o, str): return False
    try:
        iter(o)
        return True
    except TypeError:
        return False


def to_sub_blocks(blocks):
    if not is_iterable(blocks): blocks = [blocks]
    sub_blocks = []
    for b in blocks:
        current_subblocks = []
        if hasattr(b, 'resnets'):
            if hasattr(b, 'attentions') and b.attentions is not None:
                current_subblocks = list(zip_longest(b.resnets, b.attentions))
                 # if we have 1 more resnets than attentions, let the last subblock only be the resnet, not (resnet, None)
                if current_subblocks[-1][1] is None:
                    current_subblocks[-1] = current_subblocks[-1][0]
            else:
                current_subblocks = list(b.resnets)
        # upsamplers are part of the same block # q: what if we have multiple upsamplers?
        if hasattr(b, 'upsamplers') and b.upsamplers is not None: current_subblocks[-1] = list(current_subblocks[-1]) + list(b.upsamplers)
        # downsamplers are own block
        if hasattr(b, 'downsamplers') and b.downsamplers is not None: current_subblocks.append(list(b.downsamplers))   
        sub_blocks += current_subblocks
    return list(map(EmbedSequential, sub_blocks))


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
