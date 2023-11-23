import torch
from torch import nn, einsum
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn.functional as F
from .attention_processor import AttentionProcessor, Kandi3AttnProcessor
import torch.utils.checkpoint

import math
from torch import nn, einsum
from einops import rearrange, repeat
from torch.nn import Identity
from einops import rearrange
from ..utils import BaseOutput, logging
from .modeling_utils import ModelMixin
from ..configuration_utils import ConfigMixin, register_to_config

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def exist(item):
    return item is not None


def set_default_item(condition, item_1, item_2=None):
    if condition:
        return item_1
    else:
        return item_2


def set_default_layer(condition, layer_1, args_1=[], kwargs_1={}, layer_2=Identity, args_2=[], kwargs_2={}):
    if condition:
        return layer_1(*args_1, **kwargs_1)
    else:
        return layer_2(*args_2, **kwargs_2)


def get_tensor_items(x, pos, broadcast_shape):
    bs = pos.shape[0]
    ndims = len(broadcast_shape[1:])
    x = x.to(pos.device)[pos]
    return x.reshape(bs, *((1,) * ndims))


def local_patching(x, height, width, group_size):
    if group_size > 0:
        x = rearrange(
            x, 'b c (h g1) (w g2) -> b (h w) (g1 g2) c',
            h=height//group_size, w=width//group_size, g1=group_size, g2=group_size
        )
    else:
        x = rearrange(x, 'b c h w -> b (h w) c', h=height, w=width)
    return x


def local_merge(x, height, width, group_size):
    if group_size > 0:
        x = rearrange(
            x, 'b (h w) (g1 g2) c -> b c (h g1) (w g2)',
            h=height//group_size, w=width//group_size, g1=group_size, g2=group_size
        )
    else:
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)
    return x


def global_patching(x, height, width, group_size):
    x = local_patching(x, height, width, height//group_size)
    x = x.transpose(-2, -3)
    return x


def global_merge(x, height, width, group_size):
    x = x.transpose(-2, -3)
    x = local_merge(x, height, width, height//group_size)
    return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x, *args, **kwargs):
        return x


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, type_tensor=None):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Identity1(nn.Module):

    def __init__(self,):
        super().__init__()
    def forward(self, x):
        return x

class ConditionalGroupNorm(nn.Module):

    def __init__(self, groups, normalized_shape, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        self.context_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, 2 * normalized_shape)
        )
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

    def forward(self, x, context):
        context = self.context_mlp(context)
        ndims = ' 1' * len(x.shape[2:])
        context = rearrange(context, f'b c -> b c{ndims}')

        scale, shift = context.chunk(2, dim=1)
        x = self.norm(x) * (scale + 1.) + shift
        return x



class Attention(nn.Module):

    def __init__(self, in_channels, out_channels, context_dim, head_dim=64):
        super().__init__()
        assert out_channels % head_dim == 0
        self.num_heads = out_channels // head_dim
        self.scale = head_dim ** -0.5

        self.to_query = nn.Linear(in_channels, out_channels, bias=False)
        self.to_key = nn.Linear(context_dim, out_channels, bias=False)
        self.to_value = nn.Linear(context_dim, out_channels, bias=False)
        processor = Kandi3AttnProcessor()
        self.set_processor(processor)
        self.output_layer = nn.Linear(out_channels, out_channels, bias=False)
        
        
    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor
        
        
    def forward(self, x, context, context_mask=None, image_mask=None):
        return self.processor(
            self,
            x,
            context=context,
            context_mask=context_mask,
            image_mask=image_mask,
        )

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim, kernel_size=3, norm_groups=32, up_resolution=None):
        super().__init__()
        self.group_norm = ConditionalGroupNorm(norm_groups, in_channels, time_embed_dim)
        self.activation = nn.SiLU()
        self.up_sample = set_default_layer(
            exist(up_resolution) and up_resolution,
            nn.ConvTranspose2d, (in_channels, in_channels), {'kernel_size': 2, 'stride': 2}
        )
        padding = set_default_item(kernel_size == 1, 0, 1)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.down_sample = set_default_layer(
            exist(up_resolution) and not up_resolution,
            nn.Conv2d, (out_channels, out_channels), {'kernel_size': 2, 'stride': 2}
        )

    def forward(self, x, time_embed):
        x = self.group_norm(x, time_embed)
        x = self.activation(x)
        x = self.up_sample(x)
        x = self.projection(x)
        x = self.down_sample(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim, norm_groups=32, compression_ratio=2, up_resolutions=4*[None]
    ):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        hidden_channel = max(in_channels, out_channels) // compression_ratio
        hidden_channels = [(in_channels, hidden_channel)] + [(hidden_channel, hidden_channel)] * 2 + [(hidden_channel, out_channels)]
        self.resnet_blocks = nn.ModuleList([
            Block(in_channel, out_channel, time_embed_dim, kernel_size, norm_groups, up_resolution)
            for (in_channel, out_channel), kernel_size, up_resolution in zip(hidden_channels, kernel_sizes, up_resolutions)
        ])

        self.shortcut_up_sample = set_default_layer(
            True in up_resolutions,
            nn.ConvTranspose2d, (in_channels, in_channels), {'kernel_size': 2, 'stride': 2}
        )
        self.shortcut_projection = set_default_layer(
            in_channels != out_channels,
            nn.Conv2d, (in_channels, out_channels), {'kernel_size': 1}
        )
        self.shortcut_down_sample = set_default_layer(
            False in up_resolutions,
            nn.Conv2d, (out_channels, out_channels), {'kernel_size': 2, 'stride': 2}
        )

    def forward(self, x, time_embed):
        out = x
        for resnet_block in self.resnet_blocks:
            out = resnet_block(out, time_embed)

        x = self.shortcut_up_sample(x)
        x = self.shortcut_projection(x)
        x = self.shortcut_down_sample(x)
        x = x + out
        return x


class AttentionPolling(nn.Module):

    def __init__(self, num_channels, context_dim, head_dim=64):
        super().__init__()
        self.attention = Attention(context_dim, num_channels, context_dim, head_dim)

    def forward(self, x, context, context_mask=None):
        context = self.attention(context.mean(dim=1, keepdim=True), context, context_mask)
        return x + context.squeeze(1)


class AttentionBlock(nn.Module):

    def __init__(self, num_channels, time_embed_dim, context_dim=None, norm_groups=32, head_dim=64, expansion_ratio=4):
        super().__init__()
        self.in_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.attention = Attention(num_channels, num_channels, context_dim or num_channels, head_dim)

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        height, width = x.shape[-2:]
        out = self.in_norm(x, time_embed)
        out = rearrange(out, 'b c h w -> b (h w) c', h=height, w=width)
        context = set_default_item(exist(context), context, out)
        if exist(image_mask):
            mask_height, mask_width = image_mask.shape[-2:]
            kernel_size = (mask_height // height, mask_width // width)
            image_mask_max = image_mask.amax((-1, -2), keepdim=True)
            image_mask = F.max_pool2d(image_mask, kernel_size, kernel_size)
            image_mask = rearrange(image_mask, 'b h w -> b (h w)', h=height, w=width)
        out = self.attention(out, context, context_mask, image_mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=height, w=width)
        x = x + out

        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        x = x + out
        return x


class DownSampleBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim, context_dim=None,
            num_blocks=3, groups=32, head_dim=64, expansion_ratio=4, compression_ratio=2,
            down_sample=True, self_attention=True
    ):
        super().__init__()
        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (in_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
            layer_2=Identity
        )

        up_resolutions = [[None] * 4] * (num_blocks - 1) + [[None, None, set_default_item(down_sample, False), None]]
        hidden_channels = [(in_channels, out_channels)] + [(out_channels, out_channels)] * (num_blocks - 1)
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock,
                    (out_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio),
                    layer_2=Identity
                ),
                ResNetBlock(out_channel, out_channel, time_embed_dim, groups, compression_ratio, up_resolution),
            ]) for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions)
        ])

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        x = self.self_attention_block(x, time_embed, image_mask=image_mask)
        for in_resnet_block, attention, out_resnet_block in self.resnet_attn_blocks:
            x = in_resnet_block(x, time_embed)
            x = attention(x, time_embed, context, context_mask, image_mask)
            x = out_resnet_block(x, time_embed)
        return x


class UpSampleBlock(nn.Module):

    def __init__(
            self, in_channels, cat_dim, out_channels, time_embed_dim, context_dim=None,
            num_blocks=3, groups=32, head_dim=64, expansion_ratio=4, compression_ratio=2,
            up_sample=True, self_attention=True
    ):
        super().__init__()
        up_resolutions = [[None, set_default_item(up_sample, True), None, None]] + [[None] * 4] * (num_blocks - 1)
        hidden_channels = [(in_channels + cat_dim, in_channels)] + [(in_channels, in_channels)] * (num_blocks - 2) + [(in_channels, out_channels)]
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, in_channel, time_embed_dim, groups, compression_ratio, up_resolution),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock,
                    (in_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio),
                    layer_2=Identity
                ),
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio),
            ]) for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions)
        ])

        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (out_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
            layer_2=Identity
        )

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        for in_resnet_block, attention, out_resnet_block in self.resnet_attn_blocks:
            x = in_resnet_block(x, time_embed)
            x = attention(x, time_embed, context, context_mask, image_mask)
            x = out_resnet_block(x, time_embed)
        x = self.self_attention_block(x, time_embed, image_mask=image_mask)
        return x

@dataclass
class UNetKandi3(BaseOutput):

    sample: torch.FloatTensor = None
    

class UNetKandi3(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            model_channels,
            init_channels=None,
            num_channels=3,
            time_embed_dim=None,
            groups=32,
            head_dim=64,
            expansion_ratio=4,
            compression_ratio=2,
            dim_mult=(1, 2, 4, 8),
            num_blocks=(3, 3, 3, 3),
            model_dim=4096,
            context_dim=4096,
            add_cross_attention=(False, True, True, True),
            add_self_attention=(False, True, True, True)
        ):
        super().__init__()
        out_channels = num_channels
        init_channels = init_channels or model_channels
        self.projection_lin = nn.Linear(model_dim, context_dim, bias=False)
        self.projection_ln = nn.LayerNorm(context_dim)
        self.sin_emb = SinusoidalPosEmb(init_channels)
        self.to_time_embed = nn.Sequential(
            Identity1(),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention
                )
            )

        self.up_samples = nn.ModuleList([])
        for level, ((out_dim, in_dim), res_block_num, text_dim, self_attention) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_samples.append(
                UpSampleBlock(
                    in_dim, cat_dims.pop(), out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim,
                    expansion_ratio, compression_ratio, up_sample, self_attention
                )
            )

        self.out_layer = nn.Sequential(
            nn.GroupNorm(groups, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1)
        )
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

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

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(Kandi3AttnProcessor())

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

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    
    def forward(self, x, time, context=None, context_mask=None, image_mask=None, use_projections=False,
                return_dict=True, split_context=False, uncondition_mask_idx=None, control_hidden_states=None):

        context = self.projection_lin(context)
        context = self.projection_ln(context)
        if uncondition_mask_idx is not None:
            context[uncondition_mask_idx] = torch.zeros_like(context[uncondition_mask_idx])
            context_mask[uncondition_mask_idx] = torch.zeros_like(context_mask[uncondition_mask_idx])
            print('test1111')
            
        time_embed = self.to_time_embed(self.sin_emb(time).to(x.dtype))
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask, image_mask)
            if level != self.num_levels - 1:
                hidden_states.append(x)
        
        for level, up_sample in enumerate(self.up_samples):
            if level != 0:
                if control_hidden_states is not None:
                    x = torch.cat([x, hidden_states.pop() + control_hidden_states.pop()], dim=1)
                else:
                    x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_sample(x, time_embed, context, context_mask, image_mask)
        x = self.out_layer(x)
        if not return_dict:
            return (x,)
        return UNetKandi3(sample=x)
    
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class InputHint(nn.Module):
    def __init__(self, hint_channels, model_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(hint_channels, 16, 3, padding=1)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.act2 = nn.SiLU()
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.act3 = nn.SiLU()
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.act4 = nn.SiLU()
        self.conv5 = nn.Conv2d(32, 96, 3, padding=1, stride=2)
        self.act5 = nn.SiLU()
        self.conv6 = nn.Conv2d(96, 96, 3, padding=1)
        self.act6 = nn.SiLU()
        self.conv7 = nn.Conv2d(96, 256, 3, padding=1, stride=2)
        self.act7 = nn.SiLU()
        self.out = zero_module(nn.Conv2d(256, model_channels, 3, padding=1))
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x))
        x = self.out(x)
        return x
        
        

class UNetKandi3Controlnet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 model_channels,
                 init_channels=None,
                 num_channels=3,
                 time_embed_dim=None,
                 groups=32,
                 head_dim=64,
                 expansion_ratio=4,
                 compression_ratio=2,
                 dim_mult=(1, 2, 4, 8),
                 num_blocks=(3, 3, 3, 3),
                 model_dim=4096,
                 context_dim=4096,
                 add_cross_attention=(False, True, True, True),
                 add_self_attention=(False, True, True, True),
                 hint_channels=3
                 ):
        super().__init__()
        out_channels = num_channels
        init_channels = init_channels or model_channels
        self.input_hint_block = InputHint(hint_channels, init_channels)
        self.projection_lin = nn.Linear(model_dim, context_dim, bias=False)
        self.projection_ln = nn.LayerNorm(context_dim)
        self.sin_emb = SinusoidalPosEmb(init_channels)
        self.to_time_embed = nn.Sequential(
            Identity1(),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            if level == 3:
                break
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention
                )
            )
            self.convs.append(zero_module(nn.Conv2d(out_dim, out_dim, 3, padding=1)))

    def encode_context(self, context):
        context = self.projection_lin(context)
        context = self.projection_ln(context)
        return context
    
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

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

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(Kandi3AttnProcessor())

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

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

        
    
    def forward(self, x, time, context=None, context_mask=None, image_mask=None, use_projections=False, return_dict=True, split_context=False, uncondition_mask_idx=None, hint=None):
        if use_projections:
            context = self.projection_lin(context)
            context = self.projection_ln(context)
            if uncondition_mask_idx is not None:
                context[uncondition_mask_idx] = torch.zeros_like(context[uncondition_mask_idx])
                context_mask[uncondition_mask_idx] = torch.zeros_like(context_mask[uncondition_mask_idx])
                print('test1111')
                
        time_embed = self.to_time_embed(self.sin_emb(time).to(x.dtype))
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        x = x + self.input_hint_block(hint)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask, image_mask)
            x = self.convs[level](x)
            hidden_states.append(x)
        return hidden_states


class UNetKandi3Controlnet2(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 model_channels,
                 init_channels=None,
                 num_channels=3,
                 time_embed_dim=None,
                 groups=32,
                 head_dim=64,
                 expansion_ratio=4,
                 compression_ratio=2,
                 dim_mult=(1, 2, 4, 8),
                 num_blocks=(3, 3, 3, 3),
                 model_dim=4096,
                 context_dim=4096,
                 add_cross_attention=(False, True, True, True),
                 add_self_attention=(False, True, True, True),
                 hint_channels=3
                 ):
        super().__init__()
        out_channels = num_channels
        init_channels = init_channels or model_channels
        self.input_hint_block = InputHint(hint_channels, init_channels)
        self.projection_lin = nn.Linear(model_dim, context_dim, bias=False)
        self.projection_ln = nn.LayerNorm(context_dim)
        self.sin_emb = SinusoidalPosEmb(init_channels)
        self.to_time_embed = nn.Sequential(
            Identity1(),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention
                )
            )

        self.up_samples = nn.ModuleList([])
        for level, ((out_dim, in_dim), res_block_num, text_dim, self_attention) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_samples.append(
                UpSampleBlock(
                    in_dim, cat_dims.pop(), out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim,
                    expansion_ratio, compression_ratio, up_sample, self_attention
                )
            )

        self.out_layer = nn.Sequential(
            nn.GroupNorm(groups, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1)
        )
        
        

    

    def forward(self, x, time, context=None, context_mask=None, image_mask=None, use_projections=False,
                return_dict=True, split_context=False, uncondition_mask_idx=None, control_hidden_states=None, hint=None):
        if use_projections:
            context = self.projection_lin(context)
            context = self.projection_ln(context)
            if uncondition_mask_idx is not None:
                context[uncondition_mask_idx] = torch.zeros_like(context[uncondition_mask_idx])
                context_mask[uncondition_mask_idx] = torch.zeros_like(context_mask[uncondition_mask_idx])
                print('test1111')
            
        time_embed = self.to_time_embed(self.sin_emb(time).to(x.dtype))
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        x = x + self.input_hint_block(hint)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask, image_mask)
            if level != self.num_levels - 1:
                hidden_states.append(x)
        
        for level, up_sample in enumerate(self.up_samples):
            if level != 0:
                if control_hidden_states is not None:
                    x = torch.cat([x, hidden_states.pop() + control_hidden_states.pop()], dim=1)
                else:
                    x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_sample(x, time_embed, context, context_mask, image_mask)
        x = self.out_layer(x)
        if not return_dict:
            return (x,)
        return UNetKandi3(sample=x)
