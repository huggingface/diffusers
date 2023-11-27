import math
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .attention_processor import AttentionProcessor, Kandi3AttnProcessor
from .embeddings import TimestepEmbedding
from .modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Kandinsky3UNetOutput(BaseOutput):
    sample: torch.FloatTensor = None


# TODO(Yiyi): This class needs to be removed
def set_default_item(condition, item_1, item_2=None):
    if condition:
        return item_1
    else:
        return item_2


# TODO(Yiyi): This class needs to be removed
def set_default_layer(condition, layer_1, args_1=[], kwargs_1={}, layer_2=torch.nn.Identity, args_2=[], kwargs_2={}):
    if condition:
        return layer_1(*args_1, **kwargs_1)
    else:
        return layer_2(*args_2, **kwargs_2)


# TODO(Yiyi): This class should be removed and be replaced by Timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, type_tensor=None):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Kandinsky3EncoderProj(nn.Module):
    def __init__(self, encoder_hid_dim, cross_attention_dim):
        super().__init__()
        self.projection_linear = nn.Linear(encoder_hid_dim, cross_attention_dim, bias=False)
        self.projection_norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, x):
        x = self.projection_linear(x)
        x = self.projection_norm(x)
        return x


class Kandinsky3UNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        time_embedding_dim: int = 1536,
        groups: int = 32,
        attention_head_dim: int = 64,
        layers_per_block: Union[int, Tuple[int]] = 3,
        block_out_channels: Tuple[int] = (384, 768, 1536, 3072),
        cross_attention_dim: Union[int, Tuple[int]] = 4096,
        encoder_hid_dim: int = 4096,
    ):
        super().__init__()

        # TOOD(Yiyi): Give better name and put into config for the following 4 parameters
        expansion_ratio = 4
        compression_ratio = 2
        add_cross_attention = (False, True, True, True)
        add_self_attention = (False, True, True, True)

        out_channels = in_channels
        init_channels = block_out_channels[0] // 2
        # TODO(Yiyi): Should be replaced with Timesteps class -> make sure that results are the same
        # self.time_proj = Timesteps(init_channels, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.time_proj = SinusoidalPosEmb(init_channels)

        self.time_embedding = TimestepEmbedding(
            init_channels,
            time_embedding_dim,
        )

        self.add_time_condition = Kandinsky3AttentionPooling(
            time_embedding_dim, cross_attention_dim, attention_head_dim
        )

        self.conv_in = nn.Conv2d(in_channels, init_channels, kernel_size=3, padding=1)

        self.encoder_hid_proj = Kandinsky3EncoderProj(encoder_hid_dim, cross_attention_dim)

        hidden_dims = [init_channels] + list(block_out_channels)
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, cross_attention_dim) for is_exist in add_cross_attention]
        num_blocks = len(block_out_channels) * [layers_per_block]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_blocks = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(
            zip(in_out_dims, *layer_params)
        ):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_blocks.append(
                Kandinsky3DownSampleBlock(
                    in_dim,
                    out_dim,
                    time_embedding_dim,
                    text_dim,
                    res_block_num,
                    groups,
                    attention_head_dim,
                    expansion_ratio,
                    compression_ratio,
                    down_sample,
                    self_attention,
                )
            )

        self.up_blocks = nn.ModuleList([])
        for level, ((out_dim, in_dim), res_block_num, text_dim, self_attention) in enumerate(
            zip(reversed(in_out_dims), *rev_layer_params)
        ):
            up_sample = level != 0
            self.up_blocks.append(
                Kandinsky3UpSampleBlock(
                    in_dim,
                    cat_dims.pop(),
                    out_dim,
                    time_embedding_dim,
                    text_dim,
                    res_block_num,
                    groups,
                    attention_head_dim,
                    expansion_ratio,
                    compression_ratio,
                    up_sample,
                    self_attention,
                )
            )

        self.conv_norm_out = nn.GroupNorm(groups, init_channels)
        self.conv_act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1)

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

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self, sample, timestep, encoder_hidden_states=None, encoder_attention_mask=None, return_dict=True):
        # TODO(Yiyi): Clean up the following variables - these names should not be used
        # but instead only the ones that we pass to forward
        x = sample
        context_mask = encoder_attention_mask
        context = encoder_hidden_states

        if not torch.is_tensor(timestep):
            dtype = torch.float32 if isinstance(timestep, float) else torch.int32
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timestep.expand(sample.shape[0])
        time_embed_input = self.time_proj(timestep).to(x.dtype)
        time_embed = self.time_embedding(time_embed_input)

        context = self.encoder_hid_proj(context)

        if context is not None:
            time_embed = self.add_time_condition(time_embed, context, context_mask)

        hidden_states = []
        x = self.conv_in(x)
        for level, down_sample in enumerate(self.down_blocks):
            x = down_sample(x, time_embed, context, context_mask)
            if level != self.num_levels - 1:
                hidden_states.append(x)

        for level, up_sample in enumerate(self.up_blocks):
            if level != 0:
                x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_sample(x, time_embed, context, context_mask)

        x = self.conv_norm_out(x)
        x = self.conv_act_out(x)
        x = self.conv_out(x)

        if not return_dict:
            return (x,)
        return Kandinsky3UNetOutput(sample=x)


class Kandinsky3UpSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        cat_dim,
        out_channels,
        time_embed_dim,
        context_dim=None,
        num_blocks=3,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        up_sample=True,
        self_attention=True,
    ):
        super().__init__()
        up_resolutions = [[None, set_default_item(up_sample, True), None, None]] + [[None] * 4] * (num_blocks - 1)
        hidden_channels = (
            [(in_channels + cat_dim, in_channels)]
            + [(in_channels, in_channels)] * (num_blocks - 2)
            + [(in_channels, out_channels)]
        )
        attentions = []
        resnets_in = []
        resnets_out = []

        self.self_attention = self_attention
        self.context_dim = context_dim

        attentions.append(
            set_default_layer(
                self_attention,
                Kandinsky3AttentionBlock,
                (out_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
                layer_2=nn.Identity,
            )
        )

        for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions):
            resnets_in.append(
                Kandinsky3ResNetBlock(in_channel, in_channel, time_embed_dim, groups, compression_ratio, up_resolution)
            )
            attentions.append(
                set_default_layer(
                    context_dim is not None,
                    Kandinsky3AttentionBlock,
                    (in_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio),
                    layer_2=nn.Identity,
                )
            )
            resnets_out.append(
                Kandinsky3ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio)
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets_in = nn.ModuleList(resnets_in)
        self.resnets_out = nn.ModuleList(resnets_out)

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        for attention, resnet_in, resnet_out in zip(self.attentions[1:], self.resnets_in, self.resnets_out):
            x = resnet_in(x, time_embed)
            if self.context_dim is not None:
                x = attention(x, time_embed, context, context_mask, image_mask)
            x = resnet_out(x, time_embed)

        if self.self_attention:
            x = self.attentions[0](x, time_embed, image_mask=image_mask)
        return x


class Kandinsky3DownSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        context_dim=None,
        num_blocks=3,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        down_sample=True,
        self_attention=True,
    ):
        super().__init__()
        attentions = []
        resnets_in = []
        resnets_out = []

        self.self_attention = self_attention
        self.context_dim = context_dim

        attentions.append(
            set_default_layer(
                self_attention,
                Kandinsky3AttentionBlock,
                (in_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
                layer_2=nn.Identity,
            )
        )

        up_resolutions = [[None] * 4] * (num_blocks - 1) + [[None, None, set_default_item(down_sample, False), None]]
        hidden_channels = [(in_channels, out_channels)] + [(out_channels, out_channels)] * (num_blocks - 1)
        for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions):
            resnets_in.append(
                Kandinsky3ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio)
            )
            attentions.append(
                set_default_layer(
                    context_dim is not None,
                    Kandinsky3AttentionBlock,
                    (out_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio),
                    layer_2=nn.Identity,
                )
            )
            resnets_out.append(
                Kandinsky3ResNetBlock(
                    out_channel, out_channel, time_embed_dim, groups, compression_ratio, up_resolution
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets_in = nn.ModuleList(resnets_in)
        self.resnets_out = nn.ModuleList(resnets_out)

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        if self.self_attention:
            x = self.attentions[0](x, time_embed, image_mask=image_mask)

        for attention, resnet_in, resnet_out in zip(self.attentions[1:], self.resnets_in, self.resnets_out):
            x = resnet_in(x, time_embed)
            if self.context_dim is not None:
                x = attention(x, time_embed, context, context_mask, image_mask)
            x = resnet_out(x, time_embed)
        return x


class Kandinsky3ConditionalGroupNorm(nn.Module):
    def __init__(self, groups, normalized_shape, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        self.context_mlp = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, 2 * normalized_shape))
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

    def forward(self, x, context):
        context = self.context_mlp(context)

        for _ in range(len(x.shape[2:])):
            context = context.unsqueeze(-1)

        scale, shift = context.chunk(2, dim=1)
        x = self.norm(x) * (scale + 1.0) + shift
        return x


# TODO(Yiyi): This class should ideally not even exist, it slows everything needlessly down. I'm pretty
# sure we can delete it and instead just pass an attention_mask
class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim, head_dim=64):
        super().__init__()
        assert out_channels % head_dim == 0
        self.num_heads = out_channels // head_dim
        self.scale = head_dim**-0.5

        # to_q
        self.to_q = nn.Linear(in_channels, out_channels, bias=False)
        # to_k
        self.to_k = nn.Linear(context_dim, out_channels, bias=False)
        # to_v
        self.to_v = nn.Linear(context_dim, out_channels, bias=False)
        processor = Kandi3AttnProcessor()
        self.set_processor(processor)
        # to_out
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(out_channels, out_channels, bias=False))

    def set_processor(self, processor: "AttnProcessor"):  # noqa: F821
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
        )


class Kandinsky3Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, kernel_size=3, norm_groups=32, up_resolution=None):
        super().__init__()
        self.group_norm = Kandinsky3ConditionalGroupNorm(norm_groups, in_channels, time_embed_dim)
        self.activation = nn.SiLU()
        self.up_sample = set_default_layer(
            up_resolution is not None and up_resolution,
            nn.ConvTranspose2d,
            (in_channels, in_channels),
            {"kernel_size": 2, "stride": 2},
        )
        padding = int(kernel_size > 1)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.down_sample = set_default_layer(
            up_resolution is not None and not up_resolution,
            nn.Conv2d,
            (out_channels, out_channels),
            {"kernel_size": 2, "stride": 2},
        )

    def forward(self, x, time_embed):
        x = self.group_norm(x, time_embed)
        x = self.activation(x)
        x = self.up_sample(x)
        x = self.projection(x)
        x = self.down_sample(x)
        return x


class Kandinsky3ResNetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_embed_dim, norm_groups=32, compression_ratio=2, up_resolutions=4 * [None]
    ):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        hidden_channel = max(in_channels, out_channels) // compression_ratio
        hidden_channels = (
            [(in_channels, hidden_channel)] + [(hidden_channel, hidden_channel)] * 2 + [(hidden_channel, out_channels)]
        )
        self.resnet_blocks = nn.ModuleList(
            [
                Kandinsky3Block(in_channel, out_channel, time_embed_dim, kernel_size, norm_groups, up_resolution)
                for (in_channel, out_channel), kernel_size, up_resolution in zip(
                    hidden_channels, kernel_sizes, up_resolutions
                )
            ]
        )
        self.shortcut_up_sample = set_default_layer(
            True in up_resolutions, nn.ConvTranspose2d, (in_channels, in_channels), {"kernel_size": 2, "stride": 2}
        )
        self.shortcut_projection = set_default_layer(
            in_channels != out_channels, nn.Conv2d, (in_channels, out_channels), {"kernel_size": 1}
        )
        self.shortcut_down_sample = set_default_layer(
            False in up_resolutions, nn.Conv2d, (out_channels, out_channels), {"kernel_size": 2, "stride": 2}
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


class Kandinsky3AttentionPooling(nn.Module):
    def __init__(self, num_channels, context_dim, head_dim=64):
        super().__init__()
        self.attention = Attention(context_dim, num_channels, context_dim, head_dim)

    def forward(self, x, context, context_mask=None):
        context = self.attention(context.mean(dim=1, keepdim=True), context, context_mask)
        return x + context.squeeze(1)


class Kandinsky3AttentionBlock(nn.Module):
    def __init__(self, num_channels, time_embed_dim, context_dim=None, norm_groups=32, head_dim=64, expansion_ratio=4):
        super().__init__()
        self.in_norm = Kandinsky3ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.attention = Attention(num_channels, num_channels, context_dim or num_channels, head_dim)

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = Kandinsky3ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        height, width = x.shape[-2:]
        out = self.in_norm(x, time_embed)
        out = out.reshape(x.shape[0], -1, height * width).permute(0, 2, 1)
        context = context if context is not None else out

        if image_mask is not None:
            mask_height, mask_width = image_mask.shape[-2:]
            kernel_size = (mask_height // height, mask_width // width)
            image_mask = F.max_pool2d(image_mask, kernel_size, kernel_size)
            image_mask = image_mask.reshape(image_mask.shape[0], -1)

        out = self.attention(out, context, context_mask, image_mask)
        out = out.permute(0, 2, 1).unsqueeze(-1).reshape(out.shape[0], -1, height, width)
        x = x + out

        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        x = x + out
        return x
