# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.utils.import_utils import is_xformers_available


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer2DModel(ModelMixin, ConfigMixin):
    """
    Transformer block for image-like data. Takes either discrete (classes of vector embeddings) or continuous (actual
    embeddings) inputs.

    When input is continuous: First, project the input (aka embedding) and reshape to b, t, d. Then apply standard
    transformer action. Finally, reshape to image.

    When input is discrete: First, input (classes of latent pixels) is converted to embeddings and has positional
    embeddings applied, see `ImagePositionalEmbeddings`. Then apply standard transformer action. Finally, predict
    classes of unnoised image.

    Note that it is assumed one of the input classes is the masked latent pixel. The predicted classes of the unnoised
    image do not contain a prediction for the masked pixel as the unnoised image cannot be masked.

    Parameters:
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        in_channels (:
            obj:`int`, *optional*): Pass if the input is continuous. The number of channels in the input and output.
        depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
        discrete (:
            obj:`bool`, *optional*, defaults to False): Set to True if the input is discrete i.e. over classes of
            vector embeddings for each pixel. See the beginning of the docstring for a more in-depth description.
        height (:obj:`int`, *optional*): Pass if the input is discrete. The height of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        width (:obj:`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_embed (:
            obj:`int`, *optional*): Pass if the input is discrete. The number of classes of the vector embeddings of
            the latent pixels. Includes the class for the masked latent pixel.
        ff_layers (:obj:,`List[Literal["Dropout", "Linear", "ApproximateGELU", "GEGLU"]]` *optional*):
            The layers to use in the TransformerBlocks' FeedForward block.
        norm_layers (:obj: `List[Literal["LayerNorm", "AdaLayerNorm"]]`, *optional*):
            The norm layers to use for the TransformerBlocks.
        num_embeds_ada_norm (:obj: `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (:
            obj: `bool`, *optional*): Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        n_heads: int,
        d_head: int,
        in_channels: Optional[int] = None,
        depth: int = 1,
        dropout: float = 0.0,
        num_groups: int = 32,
        context_dim: Optional[int] = None,
        discrete: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_embed: Optional[int] = None,
        ff_layers: Optional[List[str]] = None,
        norm_layers: Optional[List[str]] = None,
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: Optional[bool] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner_dim = n_heads * d_head

        self.discrete = discrete

        if self.discrete:
            assert height is not None, "Transformer2DModel over discrete input must provide height"
            assert width is not None, "Transformer2DModel over discrete input must provide width"
            assert num_embed is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = height
            self.width = width
            self.num_embed = num_embed
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=self.num_embed, embed_dim=inner_dim, height=self.height, width=self.width
            )
        else:
            assert in_channels is not None, "Transformer2DModel over continuous input must provide in_channels"

            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    ff_layers=ff_layers,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_layers=norm_layers,
                )
                for d in range(depth)
            ]
        )

        if self.discrete:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_embed - 1)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, context=None, timestep=None):
        """
        Args:
            hidden_states (:obj: When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            context (:obj: `torch.LongTensor` of shape `(batch size, context dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (:obj: `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
        Returns:
            [`torch.FloatTensor` of shape `(batch size, num embed - 1, num latent pixels)`] if discrete or
            [`torch.FloatTensor` of shape `(batch size, channel, height, width)`] if continuous :
                If discrete, returns probability distributions for the unnoised latent pixels. Note that it does not
                output a prediction for the masked class.
        """
        if self.discrete:
            hidden_states = self.latent_image_embedding(hidden_states)

            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, context=context, timestep=timestep)

            logits = self.out(self.norm_out(hidden_states))
            # (batch, self.num_embed - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            return_value = F.log_softmax(logits.double(), dim=1).float()
        else:
            batch, channel, height, weight = hidden_states.shape
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)

            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, context=context, timestep=timestep)

            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)
            return_value = hidden_states + residual

        return return_value

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        for block in self.transformer_blocks:
            block._set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Parameters:
        channels (:obj:`int`): The number of channels in the input and output.
        num_head_channels (:obj:`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        num_groups (:obj:`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (:obj:`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (:obj:`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, eps=eps, affine=True)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels, 1)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # get scores
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)  # TODO: use baddmm
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
        ff_layers (:obj:,`List[Literal["Dropout", "Linear", "ApproximateGELU", "GEGLU"]]` *optional*):
            The layers to use in the FeedForward block.
        norm_layers (:obj: `List[Literal["LayerNorm", "AdaLayerNorm"]]`, *optional*):
            The norm layers. Must be of length 3. Defaults to `["LayerNorm", "LayerNorm", "LayerNorm"]`
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:obj: `bool`, *optional*): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        ff_layers: Optional[List[str]] = None,
        norm_layers: Optional[List[str]] = None,
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: Optional[bool] = None,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, layers=ff_layers)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            bias=attention_bias,
        )  # is self-attn if context is none

        norm_layers = ["LayerNorm", "LayerNorm", "LayerNorm"] if norm_layers is None else norm_layers

        assert len(norm_layers) == 3, "BasicTransformerBlock only supports 3 norm_layers"

        for idx, norm_layer in enumerate(norm_layers):
            if norm_layer == "LayerNorm":
                norm_layer_ = nn.LayerNorm(dim)
            elif norm_layer == "AdaLayerNorm":
                assert (
                    num_embeds_ada_norm is not None
                ), "When using AdaLayerNorm, you must also pass num_embeds_ada_norm."
                norm_layer_ = AdaLayerNorm(dim, num_embeds_ada_norm)

            if idx == 0:
                self.norm1 = norm_layer_
            elif idx == 1:
                self.norm2 = norm_layer_
            elif idx == 2:
                self.norm3 = norm_layer_

        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, context=None, timestep=None):
        norm1_kwargs = {"timestep": timestep} if self.norm1.__class__ == AdaLayerNorm else {}
        norm2_kwargs = {"timestep": timestep} if self.norm2.__class__ == AdaLayerNorm else {}
        norm3_kwargs = {"timestep": timestep} if self.norm3.__class__ == AdaLayerNorm else {}

        hidden_states = self.attn1(self.norm1(hidden_states, **norm1_kwargs)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states, **norm2_kwargs), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states, **norm3_kwargs)) + hidden_states
        return hidden_states


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (:obj:`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False

        bias = False if bias is None else bias

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        if query.device.type == "mps":
            # Better performance on mps (~20-25%)
            attention_scores = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output

        if query.device.type == "mps":
            hidden_states = torch.einsum("b i j, b j d -> b i d", attention_probs, value)
        else:
            hidden_states = torch.matmul(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            if query.device.type == "mps":
                # Better performance on mps (~20-25%)
                attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx])
                    * self.scale
                )
            else:
                attn_slice = (
                    torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
                )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            if query.device.type == "mps":
                attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
            else:
                attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value):
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        layers (:obj:,`List[Literal["Dropout", "Linear", "ApproximateGELU", "GEGLU"]]` *optional*):
            The list of layers to use. Note that the list must contain exactly two dimension changing layers (Linear
            and GEGLU) but may contain as many non-dimension changing layers as you want (Dropout and ApproximateGELU).
            The first dimension changing layer will project from the input dimension to the hidden dimension. The
            second dimension changing layer will project from the hidden dimension to the output dimension. Defaults to
            `["GEGLU", "Dropout", "Linear"]`.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
        layers: Optional[List[str]] = None,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([])

        layers = ["GEGLU", "Dropout", "Linear"] if layers is None else layers

        dim_idx = 0
        dims = [[dim, inner_dim], [inner_dim, dim_out]]

        error_string = "FeedForward must have exactly two dimension changing layers (Linear and GEGLU)."

        for layer in layers:
            if layer == "Dropout":
                self.net.append(nn.Dropout(dropout))
            elif layer == "Linear":
                assert dim_idx < 2, f"Too many dimension changes. {error_string}"
                self.net.append(nn.Linear(*dims[dim_idx]))
                dim_idx += 1
            elif layer == "ApproximateGELU":
                self.net.append(ApproximateGELU())
            elif layer == "GEGLU":
                assert dim_idx < 2, f"Too many dimension changes. {error_string}"
                self.net.append(GEGLU(*dims[dim_idx]))
                dim_idx += 1

        assert dim_idx == 2, f"Too few dimension changes. {error_string}"

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x
