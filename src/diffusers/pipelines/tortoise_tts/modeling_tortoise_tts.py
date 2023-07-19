import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.attention_processor import AttnProcessor
from ...models.embeddings import TimestepEmbedding, Timesteps
from ...models.resnet import AdaGroupNorm, Upsample2D, Downsample2D, upsample_2d, downsample_2d, partial
from ...utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",  # default, scale_shift, ada_group
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        # changing the Conv2d to Conv1d
        # changing kernel_size=1 from 3 and padding=0
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        # changing the Conv2d to Conv1d
        self.conv2 = torch.nn.Conv1d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            # changing the Conv2d to Conv1d
            self.conv_shortcut = torch.nn.Conv1d(
                in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            # change this line too since we are now dealing with Conv1d
            # temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift


        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


# Note: for now copy __init__ arguments from Attention + relative_pos_embeddings arg
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py#L35
class AttentionBlock(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block=False,
        processor: Optional[AttnProcessor] = None,
        relative_pos_embeddings: bool = False,
    ):
        pass

# The outputs are NOT same now!!! But when they get I will replace Attention layer with this layer.
# mostly copied from the T5 but there are some major differences
# I think its better to use TortoiseTTSDiffusionModelAttention since it compiles with the HF transformer
# design and also it makes it specific to the Diffusion model only(Since CLVP and GPT2 Attentions are slightly different)

class TortoiseTTSDiffusionModelAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        # bias set to True for
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=True)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=True)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=True)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=True)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # scores += position_bias_masked
        scores += (position_bias_masked * 8) # its actually root under the dimension of each attn head will be updated in the final version

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


class TortoiseTTSDiffusionModelSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = TortoiseTTSDiffusionModelAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = nn.GroupNorm(num_groups=32, num_channels=config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        normed_hidden_states = torch.permute(normed_hidden_states, (0, 2, 1))

        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        print(attention_output[0])

        hidden_states = torch.permute(hidden_states, (0, 2, 1))
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class AttnEncoderBlock1D(nn.Module):
    """
    1D U-Net style block with architecture (no down/upsampling)

    ResnetBlock1d => AttentionBlock
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim = 1,
        relative_pos_embeddings: bool = False,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AttentionBlock(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                    relative_pos_embeddings=relative_pos_embeddings,
                )
            )
        
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, upsample_size=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)
        
        return hidden_states, output_states
    

@dataclass
class ConditioningEncoderOutput(BaseOutput):
    """
    The output of [`ConditioningEncoder`].

    Args:
        TODO: fix
        embedding (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    embedding: torch.FloatTensor


class ConditioningEncoder(ModelMixin, ConfigMixin):
    """
    Conditioning encoder for the Tortoise TTS model with architecture

    (input transform) => [AttentionBlock] x num_layers => (output transform)
    """
    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        attention_head_dim: int = 1,
        relative_pos_embeddings: bool = False,
        input_transform: Optional[str] = None,
        input_conv_kernel_size: int = 1,
        input_conv_stride: int = 1,
        input_conv_padding: int = 0,
        input_conv2_hidden_dim: Optional[int] = None,
        output_transform: Optional[str] = None,
        output_num_groups: int = 32,
    ):
        super().__init__()

        if input_transform is None:
            self.input_transform = nn.Identity()
        elif input_transform == "conv":
            self.input_transform = nn.Conv1d(
                in_channels,
                out_channels,
                input_conv_kernel_size,
                stride=input_conv_stride,
                padding=input_conv_padding,
            )
        elif input_transform == "conv2":
            if input_conv2_hidden_dim is None:
                input_conv2_hidden_dim = in_channels
            self.input_transform = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    input_conv2_hidden_dim,
                    input_conv_kernel_size,
                    stride=input_conv_stride,
                    padding=input_conv_padding,
                ),
                nn.Conv1d(
                    input_conv2_hidden_dim,
                    out_channels,
                    input_conv_kernel_size,
                    stride=input_conv_stride,
                    padding=input_conv_padding,
                ),
            )
        else:
            raise ValueError(
                f"`input_transform` {input_transform} is not currently supported."
            )
        
        self.attention = nn.ModuleList(
            [
                AttentionBlock(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    relative_pos_embeddings=relative_pos_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        
        if output_transform is None:
            self.output_transform = nn.Identity()
        elif output_transform == "groupnorm":
            self.output_transform = nn.GroupNorm(output_num_groups, out_channels)
        else:
            raise ValueError(
                f"`output_transform` {output_transform} is not currently supported."
            )
    
    def forward(self, x, return_dict: bool = True):
        x = self.input_transform(x)
        x = self.attention(x)
        x = self.output_transform(x)

        if not return_dict:
            return (x,)
        
        return ConditioningEncoderOutput(embedding=x)


@dataclass
class TortoiseTTSDenoisingModelOutput(BaseOutput):
    """
    The output of [`TortoiseTTSDenoisingModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class TortoiseTTSDenoisingModel(ModelMixin, ConfigMixin):
    """
    The denoising model used in the diffusion portion of the Tortoise TTS model.
    """
    @register_to_config
    def __init__(
        self,
        in_channels: int = 100,
        out_channels: int = 200,
        in_latent_channels: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 8,
        num_latent_cond_layers: int = 4,
        num_timestep_integrator_layers: int = 3,
        num_post_res_blocks: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        attention_head_dim: int = 32,  # hidden_channels / num_heads = 512 / 16 = 32
        dropout: float = 0.0,
    ):
        super().__init__()

        # TODO: make sure all the blocks are initialized the same way as original code

        # 1. Define latent conditioner, which processes the latent conditioning information
        # from the autoregressive model
        self.latent_conditioner = ConditioningEncoder(
            in_channels=in_latent_channels,
            out_channels=hidden_channels,
            num_layers=num_latent_cond_layers,
            attention_head_dim=attention_head_dim,
            relative_pos_embeddings=True,
            input_transform="conv",
            input_conv_kernel_size=3,
            input_conv_stride=1,
            input_conv_padding=1,
            output_transform="groupnorm",
            output_num_groups=32,  # TODO: get accurate num_groups
        )

        # 2. Define unconditioned embedding (TODO: add more information)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1, hidden_channels, 1))

        # 3. Define conditioning timestep integrator, which combines the conditioning embedding from the
        # autoregressive model with the time embedding
        self.conditioning_timestep_integrator = AttnEncoderBlock1D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            temb_channels=hidden_channels,
            dropout=dropout,
            num_layers=num_timestep_integrator_layers,
        )

        # 4. Define the timestep embedding. Only support positional embeddings for now.
        time_embed_dim = hidden_channels
        self.time_proj = Timesteps(hidden_channels, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_channels, time_embed_dim)

        # 5. Define the inital Conv1d layers
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv_add_cond_emb_to_hidden = nn.Conv1d(2 * hidden_channels, hidden_channels, 1)

        # 6. Define the trunk of the denoising model
        self.blocks = AttnEncoderBlock1D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            temb_channels=hidden_channels,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.post_res_blocks = nn.ModuleList(
            [
                ResnetBlock1D(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    temb_channels=hidden_channels,
                    dropout=dropout,
                    time_embedding_norm="scale_shift",
                )
                for _ in range(num_post_res_blocks)
            ]
        )

        # 7. Define the output layers
        self.norm_out = nn.GroupNorm(32, hidden_channels)  # TODO: get right number of groups
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        autoregressive_latents: torch.FloatTensor,
        conditioning_audio_latents: torch.FloatTensor,
        unconditional: bool = False,
        return_dict: bool = True
    ):
        """
        TODO
        """
        # 1. Handle the conditioning embedding
        if unconditional:
            cond_embedding = self.unconditioned_embedding.repeat(sample.shape[0], 1, sample.shape[-1])
        else:
            cond_scale, cond_shift = torch.chunk(conditioning_audio_latents, 2, dim=1)
            cond_embedding = self.latent_conditioner(autoregressive_latents).embedding
            cond_embedding = cond_embedding * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)
            # Interpolate conditional embeddings...?
            cond_embedding = F.interpolate(cond_embedding, size=sample.shape[-1], mode="nearest")
        
        # 2. Handle timestep embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 3. Combine conditioning embedding with timestep embedding
        cond_embedding = self.conditioning_timestep_integrator(cond_embedding, temb=emb)[0]

        # 4. Map inital sample to hidden states
        sample = self.conv_in(sample)

        # 5. Concatenate initial hidden states with conditioning embedding and process
        sample = torch.cat([sample, cond_embedding], dim=1)
        sample = self.conv_add_cond_emb_to_hidden(sample)

        # 6. Run the hidden states through the trunk of the denoising model
        sample = self.blocks(sample, temb=emb)[0]
        sample = self.post_res_blocks(sample, emb)

        # 7. Map hidden states out to a denoised sample
        sample = F.silu(self.norm_out(sample))
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        
        return TortoiseTTSDenoisingModelOutput(sample=sample)
