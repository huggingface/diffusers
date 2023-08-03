import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.attention_processor import AttnProcessor
from ...utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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

        scale = 1 / math.sqrt(self.d_model // self.n_heads)

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
            query_states * scale, key_states.transpose(3, 2)
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

        hidden_states = torch.permute(hidden_states, (0, 2, 1))
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


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
        expand_prefixes: bool = False,
        expand_prefixes_reduce_dim: int = 1,
        input_transform: Optional[str] = None,
        input_conv_kernel_size: int = 1,
        input_conv_stride: int = 1,
        input_conv_padding: int = 0,
        input_conv2_hidden_dim: Optional[int] = None,
        input_convert_to_mel_spectrogram: bool = False,
        input_spectrogram_sampling_rate: int = 22050,
        output_transform: Optional[str] = None,
        output_num_groups: int = 32,
        output_type: Optional[str] = None,
    ):
        super().__init__()

        if self.config.input_convert_to_mel_spectrogram:
            # Hardcoded for now (except sample rate)
            # TODO: make this configurable in __init__?
            self.mel_stft = torchaudio.transforms.MelSpectrogram(
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                power=2,
                normalized=False,
                sample_rate=input_spectrogram_sampling_rate,
                f_min=0,
                f_max=8000,
                n_mels=80,
                norm="slaney",
            )

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
    
    def forward(self, x: torch.FloatTensor, return_dict: bool = True):
        """
        Converts either a waveform or MEL spectrogram a conditioning embedding.

        Args:
            x (`torch.FloatTensor`):
                An input waveform tensor of shape `(batch_size, time)` if `self.convert_to_mel_spectrogram = True` or
                an input MEL spectrogram of shape `(batch_size, n_mels, time)` otherwise.
            return_dict: (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `ConditioningEncoderOutput` instead of a plain tuple.
        
        Returns:
            `ConditioningEncoderOutput` or `tuple`:
            `ConditioningEncoderOutput` if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the
            first element is a `torch.FloatTensor` of shape TODO.
        """
        if self.config.input_convert_to_mel_spectrogram:
            x = self.mel_stft(x)  # ()
            # TODO: Original code uses dynamic range compression with mel norms, currently not implemented
        
        # x should have shape (batch_size, n_mels, time) ???
        if self.config.expand_prefixes:
            # Get the prefixes of each sample in x with respect to the mel channels and stack them together.
            prefixes = []
            for i in range(x.shape[1]):
                prefix = x[:, i]
                prefix = self.input_transform(prefix)
                prefix = self.attention(prefix)
                prefix = self.output_transform(prefix)
                if self.config.output_type == "mean":
                    prefix = prefix.mean(dim=2)
                elif self.config.output_type == "first":
                    prefix = prefix[:, :, 0]
                prefixes.append(prefix)
            prefixes = torch.stack(prefixes, dim=self.config.expand_prefixes_reduce_dim)
            x = prefixes.mean(dim=self.config.expand_prefixes_reduce_dim)
        else:
            x = self.input_transform(x)
            for attn_layer in self.attention:
                x = attn_layer(x)
            x = self.output_transform(x)
            if self.config.output_type == "mean":
                x = x.mean(dim=2)
            elif self.config.output_type == "first":
                x = x[:, :, 0]

        if not return_dict:
            return (x,)
        
        return ConditioningEncoderOutput(embedding=x)


# From tortoise.models.random_latent_generator.fused_leaky_relu
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/random_latent_generator.py#L8
def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim),
                negative_slope=negative_slope,
            )
            * scale
        )
    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


# From tortoise.models.random_latent_generator.EqualLinear
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/random_latent_generator.py#L22
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        out = fused_leaky_relu(out, self.bias * self.lr_mul)
        return out


@dataclass
class RandomLatentConverterOutput(BaseOutput):
    """
    The output of [`RandomLatentConverter`].

    Args:
        TODO: fix
        latents (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    latents: torch.FloatTensor


# Based on tortoise.models.random_latent_generator.RandomLatentConverter
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/random_latent_generator.py#L39
class RandomLatentConverter(ModelMixin, ConfigMixin):
    """
    Converts standard Gaussian noise to random latents suitable for use as conditioning embeddings in place of output
    from a [`ConditioningEncoder`] class, when no conditioning audio is available.

    Parameters:
        channels (`int`):
            The number of input channels of the incoming Gaussian noise tensors.
        num_equallinear_layers (`int`, *optional*, defaults to 5):
            The number of `EqualLinear` layers to use (before the final linear layer).
        lr_mul (`float`, *optional*, defaults to 0.1):
            TODO
    """
    @register_to_config
    def __init__(self, channels: int, num_equallinear_layers: int = 5, lr_mul: float = 0.1):
        super().__init__()

        self.equallinear = nn.ModuleList(
            [
                EqualLinear(channels, channels, lr_mul=lr_mul)
                for _ in range(num_equallinear_layers)
            ]
        )
        self.linear = nn.Linear(channels, channels)
    
    def forward(self, noise: torch.FloatTensor, return_dict: bool = True):
        """
        Converts standard Gaussian noise into latents.

        Args:
            noise (`torch.FloatTensor`):
                A tensor of standard Gaussian noise (e.g. from `torch.randn`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`RandomLatentConverterOutput`] instead of a plain tuple.

        Returns:
            [`RandomLatentConverterOutput`] or `tuple`:
            [`RandomLatentConverterOutput`] if `return_dict` is `True`, otherwise a `tuple`.
            When returning a tuple the first element is the rnadom latents.
        """
        assert noise.shape[-1] == self.config.channels, "The last dim of `noise` must match `self.config.channels`."
        
        for equallinear_layer in self.equallinear:
            noise = equallinear_layer(noise)
        latents = self.linear(noise)
        
        if not return_dict:
            return (latents,)
        
        return RandomLatentConverterOutput(latents=latents)
