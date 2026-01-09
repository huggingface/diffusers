# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, Optional

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ..attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..resnet import AlphaBlender


@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    The output of [`TransformerTemporalModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size x num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input.
    """

    sample: torch.Tensor


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward. See `diffusers.models.activations.get_activation` for supported
            activation functions.
        norm_elementwise_affine (`bool`, *optional*):
            Configure if the `TransformerBlock` should use learnable elementwise affine parameters for normalization.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        positional_embeddings: (`str`, *optional*):
            The type of positional embeddings to apply to the sequence input before passing use.
        num_positional_embeddings: (`int`, *optional*):
            The maximum length of the sequence over which to apply positional embeddings.
    """

    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=num_positional_embeddings,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: torch.LongTensor = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> TransformerTemporalModelOutput:
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.Tensor` of shape `(batch size, channel, height, width)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            num_frames (`int`, *optional*, defaults to 1):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`]
                instead of a plain tuple.

        Returns:
            [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an
                [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] is returned, otherwise a
                `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, num_frames, channel)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)


class TransformerSpatioTemporalModel(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`]
                instead of a plain tuple.

        Returns:
            [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an
                [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] is returned, otherwise a
                `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        )
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, None, encoder_hidden_states, None
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
