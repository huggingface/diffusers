# Copyright 2024 the Latte Team and The HuggingFace Team. All rights reserved.
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
from typing import Optional

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.embeddings import PixArtAlphaTextProjection, get_1d_sincos_pos_embed_from_grid
from ..attention import BasicTransformerBlock
from ..embeddings import PatchEmbed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle


class LatteTransformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 3D Transformer model for video-like data, paper: https://arxiv.org/abs/2401.03048, offical code:
    https://github.com/Vchitect/Latte

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input.
        out_channels (`int`, *optional*):
            The number of channels in the output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*):
            The size of the patches to use in the patch embedding layer.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states. During inference, you can denoise for up to but not more steps than
            `num_embeds_ada_norm`.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of normalization to use. Options are `"layer_norm"` or `"ada_layer_norm"`.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use in normalization layers.
        caption_channels (`int`, *optional*):
            The number of channels in the caption embeddings.
        video_length (`int`, *optional*):
            The number of frames in the video-like data.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: int = 64,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        video_length: int = 16,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Define input layers
        self.height = sample_size
        self.width = sample_size

        interpolation_scale = self.config.sample_size // 64
        interpolation_scale = max(interpolation_scale, 1)
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        # 2. Define spatial transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        # 3. Define temporal transformers blocks
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=None,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. Latte other blocks.
        self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        # define temporal positional embedding
        temp_pos_embed = get_1d_sincos_pos_embed_from_grid(
            inner_dim, torch.arange(0, video_length).unsqueeze(1), output_type="pt"
        )  # 1152 hidden size
        self.register_buffer("temp_pos_embed", temp_pos_embed.float().unsqueeze(0), persistent=False)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        """
        The [`LatteTransformer3DModel`] forward method.

        Args:
            hidden_states shape `(batch size, channel, num_frame, height, width)`:
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batcheight, sequence_length)` True = keep, False = discard.
                    * Bias `(batcheight, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            enable_temporal_attentions:
                (`bool`, *optional*, defaults to `True`): Whether to enable temporal attentions.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        # Reshape hidden states
        batch_size, channels, num_frame, height, width = hidden_states.shape
        # batch_size channels num_frame height width -> (batch_size * num_frame) channels height width
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        # Input
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        num_patches = height * width

        hidden_states = self.pos_embed(hidden_states)  # alrady add positional embeddings

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        # Prepare text embeddings for spatial block
        # batch_size num_tokens hidden_size -> (batch_size * num_frame) num_tokens hidden_size
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # 3 120 1152
        encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(num_frame, dim=0).view(
            -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
        )

        # Prepare timesteps for spatial and temporal block
        timestep_spatial = timestep.repeat_interleave(num_frame, dim=0).view(-1, timestep.shape[-1])
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0).view(-1, timestep.shape[-1])

        # Spatial and temporal transformer blocks
        for i, (spatial_block, temp_block) in enumerate(
            zip(self.transformer_blocks, self.temporal_transformer_blocks)
        ):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                    use_reentrant=False,
                )
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                )

            if enable_temporal_attentions:
                # (batch_size * num_frame) num_tokens hidden_size -> (batch_size * num_tokens) num_frame hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

                if i == 0 and num_frame > 1:
                    hidden_states = hidden_states + self.temp_pos_embed

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        temp_block,
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                        use_reentrant=False,
                    )
                else:
                    hidden_states = temp_block(
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                    )

                # (batch_size * num_tokens) num_frame hidden_size -> (batch_size * num_frame) num_tokens hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        embedded_timestep = embedded_timestep.repeat_interleave(num_frame, dim=0).view(-1, embedded_timestep.shape[-1])
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )
        output = output.reshape(batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1]).permute(
            0, 2, 1, 3, 4
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
