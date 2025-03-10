# Copyright 2025 The EasyAnimate team and The HuggingFace Team.
# All rights reserved.
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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..embeddings import TimestepEmbedding, Timesteps, get_3d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, FP32LayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class EasyAnimateLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "fp32_layer_norm",
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale.unsqueeze(1)) + enc_shift.unsqueeze(
            1
        )
        return hidden_states, encoder_hidden_states, gate, enc_gate


class EasyAnimateRotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, rope_dim: List[int]) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.rope_dim = rope_dim

    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bs, c, num_frames, grid_height, grid_width = hidden_states.size()
        grid_height = grid_height // self.patch_size
        grid_width = grid_width // self.patch_size
        base_size_width = 90 // self.patch_size
        base_size_height = 60 // self.patch_size

        grid_crops_coords = self.get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        image_rotary_emb = get_3d_rotary_pos_embed(
            self.rope_dim,
            grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=hidden_states.size(2),
            use_real=True,
        )
        return image_rotary_emb


class EasyAnimateAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the EasyAnimateTransformer3DModel model.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "EasyAnimateAttnProcessor2_0 requires PyTorch 2.0 or above. To use it, please install PyTorch 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            query[:, :, encoder_hidden_states.shape[1] :] = apply_rotary_emb(
                query[:, :, encoder_hidden_states.shape[1] :], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, encoder_hidden_states.shape[1] :] = apply_rotary_emb(
                    key[:, :, encoder_hidden_states.shape[1] :], image_rotary_emb
                )

        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        else:
            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class EasyAnimateTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        qk_norm: bool = True,
        after_norm: bool = False,
        norm_type: str = "fp32_layer_norm",
        is_mmdit_block: bool = True,
    ):
        super().__init__()

        # Attention Part
        self.norm1 = EasyAnimateLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, norm_type=norm_type, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            added_proj_bias=True,
            added_kv_proj_dim=dim if is_mmdit_block else None,
            context_pre_only=False if is_mmdit_block else None,
            processor=EasyAnimateAttnProcessor2_0(),
        )

        # FFN Part
        self.norm2 = EasyAnimateLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, norm_type=norm_type, bias=True
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.txt_ff = None
        if is_mmdit_block:
            self.txt_ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

        self.norm3 = None
        if after_norm:
            self.norm3 = FP32LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Attention
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa.unsqueeze(1) * attn_encoder_hidden_states

        # 2. Feed-forward
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
        if self.norm3 is not None:
            norm_hidden_states = self.norm3(self.ff(norm_hidden_states))
            if self.txt_ff is not None:
                norm_encoder_hidden_states = self.norm3(self.txt_ff(norm_encoder_hidden_states))
            else:
                norm_encoder_hidden_states = self.norm3(self.ff(norm_encoder_hidden_states))
        else:
            norm_hidden_states = self.ff(norm_hidden_states)
            if self.txt_ff is not None:
                norm_encoder_hidden_states = self.txt_ff(norm_encoder_hidden_states)
            else:
                norm_encoder_hidden_states = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + gate_ff.unsqueeze(1) * norm_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff.unsqueeze(1) * norm_encoder_hidden_states
        return hidden_states, encoder_hidden_states


class EasyAnimateTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [EasyAnimate](https://github.com/aigc-apps/EasyAnimate).

    Parameters:
        num_attention_heads (`int`, defaults to `48`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        mmdit_layers (`int`, defaults to `1000`):
            The number of layers of Multi Modal Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_position_encoding_type (`str`, defaults to `3d_rope`):
            Type of time position encoding.
        after_norm (`bool`, defaults to `False`):
            Flag to apply normalization after.
        resize_inpaint_mask_directly (`bool`, defaults to `True`):
            Flag to resize inpaint mask directly.
        enable_text_attention_mask (`bool`, defaults to `True`):
            Flag to enable text attention mask.
        add_noise_in_inpaint_model (`bool`, defaults to `False`):
            Flag to add noise in inpaint model.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["EasyAnimateTransformerBlock"]
    _skip_layerwise_casting_patterns = ["^proj$", "norm", "^proj_out$"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 64,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        sample_width: int = 90,
        sample_height: int = 60,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        freq_shift: int = 0,
        num_layers: int = 48,
        mmdit_layers: int = 48,
        dropout: float = 0.0,
        time_embed_dim: int = 512,
        add_norm_text_encoder: bool = False,
        text_embed_dim: int = 3584,
        text_embed_dim_t5: int = None,
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
        flip_sin_to_cos: bool = True,
        time_position_encoding_type: str = "3d_rope",
        after_norm=False,
        resize_inpaint_mask_directly: bool = True,
        enable_text_attention_mask: bool = True,
        add_noise_in_inpaint_model: bool = True,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Timestep embedding
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)
        self.rope_embedding = EasyAnimateRotaryPosEmbed(patch_size, attention_head_dim)

        # 2. Patch embedding
        self.proj = nn.Conv2d(
            in_channels, inner_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=True
        )

        # 3. Text refined embedding
        self.text_proj = None
        self.text_proj_t5 = None
        if not add_norm_text_encoder:
            self.text_proj = nn.Linear(text_embed_dim, inner_dim)
            if text_embed_dim_t5 is not None:
                self.text_proj_t5 = nn.Linear(text_embed_dim_t5, inner_dim)
        else:
            self.text_proj = nn.Sequential(
                RMSNorm(text_embed_dim, 1e-6, elementwise_affine=True), nn.Linear(text_embed_dim, inner_dim)
            )
            if text_embed_dim_t5 is not None:
                self.text_proj_t5 = nn.Sequential(
                    RMSNorm(text_embed_dim, 1e-6, elementwise_affine=True), nn.Linear(text_embed_dim_t5, inner_dim)
                )

        # 4. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EasyAnimateTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    after_norm=after_norm,
                    is_mmdit_block=True if _ < mmdit_layers else False,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 5. Output norm & projection
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_cond: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_t5: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        batch_size, channels, video_length, height, width = hidden_states.size()
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. Time embedding
        temb = self.time_proj(timestep).to(dtype=hidden_states.dtype)
        temb = self.time_embedding(temb, timestep_cond)
        image_rotary_emb = self.rope_embedding(hidden_states)

        # 2. Patch embedding
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 1)
        if control_latents is not None:
            hidden_states = torch.concat([hidden_states, control_latents], 1)

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B, C, F, H, W] -> [BF, C, H, W]
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
            0, 2, 1, 3, 4
        )  # [BF, C, H, W] -> [B, F, C, H, W]
        hidden_states = hidden_states.flatten(2, 4).transpose(1, 2)  # [B, F, C, H, W] -> [B, FHW, C]

        # 3. Text embedding
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        if encoder_hidden_states_t5 is not None:
            encoder_hidden_states_t5 = self.text_proj_t5(encoder_hidden_states_t5)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1).contiguous()

        # 4. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, image_rotary_emb
                )

        hidden_states = self.norm_final(hidden_states)

        # 5. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb=temb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, video_length, post_patch_height, post_patch_width, channels, p, p)
        output = output.permute(0, 4, 1, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
