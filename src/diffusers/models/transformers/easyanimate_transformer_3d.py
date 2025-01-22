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

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, reduce

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..attention_processor import AttentionProcessor, EasyAnimateAttnProcessor2_0
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, EasyAnimateRMSNorm, EasyAnimateLayerNormZero, FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class EasyAnimateDiTBlock(nn.Module):
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
        norm_type: str="fp32_layer_norm",
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
            processor=EasyAnimateAttnProcessor2_0(),
        )
        if is_mmdit_block:
            self.attn2 = Attention(
                query_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                qk_norm="layer_norm" if qk_norm else None,
                eps=1e-6,
                bias=True,
                processor=EasyAnimateAttnProcessor2_0(),
            )
        else:
            self.attn2 = None
        
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
        if is_mmdit_block:
            self.txt_ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )
        else:
            self.txt_ff = None
            
        if after_norm:
            self.norm3 = FP32LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        else:
            self.norm3 = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_frames = None,
        height = None,
        width = None
    ) -> torch.Tensor:
        # Norm
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # Attn
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attn2=self.attn2
        )
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # Norm
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # FFN
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
        hidden_states = hidden_states + gate_ff * norm_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * norm_encoder_hidden_states
        return hidden_states, encoder_hidden_states


class EasyAnimateTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [EasyAnimate](https://github.com/aigc-apps/EasyAnimate).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
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

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        sample_width: int = 90,
        sample_height: int = 60,

        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        freq_shift: int = 0,
        num_layers: int = 30,
        mmdit_layers: int = 10000,
        dropout: float = 0.0,
        time_embed_dim: int = 512,
        add_norm_text_encoder: bool = False,
        text_embed_dim: int = 4096,
        text_embed_dim_t5: int = 4096,
        norm_eps: float = 1e-5,

        norm_elementwise_affine: bool = True,
        flip_sin_to_cos: bool = True,
    
        time_position_encoding_type: str = "3d_rope", 
        after_norm = False,
        resize_inpaint_mask_directly: bool = False,
        enable_text_attention_mask: bool = True,
        add_noise_in_inpaint_model: bool = False,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.resize_inpaint_mask_directly = resize_inpaint_mask_directly
        self.patch_size = patch_size

        post_patch_height = sample_height // patch_size
        post_patch_width = sample_width // patch_size
        self.post_patch_height = post_patch_height
        self.post_patch_width = post_patch_width

        self.time_proj = Timesteps(self.inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(self.inner_dim, time_embed_dim, timestep_activation_fn)

        self.proj = nn.Conv2d(
            in_channels, self.inner_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=True
        )
        if not add_norm_text_encoder:
            self.text_proj = nn.Linear(text_embed_dim, self.inner_dim)
            if text_embed_dim_t5 is not None:
                self.text_proj_t5 = nn.Linear(text_embed_dim_t5, self.inner_dim)
        else:
            self.text_proj = nn.Sequential(
                EasyAnimateRMSNorm(text_embed_dim),
                nn.Linear(text_embed_dim, self.inner_dim)
            )
            if text_embed_dim_t5 is not None:
                self.text_proj_t5 = nn.Sequential(
                    EasyAnimateRMSNorm(text_embed_dim),
                    nn.Linear(text_embed_dim_t5, self.inner_dim)
                )

        self.transformer_blocks = nn.ModuleList(
            [
                EasyAnimateDiTBlock(
                    dim=self.inner_dim,
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
        self.norm_final = nn.LayerNorm(self.inner_dim, norm_eps, norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * self.inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_cond = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        text_embedding_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_t5: Optional[torch.Tensor] = None,
        text_embedding_mask_t5: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        return_dict=True,
    ):
        batch_size, channels, video_length, height, width = hidden_states.size()

        # 1. Time embedding
        temb = self.time_proj(timestep).to(dtype=hidden_states.dtype)
        temb = self.time_embedding(temb, timestep_cond)

        # 2. Patch embedding
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 1)
        if control_latents is not None:
            hidden_states = torch.concat([hidden_states, control_latents], 1)

        hidden_states = rearrange(hidden_states, "b c f h w ->(b f) c h w")
        hidden_states = self.proj(hidden_states)
        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length, h=height // self.patch_size, w=width // self.patch_size)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        if encoder_hidden_states_t5 is not None:
            encoder_hidden_states_t5 = self.text_proj_t5(encoder_hidden_states_t5)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1).contiguous()

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    video_length,
                    height // self.patch_size,
                    width // self.patch_size,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    num_frames=video_length,
                    height=height // self.patch_size,
                    width=width // self.patch_size
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, encoder_hidden_states.size()[1]:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=temb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, video_length, height // p, width // p, channels, p, p)
        output = output.permute(0, 4, 1, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)