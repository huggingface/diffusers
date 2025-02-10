# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.attention import FeedForward
from ...models.attention_processor import Attention
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous
from ...utils import logging
from ..embeddings import CogView3CombinedTimestepSizeEmbeddings
from ..modeling_outputs import Transformer2DModelOutput
from ..normalization import CogView3PlusAdaLayerNormZeroTextImage


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogView4PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        text_hidden_size: int = 4096,
        pos_embed_max_size: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.text_hidden_size = text_hidden_size
        self.pos_embed_max_size = pos_embed_max_size
        # Linear projection for image patches
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

        # Linear projection for text embeddings
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Height and width must be divisible by patch size")

        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        # b, c, h, w -> b, c, patch_height, patch_size, patch_width, patch_size
        #            -> b, patch_height, patch_width, c, patch_size, patch_size
        #            -> b, patch_height * patch_width, c * patch_size * patch_size
        hidden_states = (
            hidden_states.reshape(batch_size, channel, patch_height, self.patch_size, patch_width, self.patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, patch_height * patch_width, channel * self.patch_size * self.patch_size)
        )

        # project
        hidden_states = self.proj(hidden_states)  # embed_dim: 64 -> 4096
        prompt_embeds = self.text_proj(prompt_embeds)  # embed_dim: 4096 -> 4096
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = self.text_proj(negative_prompt_embeds)  # embed_dim: 4096 -> 4096
        return hidden_states, prompt_embeds, negative_prompt_embeds


class CogView4AttnProcessor:
    """
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogView4AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
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

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:

            def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
                cos, sin = freqs
                x_real, x_imag = x.chunk(2, dim=-1)
                x_rotated = torch.cat([-x_imag, x_real], dim=-1)
                x_out = cos * x.float() + sin * x_rotated.float()
                return x_out

            query[:, :, text_seq_length:, :] = apply_rotary_emb(query[:, :, text_seq_length:, :], image_rotary_emb)
            key[:, :, text_seq_length:, :] = apply_rotary_emb(key[:, :, text_seq_length:, :], image_rotary_emb)

        # 4. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogView4TransformerBlock(nn.Module):
    def __init__(
        self, dim: int = 2560, num_attention_heads: int = 64, attention_head_dim: int = 40, time_embed_dim: int = 512
    ) -> None:
        super().__init__()

        self.norm1 = CogView3PlusAdaLayerNormZeroTextImage(embedding_dim=time_embed_dim, dim=dim)
        self.adaln = self.norm1.linear
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-5,
            processor=CogView4AttnProcessor(),
        )

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def multi_modulate(self, hidden_states, encoder_hidden_states, factors) -> torch.Tensor:
        _, _, h = factors[0].shape
        shift_factor, scale_factor = factors[0].view(-1, h), factors[1].view(-1, h)

        shift_factor_hidden_states, shift_factor_encoder_hidden_states = shift_factor.chunk(2, dim=0)
        scale_factor_hidden_states, scale_factor_encoder_hidden_states = scale_factor.chunk(2, dim=0)
        shift_factor_hidden_states = shift_factor_hidden_states.unsqueeze(1)
        scale_factor_hidden_states = scale_factor_hidden_states.unsqueeze(1)
        hidden_states = torch.addcmul(shift_factor_hidden_states, hidden_states, (1 + scale_factor_hidden_states))

        shift_factor_encoder_hidden_states = shift_factor_encoder_hidden_states.unsqueeze(1)
        scale_factor_encoder_hidden_states = scale_factor_encoder_hidden_states.unsqueeze(1)
        encoder_hidden_states = torch.addcmul(
            shift_factor_encoder_hidden_states, encoder_hidden_states, (1 + scale_factor_encoder_hidden_states)
        )

        return hidden_states, encoder_hidden_states

    def multi_gate(self, hidden_states, encoder_hidden_states, factor):
        _, _, hidden_dim = hidden_states.shape
        gate_factor = factor.view(-1, hidden_dim)
        gate_factor_hidden_states, gate_factor_encoder_hidden_states = gate_factor.chunk(2, dim=0)
        gate_factor_hidden_states = gate_factor_hidden_states.unsqueeze(1)
        gate_factor_encoder_hidden_states = gate_factor_encoder_hidden_states.unsqueeze(1)
        hidden_states = gate_factor_hidden_states * hidden_states
        encoder_hidden_states = gate_factor_encoder_hidden_states * encoder_hidden_states

        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, encoder_hidden_states_len, hidden_dim = encoder_hidden_states.shape
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        layernorm_factor = (
            self.adaln(temb)
            .view(
                temb.shape[0],
                6,
                2,
                hidden_states.shape[-1],
            )
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        hidden_states = self.layernorm(hidden_states)
        hidden_states, encoder_hidden_states = self.multi_modulate(
            hidden_states=hidden_states[:, encoder_hidden_states_len:],
            encoder_hidden_states=hidden_states[:, :encoder_hidden_states_len],
            factors=(layernorm_factor[0], layernorm_factor[1]),
        )
        hidden_states, encoder_hidden_states = self.attn1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states, encoder_hidden_states = self.multi_gate(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            factor=layernorm_factor[2],
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states, encoder_hidden_states = self.multi_modulate(
            hidden_states=hidden_states[:, encoder_hidden_states_len:],
            encoder_hidden_states=hidden_states[:, :encoder_hidden_states_len],
            factors=(layernorm_factor[3], layernorm_factor[4]),
        )
        hidden_states = self.ff(hidden_states)
        encoder_hidden_states = self.ff(encoder_hidden_states)
        hidden_states, encoder_hidden_states = self.multi_gate(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            factor=layernorm_factor[5],
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states += residual
        hidden_states, encoder_hidden_states = (
            hidden_states[:, encoder_hidden_states_len:],
            hidden_states[:, :encoder_hidden_states_len],
        )
        return hidden_states, encoder_hidden_states


class CogView4RotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int, rope_axes_dim: Tuple[int, int], theta: float = 10000.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.rope_axes_dim = rope_axes_dim

        dim_h, dim_w = dim // 2, dim // 2
        h_inv_freq = 1.0 / (theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)].float() / dim_h))
        w_inv_freq = 1.0 / (theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32)[: (dim_w // 2)].float() / dim_w))
        h_seq = torch.arange(self.rope_axes_dim[0])
        w_seq = torch.arange(self.rope_axes_dim[1])
        self.freqs_h = torch.outer(h_seq, h_inv_freq)
        self.freqs_w = torch.outer(w_seq, w_inv_freq)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        h_idx = torch.arange(height)
        w_idx = torch.arange(width)
        inner_h_idx = h_idx * self.rope_axes_dim[0] // height
        inner_w_idx = w_idx * self.rope_axes_dim[1] // width

        self.freqs_h = self.freqs_h.to(hidden_states.device)
        self.freqs_w = self.freqs_w.to(hidden_states.device)
        freqs_h = self.freqs_h[inner_h_idx]
        freqs_w = self.freqs_w[inner_w_idx]

        # Create position matrices for height and width
        # [height, 1, dim//4] and [1, width, dim//4]
        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)
        # Broadcast freqs_h and freqs_w to [height, width, dim//4]
        freqs_h = freqs_h.expand(height, width, -1)
        freqs_w = freqs_w.expand(height, width, -1)

        # Concatenate along last dimension to get [height, width, dim//2]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)  # [height, width, dim]
        freqs = freqs.reshape(height * width, -1)
        return (freqs.cos(), freqs.sin())


class CogView4Transformer2DModel(ModelMixin, ConfigMixin):
    r"""
    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, defaults to `40`):
            The number of channels in each head.
        num_attention_heads (`int`, defaults to `64`):
            The number of heads to use for multi-head attention.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        condition_dim (`int`, defaults to `256`):
            The embedding dimension of the input SDXL-style resolution conditions (original_size, target_size,
            crop_coords).
        pos_embed_max_size (`int`, defaults to `128`):
            The maximum resolution of the positional embeddings, from which slices of shape `H x W` are taken and added
            to input patched latents, where `H` and `W` are the latent height and width respectively. A value of 128
            means that the maximum supported height and width for image generation is `128 * vae_scale_factor *
            patch_size => 128 * 8 * 2 => 2048`.
        sample_size (`int`, defaults to `128`):
            The base resolution of input latents. If height/width is not provided during generation, this value is used
            to determine the resolution as `sample_size * vae_scale_factor => 128 * 8 => 1024`
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogView4TransformerBlock", "CogView4PatchEmbed", "CogView4PatchEmbed"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm", "proj_out"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        out_channels: int = 16,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        pos_embed_max_size: int = 128,
        sample_size: int = 128,
        rope_axes_dim: Tuple[int, int] = (256, 256),
    ):
        super().__init__()

        # CogView3 uses 3 additional SDXL-like conditions - original_size, target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        pooled_projection_dim = 3 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels

        # 1. RoPE
        self.rope = CogView4RotaryPosEmbed(attention_head_dim, patch_size, rope_axes_dim, theta=10000.0)

        # 2. Patch & Text-timestep embedding
        self.patch_embed = CogView4PatchEmbed(
            in_channels=in_channels,
            hidden_size=inner_dim,
            patch_size=patch_size,
            text_hidden_size=text_embed_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.time_condition_embed = CogView3CombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=inner_dim,
        )

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogView4TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, time_embed_dim)
                for _ in range(num_layers)
            ]
        )

        # 4. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, time_embed_dim, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        timestep: torch.LongTensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`CogView3PlusTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor`):
                Input `hidden_states` of shape `(batch size, channel, height, width)`.
            encoder_hidden_states (`torch.Tensor`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) of shape
                `(batch_size, sequence_len, text_embed_dim)`
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            original_size (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for original image size as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for target image size as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crop_coords (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for crop coordinates as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            `torch.Tensor` or [`~models.transformer_2d.Transformer2DModelOutput`]:
                The denoised latents using provided inputs as conditioning.
        """
        batch_size, num_channels, height, width = hidden_states.shape
        do_cfg = negative_prompt_embeds is not None

        if do_cfg:
            assert (
                batch_size == prompt_embeds.shape[0] + negative_prompt_embeds.shape[0]
            ), "batch size mismatch in CFG mode"
        else:
            assert batch_size == prompt_embeds.shape[0], "batch size mismatch in non-CFG mode"

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_condition_embed(timestep, original_size, target_size, crop_coords, hidden_states.dtype)
        temb = F.silu(temb)
        temb_cond, temb_uncond = temb.chunk(2)
        hidden_states, prompt_embeds, negative_prompt_embeds = self.patch_embed(
            hidden_states, prompt_embeds, negative_prompt_embeds
        )
        hidden_states_cond, hidden_states_uncond = hidden_states.chunk(2)

        encoder_hidden_states_cond = prompt_embeds
        encoder_hidden_states_uncond = negative_prompt_embeds

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states_cond, encoder_hidden_states_cond = self._gradient_checkpointing_func(
                    block, hidden_states_cond, encoder_hidden_states_cond, temb_cond, image_rotary_emb
                )
                hidden_states_uncond, encoder_hidden_states_uncond = self._gradient_checkpointing_func(
                    block, hidden_states_uncond, encoder_hidden_states_uncond, temb_uncond, image_rotary_emb
                )
            else:
                hidden_states_cond, encoder_hidden_states_cond = block(
                    hidden_states_cond, encoder_hidden_states_cond, temb_cond, image_rotary_emb
                )
                hidden_states_uncond, encoder_hidden_states_uncond = block(
                    hidden_states_uncond, encoder_hidden_states_uncond, temb_uncond, image_rotary_emb
                )

        hidden_states_cond, encoder_hidden_states_cond = (
            self.norm_out(hidden_states_cond, temb_cond),
            self.norm_out(encoder_hidden_states_cond, temb_cond),
        )
        hidden_states_uncond, encoder_hidden_states_uncond = (
            self.norm_out(hidden_states_uncond, temb_uncond),
            self.norm_out(encoder_hidden_states_uncond, temb_uncond),
        )

        hidden_states_cond = self.proj_out(hidden_states_cond)
        hidden_states_uncond = self.proj_out(hidden_states_uncond)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states_cond = hidden_states_cond.reshape(
            shape=(hidden_states_cond.shape[0], height, width, -1, patch_size, patch_size)
        )
        hidden_states_cond = torch.einsum("nhwcpq->nchpwq", hidden_states_cond)
        output_cond = hidden_states_cond.reshape(
            shape=(hidden_states_cond.shape[0], -1, height * patch_size, width * patch_size)
        )

        hidden_states_uncond = hidden_states_uncond.reshape(
            hidden_states_uncond.shape[0], height, width, -1, patch_size, patch_size
        )
        hidden_states_uncond = torch.einsum("nhwcpq->nchpwq", hidden_states_uncond)
        output_uncond = hidden_states_uncond.reshape(
            hidden_states_uncond.shape[0], -1, height * patch_size, width * patch_size
        )

        if not return_dict:
            return (output_cond, output_uncond)
        return Transformer2DModelOutput(sample=output_cond), Transformer2DModelOutput(sample=output_uncond)
