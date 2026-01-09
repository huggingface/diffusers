# Copyright 2025 OmniGen team and The HuggingFace Team. All rights reserved.
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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention_processor import Attention
from ..embeddings import TimestepEmbedding, Timesteps, get_2d_sincos_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class OmniGenFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


class OmniGenPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
        interpolation_scale: float = 1,
        pos_embed_max_size: int = 192,
        base_size: int = 64,
    ):
        super().__init__()

        self.output_image_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.input_image_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

        self.patch_size = patch_size
        self.interpolation_scale = interpolation_scale
        self.pos_embed_max_size = pos_embed_max_size

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            self.pos_embed_max_size,
            base_size=base_size,
            interpolation_scale=self.interpolation_scale,
            output_type="pt",
        )
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=True)

    def _cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def _patch_embeddings(self, hidden_states: torch.Tensor, is_input_image: bool) -> torch.Tensor:
        if is_input_image:
            hidden_states = self.input_image_proj(hidden_states)
        else:
            hidden_states = self.output_image_proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states

    def forward(
        self, hidden_states: torch.Tensor, is_input_image: bool, padding_latent: torch.Tensor = None
    ) -> torch.Tensor:
        if isinstance(hidden_states, list):
            if padding_latent is None:
                padding_latent = [None] * len(hidden_states)
            patched_latents = []
            for sub_latent, padding in zip(hidden_states, padding_latent):
                height, width = sub_latent.shape[-2:]
                sub_latent = self._patch_embeddings(sub_latent, is_input_image)
                pos_embed = self._cropped_pos_embed(height, width)
                sub_latent = sub_latent + pos_embed
                if padding is not None:
                    sub_latent = torch.cat([sub_latent, padding.to(sub_latent.device)], dim=-2)
                patched_latents.append(sub_latent)
        else:
            height, width = hidden_states.shape[-2:]
            pos_embed = self._cropped_pos_embed(height, width)
            hidden_states = self._patch_embeddings(hidden_states, is_input_image)
            patched_latents = hidden_states + pos_embed

        return patched_latents


class OmniGenSuScaledRotaryEmbedding(nn.Module):
    def __init__(
        self, dim, max_position_embeddings=131072, original_max_position_embeddings=4096, base=10000, rope_scaling=None
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

        self.short_factor = rope_scaling["short_factor"]
        self.long_factor = rope_scaling["long_factor"]
        self.original_max_position_embeddings = original_max_position_embeddings

    def forward(self, hidden_states, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=hidden_states.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=hidden_states.device)

        inv_freq_shape = (
            torch.arange(0, self.dim, 2, dtype=torch.int64, device=hidden_states.device).float() / self.dim
        )
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = hidden_states.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)[0]

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos, sin


class OmniGenAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the OmniGen model.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        bsz, q_len, query_dim = query.size()
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real_unbind_dim=-2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).type_as(query)
        hidden_states = hidden_states.reshape(bsz, q_len, attn.out_dim)
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states


class OmniGenBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=hidden_size,
            dim_head=hidden_size // num_attention_heads,
            heads=num_attention_heads,
            kv_heads=num_key_value_heads,
            bias=False,
            out_dim=hidden_size,
            out_bias=False,
            processor=OmniGenAttnProcessor2_0(),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = OmniGenFeedForward(hidden_size, intermediate_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, image_rotary_emb: torch.Tensor
    ) -> torch.Tensor:
        # 1. Attention
        norm_hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Feed Forward
        norm_hidden_states = self.post_attention_layernorm(hidden_states)
        ff_output = self.mlp(norm_hidden_states)
        hidden_states = hidden_states + ff_output
        return hidden_states


class OmniGenTransformer2DModel(ModelMixin, ConfigMixin):
    """
    The Transformer model introduced in OmniGen (https://huggingface.co/papers/2409.11340).

    Parameters:
        in_channels (`int`, defaults to `4`):
            The number of channels in the input.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        hidden_size (`int`, defaults to `3072`):
            The dimensionality of the hidden layers in the model.
        rms_norm_eps (`float`, defaults to `1e-5`):
            Eps for RMSNorm layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        num_key_value_heads (`int`, defaults to `32`):
            The number of heads to use for keys and values in multi-head attention.
        intermediate_size (`int`, defaults to `8192`):
            Dimension of the hidden layer in FeedForward layers.
        num_layers (`int`, default to `32`):
            The number of layers of transformer blocks to use.
        pad_token_id (`int`, default to `32000`):
            The id of the padding token.
        vocab_size (`int`, default to `32064`):
            The size of the vocabulary of the embedding vocabulary.
        rope_base (`int`, default to `10000`):
            The default theta value to use when creating RoPE.
        rope_scaling (`Dict`, optional):
            The scaling factors for the RoPE. Must contain `short_factor` and `long_factor`.
        pos_embed_max_size (`int`, default to `192`):
            The maximum size of the positional embeddings.
        time_step_dim (`int`, default to `256`):
            Output dimension of timestep embeddings.
        flip_sin_to_cos (`bool`, default to `True`):
            Whether to flip the sin and cos in the positional embeddings when preparing timestep embeddings.
        downscale_freq_shift (`int`, default to `0`):
            The frequency shift to use when downscaling the timestep embeddings.
        timestep_activation_fn (`str`, default to `silu`):
            The activation function to use for the timestep embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["OmniGenBlock"]
    _skip_layerwise_casting_patterns = ["patch_embedding", "embed_tokens", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 3072,
        rms_norm_eps: float = 1e-5,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 8192,
        num_layers: int = 32,
        pad_token_id: int = 32000,
        vocab_size: int = 32064,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        rope_base: int = 10000,
        rope_scaling: Dict = None,
        pos_embed_max_size: int = 192,
        time_step_dim: int = 256,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: int = 0,
        timestep_activation_fn: str = "silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.patch_embedding = OmniGenPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.time_proj = Timesteps(time_step_dim, flip_sin_to_cos, downscale_freq_shift)
        self.time_token = TimestepEmbedding(time_step_dim, hidden_size, timestep_activation_fn)
        self.t_embedder = TimestepEmbedding(time_step_dim, hidden_size, timestep_activation_fn)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.rope = OmniGenSuScaledRotaryEmbedding(
            hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
        )

        self.layers = nn.ModuleList(
            [
                OmniGenBlock(hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_out = AdaLayerNorm(hidden_size, norm_elementwise_affine=False, norm_eps=1e-6, chunk_dim=1)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def _get_multimodal_embeddings(
        self, input_ids: torch.Tensor, input_img_latents: List[torch.Tensor], input_image_sizes: Dict
    ) -> Optional[torch.Tensor]:
        if input_ids is None:
            return None

        input_img_latents = [x.to(self.dtype) for x in input_img_latents]
        condition_tokens = self.embed_tokens(input_ids)
        input_img_inx = 0
        input_image_tokens = self.patch_embedding(input_img_latents, is_input_image=True)
        for b_inx in input_image_sizes.keys():
            for start_inx, end_inx in input_image_sizes[b_inx]:
                # replace the placeholder in text tokens with the image embedding.
                condition_tokens[b_inx, start_inx:end_inx] = input_image_tokens[input_img_inx].to(
                    condition_tokens.dtype
                )
                input_img_inx += 1
        return condition_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.FloatTensor],
        input_ids: torch.Tensor,
        input_img_latents: List[torch.Tensor],
        input_image_sizes: Dict[int, List[int]],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        # 1. Patch & Timestep & Conditional Embedding
        hidden_states = self.patch_embedding(hidden_states, is_input_image=False)
        num_tokens_for_output_image = hidden_states.size(1)

        timestep_proj = self.time_proj(timestep).type_as(hidden_states)
        time_token = self.time_token(timestep_proj).unsqueeze(1)
        temb = self.t_embedder(timestep_proj)

        condition_tokens = self._get_multimodal_embeddings(input_ids, input_img_latents, input_image_sizes)
        if condition_tokens is not None:
            hidden_states = torch.cat([condition_tokens, time_token, hidden_states], dim=1)
        else:
            hidden_states = torch.cat([time_token, hidden_states], dim=1)

        seq_length = hidden_states.size(1)
        position_ids = position_ids.view(-1, seq_length).long()

        # 2. Attention mask preprocessing
        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = hidden_states.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).type_as(hidden_states)

        # 3. Rotary position embedding
        image_rotary_emb = self.rope(hidden_states, position_ids)

        # 4. Transformer blocks
        for block in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, attention_mask, image_rotary_emb
                )
            else:
                hidden_states = block(hidden_states, attention_mask=attention_mask, image_rotary_emb=image_rotary_emb)

        # 5. Output norm & projection
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states[:, -num_tokens_for_output_image:]
        hidden_states = self.norm_out(hidden_states, temb=temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, p, p, -1)
        output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
