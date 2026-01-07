# Copyright 2025 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import GlmImageDecoderCombinedTimestepSizeEmbeddings
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import LayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class GlmImageDecoderImageProjector(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size, channel, post_patch_height, self.patch_size, post_patch_width, self.patch_size
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        hidden_states = self.proj(hidden_states)

        return hidden_states


class GlmImageDecoderAdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, glyph_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        norm_hidden_states = self.norm(hidden_states).to(dtype=dtype)
        norm_glyph_hidden_states = self.norm_context(glyph_hidden_states).to(dtype=dtype)

        emb = self.linear(temb)
        (
            shift_msa,
            c_shift_msa,
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)

        hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        glyph_hidden_states = norm_glyph_hidden_states * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            glyph_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class GlmImageDecoderAttenProcessorState(Enum):
    ImageGen = "ImageGen"
    ImageEditWriteKV = "ImageEditWriteKV"
    ImageEditReadKV = "ImageEditReadKV"
    ImageEditDontReadKV = "ImageEditNoReadKV"


class GlmImageDecoderAttnProcessor:
    """
    Processor for implementing scaled dot-product attention for the GlmImageDecoder model. It applies a rotary
    embedding on query and key vectors, but does not include spatial normalization.

    The processor supports passing an attention mask for text tokens. The attention mask should have shape (batch_size,
    text_seq_length) where 1 indicates a non-padded token and 0 indicates a padded token.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "GlmImageDecoderAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
        self.processor_state = GlmImageDecoderAttenProcessorState.ImageGen
        self.k_cache = None
        self.v_cache = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = encoder_hidden_states.dtype

        batch_size, text_seq_length, embed_dim = encoder_hidden_states.shape
        batch_size, image_seq_length, embed_dim = hidden_states.shape
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
            query = attn.norm_q(query).to(dtype=dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype=dtype)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            query[:, :, text_seq_length:, :] = apply_rotary_emb(
                query[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
            )
            key[:, :, text_seq_length:, :] = apply_rotary_emb(
                key[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
            )

        if self.processor_state == GlmImageDecoderAttenProcessorState.ImageEditWriteKV:
            self.k_cache = key if self.k_cache is None else torch.cat([self.k_cache, key], dim=2)
            self.v_cache = value if self.v_cache is None else torch.cat([self.v_cache, value], dim=2)
        elif self.processor_state == GlmImageDecoderAttenProcessorState.ImageEditReadKV:
            key = torch.cat([self.k_cache, key], dim=2) if self.k_cache is not None else key
            value = torch.cat([self.v_cache, value], dim=2) if self.v_cache is not None else value

        # 4. Attention
        if attention_mask is not None:
            text_attn_mask = attention_mask
            assert text_attn_mask.dim() == 2, "the shape of text_attn_mask should be (batch_size, text_seq_length)"
            text_attn_mask = text_attn_mask.float().to(query.device)
            mix_attn_mask = torch.ones((batch_size, text_seq_length + image_seq_length), device=query.device)
            mix_attn_mask[:, :text_seq_length] = text_attn_mask
            mix_attn_mask = mix_attn_mask.unsqueeze(2)
            attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)
            attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)

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


@maybe_allow_in_graph
class GlmImageDecoderTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()

        # 1. Attention
        self.norm1 = GlmImageDecoderAdaLayerNormZero(time_embed_dim, dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-5,
            processor=GlmImageDecoderAttnProcessor(),
        )

        # 2. Feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        glyph_hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Timestep conditioning
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_glyph_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, glyph_hidden_states, temb)

        # 2. Attention
        if attention_kwargs is None:
            attention_kwargs = {}

        attn_hidden_states, attn_glyph_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_glyph_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **attention_kwargs,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        glyph_hidden_states = glyph_hidden_states + attn_glyph_hidden_states * c_gate_msa.unsqueeze(1)

        # 3. Feedforward
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        norm_glyph_hidden_states = self.norm2_context(glyph_hidden_states) * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_glyph_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        glyph_hidden_states = glyph_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)

        return hidden_states, glyph_hidden_states


class GlmImageDecoderRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int, theta: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)].float() / dim_h)
        )
        w_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32)[: (dim_w // 2)].float() / dim_w)
        )
        h_seq = torch.arange(height)
        w_seq = torch.arange(width)
        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)

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


class GlmImageDecoderAdaLayerNormContinuous(nn.Module):
    """
    GlmImageDecoder-only final AdaLN: LN(x) -> Linear(cond) -> chunk -> affine. Matches Megatron: **no activation**
    before the Linear on conditioning embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # *** NO SiLU here ***
        emb = self.linear(conditioning_embedding.to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class GlmImageDecoderTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
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
    _no_split_modules = [
        "GlmImageDecoderTransformerBlock",
        "GlmImageDecoderImageProjector",
        "GlmImageDecoderImageProjector",
    ]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm", "proj_out"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        text_embed_dim: int = 4096,
        glyph_embed_dim: int = 1472,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        pos_embed_max_size: int = 128,
        sample_size: int = 128,
        prior_vq_quantizer_codebook_size: int = 16384,
    ):
        super().__init__()

        # GlmImageDecoder uses 2 additional SDXL-like conditions - target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        pooled_projection_dim = 2 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels

        # 1. RoPE
        self.rope = GlmImageDecoderRotaryPosEmbed(attention_head_dim, patch_size, theta=10000.0)

        # 2. Patch & Text-timestep embedding
        self.image_projector = GlmImageDecoderImageProjector(in_channels, inner_dim, patch_size)
        # 这次没有，未来可能有text_projector
        # self.text_projector = FeedForward(text_embed_dim, inner_dim, activation_fn="gelu")
        self.glyph_projector = FeedForward(glyph_embed_dim, inner_dim, inner_dim=inner_dim, activation_fn="gelu")

        self.prior_token_embedding = nn.Embedding(prior_vq_quantizer_codebook_size, inner_dim)
        self.prior_projector = FeedForward(inner_dim, inner_dim, inner_dim=inner_dim, activation_fn="linear-silu")

        self.time_condition_embed = GlmImageDecoderCombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=time_embed_dim,
        )

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                GlmImageDecoderTransformerBlock(inner_dim, num_attention_heads, attention_head_dim, time_embed_dim)
                for _ in range(num_layers)
            ]
        )

        # 4. Output projection
        self.norm_out = GlmImageDecoderAdaLayerNormContinuous(inner_dim, time_embed_dim, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        glyph_hidden_states: torch.Tensor,
        prior_token_id: torch.Tensor,
        prior_token_drop: torch.Tensor,
        timestep: torch.LongTensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        batch_size, num_channels, height, width = hidden_states.shape

        # 1. RoPE
        if image_rotary_emb is None:
            image_rotary_emb = self.rope(hidden_states)

        # 2. Patch & Timestep embeddings
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = self.image_projector(hidden_states)
        glyph_hidden_states = self.glyph_projector(glyph_hidden_states)
        prior_embedding = self.prior_token_embedding(prior_token_id)
        prior_embedding[prior_token_drop] *= 0.0
        prior_hidden_states = self.prior_projector(prior_embedding)

        hidden_states = hidden_states + prior_hidden_states

        temb = self.time_condition_embed(timestep, target_size, crop_coords, hidden_states.dtype)
        temb = F.silu(temb)

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, glyph_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    glyph_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    attention_kwargs,
                )
            else:
                hidden_states, glyph_hidden_states = block(
                    hidden_states,
                    glyph_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    attention_kwargs,
                )

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, -1, p, p)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def set_attention_processors_state(self, state: GlmImageDecoderAttenProcessorState):
        for block in self.transformer_blocks:
            block.attn1.processor.processor_state = state

    def clear_attention_processors_cache(self):
        for block in self.transformer_blocks:
            block.attn1.processor.k_cache = None
            block.attn1.processor.v_cache = None

    def repeat_attention_processors_cache(self, repeats: int):
        for block in self.transformer_blocks:
            if block.attn1.processor.k_cache is None or block.attn1.processor.v_cache is None:
                continue
            block.attn1.processor.k_cache = torch.repeat_interleave(block.attn1.processor.k_cache, repeats, dim=2)
            block.attn1.processor.v_cache = torch.repeat_interleave(block.attn1.processor.v_cache, repeats, dim=2)


if __name__ == "__main__":

    def swap_scale_shift(weight, dim):
        """
        Swap the scale and shift components in the weight tensor.

        Args:
            weight (torch.Tensor): The original weight tensor.
            dim (int): The dimension along which to split.

        Returns:
            torch.Tensor: The modified weight tensor with scale and shift swapped.
        """
        shift, scale = weight.chunk(2, dim=dim)
        new_weight = torch.cat([scale, shift], dim=dim)
        return new_weight

    def convert_megatron_transformer_checkpoint_to_diffusers(
        ckpt_path: str,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
    ):
        """
        Convert a Megatron Transformer checkpoint to Diffusers format.

        Args:
            ckpt_path (str): Path to the Megatron Transformer checkpoint.
            num_layers (int): Number of Transformer layers.
            num_heads (int): Number of attention heads.
            hidden_size (int): Hidden size of the Transformer.

        Returns:
            dict: The converted state dictionary compatible with Diffusers.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mega = ckpt["model"]
        used_keys = set()

        def get_mega(key):
            used_keys.add(key)
            return mega[key]

        new_state_dict = {}

        # Patch Embedding
        new_state_dict["image_projector.proj.weight"] = get_mega("encoder_expand_linear.weight").reshape(
            hidden_size, 64
        )
        new_state_dict["image_projector.proj.bias"] = get_mega("encoder_expand_linear.bias")

        new_state_dict["glyph_projector.net.0.proj.weight"] = get_mega("glyph_projector.linear_fc1.weight")
        new_state_dict["glyph_projector.net.0.proj.bias"] = get_mega("glyph_projector.linear_fc1.bias")
        new_state_dict["glyph_projector.net.2.weight"] = get_mega("glyph_projector.linear_fc2.weight")
        new_state_dict["glyph_projector.net.2.bias"] = get_mega("glyph_projector.linear_fc2.bias")

        new_state_dict["prior_token_embedding.weight"] = get_mega("xomni_token_id_embedding.weight")
        new_state_dict["prior_projector.net.0.proj.weight"] = get_mega("prior_condition_embedding.0.weight")
        new_state_dict["prior_projector.net.0.proj.bias"] = get_mega("prior_condition_embedding.0.bias")
        new_state_dict["prior_projector.net.2.weight"] = get_mega("prior_condition_embedding.2.weight")
        new_state_dict["prior_projector.net.2.bias"] = get_mega("prior_condition_embedding.2.bias")

        # Time Condition Embedding
        new_state_dict["time_condition_embed.timestep_embedder.linear_1.weight"] = get_mega(
            "time_embedding.time_embed.0.weight"
        )
        new_state_dict["time_condition_embed.timestep_embedder.linear_1.bias"] = get_mega(
            "time_embedding.time_embed.0.bias"
        )
        new_state_dict["time_condition_embed.timestep_embedder.linear_2.weight"] = get_mega(
            "time_embedding.time_embed.2.weight"
        )
        new_state_dict["time_condition_embed.timestep_embedder.linear_2.bias"] = get_mega(
            "time_embedding.time_embed.2.bias"
        )

        new_state_dict["time_condition_embed.condition_embedder.linear_1.weight"] = get_mega(
            "label_embedding.label_embed.0.weight"
        )
        new_state_dict["time_condition_embed.condition_embedder.linear_1.bias"] = get_mega(
            "label_embedding.label_embed.0.bias"
        )
        new_state_dict["time_condition_embed.condition_embedder.linear_2.weight"] = get_mega(
            "label_embedding.label_embed.2.weight"
        )
        new_state_dict["time_condition_embed.condition_embedder.linear_2.bias"] = get_mega(
            "label_embedding.label_embed.2.bias"
        )

        # Convert each Transformer layer
        from tqdm import tqdm

        for i in tqdm(range(num_layers), desc="Converting layers (Megatron->Diffusers)"):
            block_prefix = f"transformer_blocks.{i}."

            # AdaLayerNorm
            new_state_dict[block_prefix + "norm1.linear.weight"] = get_mega(f"decoder.layers.{i}.adaln.weight")
            new_state_dict[block_prefix + "norm1.linear.bias"] = get_mega(f"decoder.layers.{i}.adaln.bias")
            qkv_weight = get_mega(f"decoder.layers.{i}.self_attention.linear_qkv.weight")
            qkv_bias = get_mega(f"decoder.layers.{i}.self_attention.linear_qkv.bias")

            # Reshape to match SAT logic
            qkv_weight = qkv_weight.view(num_heads, 3, hidden_size // num_heads, hidden_size)
            qkv_weight = qkv_weight.permute(1, 0, 2, 3).reshape(3 * hidden_size, hidden_size)

            qkv_bias = qkv_bias.view(num_heads, 3, hidden_size // num_heads)
            qkv_bias = qkv_bias.permute(1, 0, 2).reshape(3 * hidden_size)

            # Assign to Diffusers keys
            q, k, v = torch.chunk(qkv_weight, 3, dim=0)
            qb, kb, vb = torch.chunk(qkv_bias, 3, dim=0)

            new_state_dict[block_prefix + "attn1.to_q.weight"] = q
            new_state_dict[block_prefix + "attn1.to_q.bias"] = qb
            new_state_dict[block_prefix + "attn1.to_k.weight"] = k
            new_state_dict[block_prefix + "attn1.to_k.bias"] = kb
            new_state_dict[block_prefix + "attn1.to_v.weight"] = v
            new_state_dict[block_prefix + "attn1.to_v.bias"] = vb

            # Attention Output
            new_state_dict[block_prefix + "attn1.to_out.0.weight"] = get_mega(
                f"decoder.layers.{i}.self_attention.linear_proj.weight"
            )
            new_state_dict[block_prefix + "attn1.to_out.0.bias"] = get_mega(
                f"decoder.layers.{i}.self_attention.linear_proj.bias"
            )

            # MLP
            new_state_dict[block_prefix + "ff.net.0.proj.weight"] = get_mega(
                f"decoder.layers.{i}.mlp.linear_fc1.weight"
            )
            new_state_dict[block_prefix + "ff.net.0.proj.bias"] = get_mega(f"decoder.layers.{i}.mlp.linear_fc1.bias")
            new_state_dict[block_prefix + "ff.net.2.weight"] = get_mega(f"decoder.layers.{i}.mlp.linear_fc2.weight")
            new_state_dict[block_prefix + "ff.net.2.bias"] = get_mega(f"decoder.layers.{i}.mlp.linear_fc2.bias")

        # Final Layers
        new_state_dict["norm_out.linear.weight"] = swap_scale_shift(get_mega("adaln_final.weight"), dim=0)
        new_state_dict["norm_out.linear.bias"] = swap_scale_shift(get_mega("adaln_final.bias"), dim=0)
        new_state_dict["proj_out.weight"] = get_mega("output_projector.weight")
        new_state_dict["proj_out.bias"] = get_mega("output_projector.bias")

        # Check for unused keys
        all_keys = set(mega.keys())
        unused_keys = all_keys - used_keys
        if unused_keys:
            print(f"\n[WARNING] The following {len(unused_keys)} keys in mega were NOT used:")
            for key in sorted(unused_keys):
                print(f"  - {key}")
            raise ValueError(
                f"Found {len(unused_keys)} unused keys in Megatron checkpoint. Please update the conversion script to handle these keys."
            )
        else:
            print(f"\n[INFO] All {len(all_keys)} keys in mega were successfully used.")

        return new_state_dict

    transformer = GlmImageDecoderTransformer2DModel(
        patch_size=2,
        in_channels=16,
        num_layers=30,
        attention_head_dim=128,
        num_attention_heads=32,
        out_channels=16,
        text_embed_dim=4096,
        time_embed_dim=512,
        glyph_embed_dim=1472,
        condition_dim=256,
        pos_embed_max_size=128,
    ).to(torch.bfloat16)
    converted_transformer_state_dict = convert_megatron_transformer_checkpoint_to_diffusers(
        ckpt_path="/workspace/ckpt/tjy/Glm-train-dev/examples/cogview/ckpts/merge/1+6_0.5+0.5/iter_0000000/mp_rank_00/model_optim_rng.pt",
        num_layers=30,
        num_heads=32,
        hidden_size=4096,
    )
    transformer.load_state_dict(converted_transformer_state_dict)
    transformer.cuda()

    latent = torch.load("/workspace/ckpt/tjy/glm-train-dev/examples/cogview/latent.pt").to(torch.bfloat16)
    latent = rearrange(latent, "(b h w) (c p q) -> b c (h p) (w q)", b=8, h=72, w=54, p=2, q=2)
    glyph_hidden_states = torch.load(
        "/workspace/ckpt/tjy/glm-train-dev/examples/cogview/glyph_condition_embedding.pt"
    ).to(torch.bfloat16)
    glyph_hidden_states = rearrange(glyph_hidden_states, "(b n) c -> b n c", b=8, n=2)
    prior_token_id = torch.load("/workspace/ckpt/tjy/glm-train-dev/examples/cogview/xomni_token_id.pt")
    prior_token_drop = torch.load("/workspace/ckpt/tjy/glm-train-dev/examples/cogview/xomni_drop.pt")
    prior_token_id = rearrange(prior_token_id, "(b n) -> b n", b=8)
    prior_token_drop = rearrange(prior_token_drop, "(b n)-> b n", b=8)

    with torch.no_grad():
        output = transformer(
            hidden_states=latent,
            glyph_hidden_states=glyph_hidden_states,
            prior_token_id=prior_token_id,
            prior_token_drop=prior_token_drop,
            timestep=torch.tensor([999.0] * 8).cuda(),
            original_size=torch.tensor([[144, 108]] * 8).cuda(),
            target_size=torch.tensor([[144, 108]] * 8).cuda(),
            crop_coords=torch.tensor([[0, 0]] * 8).cuda(),
        )
