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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import CogView3CombinedTimestepSizeEmbeddings
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import LayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogView4PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        text_hidden_size: int = 4096,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size, channel, post_patch_height, self.patch_size, post_patch_width, self.patch_size
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class CogView4AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        norm_hidden_states = self.norm(hidden_states).to(dtype=dtype)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states).to(dtype=dtype)

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
        encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class CogView4AttnProcessor:
    """
    Processor for implementing scaled dot-product attention for the CogView4 model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.

    The processor supports passing an attention mask for text tokens. The attention mask should have shape (batch_size,
    text_seq_length) where 1 indicates a non-padded token and 0 indicates a padded token.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogView4AttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

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


class CogView4TrainingAttnProcessor:
    """
    Training Processor for implementing scaled dot-product attention for the CogView4 model. It applies a rotary
    embedding on query and key vectors, but does not include spatial normalization.

    This processor differs from CogView4AttnProcessor in several important ways:
    1. It supports attention masking with variable sequence lengths for multi-resolution training
    2. It unpacks and repacks sequences for efficient training with variable sequence lengths when batch_flag is
       provided
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogView4AttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        latent_attn_mask: Optional[torch.Tensor] = None,
        text_attn_mask: Optional[torch.Tensor] = None,
        batch_flag: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            attn (`Attention`):
                The attention module.
            hidden_states (`torch.Tensor`):
                The input hidden states.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states for cross-attention.
            latent_attn_mask (`torch.Tensor`, *optional*):
                Mask for latent tokens where 0 indicates pad token and 1 indicates non-pad token. If None, full
                attention is used for all latent tokens. Note: the shape of latent_attn_mask is (batch_size,
                num_latent_tokens).
            text_attn_mask (`torch.Tensor`, *optional*):
                Mask for text tokens where 0 indicates pad token and 1 indicates non-pad token. If None, full attention
                is used for all text tokens.
            batch_flag (`torch.Tensor`, *optional*):
                Values from 0 to n-1 indicating which samples belong to the same batch. Samples with the same
                batch_flag are packed together. Example: [0, 1, 1, 2, 2] means sample 0 forms batch0, samples 1-2 form
                batch1, and samples 3-4 form batch2. If None, no packing is used.
            image_rotary_emb (`Tuple[torch.Tensor, torch.Tensor]` or `list[Tuple[torch.Tensor, torch.Tensor]]`, *optional*):
                The rotary embedding for the image part of the input.
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: The processed hidden states for both image and text streams.
        """

        # Get dimensions and device info
        batch_size, text_seq_length, embed_dim = encoder_hidden_states.shape
        batch_size, image_seq_length, embed_dim = hidden_states.shape
        dtype = encoder_hidden_states.dtype
        device = encoder_hidden_states.device
        latent_hidden_states = hidden_states
        # Combine text and image streams for joint processing
        mixed_hidden_states = torch.cat([encoder_hidden_states, latent_hidden_states], dim=1)

        # 1. Construct attention mask and maybe packing input
        # Create default masks if not provided
        if text_attn_mask is None:
            text_attn_mask = torch.ones((batch_size, text_seq_length), dtype=torch.int32, device=device)
        if latent_attn_mask is None:
            latent_attn_mask = torch.ones((batch_size, image_seq_length), dtype=torch.int32, device=device)

        # Validate mask shapes and types
        assert text_attn_mask.dim() == 2, "the shape of text_attn_mask should be (batch_size, text_seq_length)"
        assert text_attn_mask.dtype == torch.int32, "the dtype of text_attn_mask should be torch.int32"
        assert latent_attn_mask.dim() == 2, "the shape of latent_attn_mask should be (batch_size, num_latent_tokens)"
        assert latent_attn_mask.dtype == torch.int32, "the dtype of latent_attn_mask should be torch.int32"

        # Create combined mask for text and image tokens
        mixed_attn_mask = torch.ones(
            (batch_size, text_seq_length + image_seq_length), dtype=torch.int32, device=device
        )
        mixed_attn_mask[:, :text_seq_length] = text_attn_mask
        mixed_attn_mask[:, text_seq_length:] = latent_attn_mask

        # Convert mask to attention matrix format (where 1 means attend, 0 means don't attend)
        mixed_attn_mask_input = mixed_attn_mask.unsqueeze(2).to(dtype=dtype)
        attn_mask_matrix = mixed_attn_mask_input @ mixed_attn_mask_input.transpose(1, 2)

        # Handle batch packing if enabled
        if batch_flag is not None:
            assert batch_flag.dim() == 1
            # Determine packed batch size based on batch_flag
            packing_batch_size = torch.max(batch_flag).item() + 1

            # Calculate actual sequence lengths for each sample based on masks
            text_seq_length = torch.sum(text_attn_mask, dim=1)
            latent_seq_length = torch.sum(latent_attn_mask, dim=1)
            mixed_seq_length = text_seq_length + latent_seq_length

            # Calculate packed sequence lengths for each packed batch
            mixed_seq_length_packed = [
                torch.sum(mixed_attn_mask[batch_flag == batch_idx]).item() for batch_idx in range(packing_batch_size)
            ]

            assert len(mixed_seq_length_packed) == packing_batch_size

            # Pack sequences by removing padding tokens
            mixed_attn_mask_flatten = mixed_attn_mask.flatten(0, 1)
            mixed_hidden_states_flatten = mixed_hidden_states.flatten(0, 1)
            mixed_hidden_states_unpad = mixed_hidden_states_flatten[mixed_attn_mask_flatten == 1]
            assert torch.sum(mixed_seq_length) == mixed_hidden_states_unpad.shape[0]

            # Split the unpadded sequence into packed batches
            mixed_hidden_states_packed = torch.split(mixed_hidden_states_unpad, mixed_seq_length_packed)

            # Re-pad to create packed batches with right-side padding
            mixed_hidden_states_packed_padded = torch.nn.utils.rnn.pad_sequence(
                mixed_hidden_states_packed,
                batch_first=True,
                padding_value=0.0,
                padding_side="right",
            )

            # Create attention mask for packed batches
            l = mixed_hidden_states_packed_padded.shape[1]
            attn_mask_matrix = torch.zeros(
                (packing_batch_size, l, l),
                dtype=dtype,
                device=device,
            )

            # Fill attention mask with block diagonal matrices
            # This ensures that tokens can only attend to other tokens within the same original sample
            for idx, mask in enumerate(attn_mask_matrix):
                seq_lengths = mixed_seq_length[batch_flag == idx]
                offset = 0
                for length in seq_lengths:
                    # Create a block of 1s for each sample in the packed batch
                    mask[offset : offset + length, offset : offset + length] = 1
                    offset += length

        attn_mask_matrix = attn_mask_matrix.to(dtype=torch.bool)
        attn_mask_matrix = attn_mask_matrix.unsqueeze(1)  # Add attention head dim
        attention_mask = attn_mask_matrix

        # Prepare hidden states for attention computation
        if batch_flag is None:
            # If no packing, just combine text and image tokens
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        else:
            # If packing, use the packed sequence
            hidden_states = mixed_hidden_states_packed_padded

        # 2. QKV projections - convert hidden states to query, key, value
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape for multi-head attention: [batch, seq_len, heads*dim] -> [batch, heads, seq_len, dim]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 3. QK normalization - apply layer norm to queries and keys if configured
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(dtype=dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype=dtype)

        # 4. Apply rotary positional embeddings to image tokens only
        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            if batch_flag is None:
                # Apply RoPE only to image tokens (after text tokens)
                query[:, :, text_seq_length:, :] = apply_rotary_emb(
                    query[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
                )
                key[:, :, text_seq_length:, :] = apply_rotary_emb(
                    key[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
                )
            else:
                # For packed batches, need to carefully apply RoPE to appropriate tokens
                assert query.shape[0] == packing_batch_size
                assert key.shape[0] == packing_batch_size
                assert len(image_rotary_emb) == batch_size

                rope_idx = 0
                for idx in range(packing_batch_size):
                    offset = 0
                    # Get text and image sequence lengths for samples in this packed batch
                    text_seq_length_bi = text_seq_length[batch_flag == idx]
                    latent_seq_length_bi = latent_seq_length[batch_flag == idx]

                    # Apply RoPE to each image segment in the packed sequence
                    for tlen, llen in zip(text_seq_length_bi, latent_seq_length_bi):
                        mlen = tlen + llen
                        # Apply RoPE only to image tokens (after text tokens)
                        query[idx, :, offset + tlen : offset + mlen, :] = apply_rotary_emb(
                            query[idx, :, offset + tlen : offset + mlen, :],
                            image_rotary_emb[rope_idx],
                            use_real_unbind_dim=-2,
                        )
                        key[idx, :, offset + tlen : offset + mlen, :] = apply_rotary_emb(
                            key[idx, :, offset + tlen : offset + mlen, :],
                            image_rotary_emb[rope_idx],
                            use_real_unbind_dim=-2,
                        )
                        offset += mlen
                        rope_idx += 1

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back: [batch, heads, seq_len, dim] -> [batch, seq_len, heads*dim]
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 5. Output projection - project attention output to model dimension
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Split the output back into text and image streams
        if batch_flag is None:
            # Simple split for non-packed case
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
        else:
            # For packed case: need to unpack, split text/image, then restore to original shapes
            # First, unpad the sequence based on the packed sequence lengths
            hidden_states_unpad = torch.nn.utils.rnn.unpad_sequence(
                hidden_states,
                lengths=torch.tensor(mixed_seq_length_packed),
                batch_first=True,
            )
            # Concatenate all unpadded sequences
            hidden_states_flatten = torch.cat(hidden_states_unpad, dim=0)
            # Split by original sample sequence lengths
            hidden_states_unpack = torch.split(hidden_states_flatten, mixed_seq_length.tolist())
            assert len(hidden_states_unpack) == batch_size

            # Further split each sample's sequence into text and image parts
            hidden_states_unpack = [
                torch.split(h, [tlen, llen])
                for h, tlen, llen in zip(hidden_states_unpack, text_seq_length, latent_seq_length)
            ]
            # Separate text and image sequences
            encoder_hidden_states_unpad = [h[0] for h in hidden_states_unpack]
            hidden_states_unpad = [h[1] for h in hidden_states_unpack]

            # Update the original tensors with the processed values, respecting the attention masks
            for idx in range(batch_size):
                # Place unpacked text tokens back in the encoder_hidden_states tensor
                encoder_hidden_states[idx][text_attn_mask[idx] == 1] = encoder_hidden_states_unpad[idx]
                # Place unpacked image tokens back in the latent_hidden_states tensor
                latent_hidden_states[idx][latent_attn_mask[idx] == 1] = hidden_states_unpad[idx]

            # Update the output hidden states
            hidden_states = latent_hidden_states

        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class CogView4TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()

        # 1. Attention
        self.norm1 = CogView4AdaLayerNormZero(time_embed_dim, dim)
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

        # 2. Feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
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
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)

        # 2. Attention
        if attention_kwargs is None:
            attention_kwargs = {}
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **attention_kwargs,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)

        # 3. Feedforward
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class CogView4RotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int, rope_axes_dim: Tuple[int, int], theta: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.rope_axes_dim = rope_axes_dim
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
        h_seq = torch.arange(self.rope_axes_dim[0])
        w_seq = torch.arange(self.rope_axes_dim[1])
        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)

        h_idx = torch.arange(height, device=freqs_h.device)
        w_idx = torch.arange(width, device=freqs_w.device)
        inner_h_idx = h_idx * self.rope_axes_dim[0] // height
        inner_w_idx = w_idx * self.rope_axes_dim[1] // width

        freqs_h = freqs_h[inner_h_idx]
        freqs_w = freqs_w[inner_w_idx]

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


class CogView4AdaLayerNormContinuous(nn.Module):
    """
    CogView4-only final AdaLN: LN(x) -> Linear(cond) -> chunk -> affine. Matches Megatron: **no activation** before the
    Linear on conditioning embedding.
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


class CogView4Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
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
        out_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        pos_embed_max_size: int = 128,
        sample_size: int = 128,
        rope_axes_dim: Tuple[int, int] = (256, 256),
    ):
        super().__init__()

        # CogView4 uses 3 additional SDXL-like conditions - original_size, target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        pooled_projection_dim = 3 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels

        # 1. RoPE
        self.rope = CogView4RotaryPosEmbed(attention_head_dim, patch_size, rope_axes_dim, theta=10000.0)

        # 2. Patch & Text-timestep embedding
        self.patch_embed = CogView4PatchEmbed(in_channels, inner_dim, patch_size, text_embed_dim)

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
        self.norm_out = CogView4AdaLayerNormContinuous(inner_dim, time_embed_dim, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
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
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, height, width = hidden_states.shape

        # 1. RoPE
        if image_rotary_emb is None:
            image_rotary_emb = self.rope(hidden_states)

        # 2. Patch & Timestep embeddings
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states, encoder_hidden_states = self.patch_embed(hidden_states, encoder_hidden_states)

        temb = self.time_condition_embed(timestep, original_size, target_size, crop_coords, hidden_states.dtype)
        temb = F.silu(temb)

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
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

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
