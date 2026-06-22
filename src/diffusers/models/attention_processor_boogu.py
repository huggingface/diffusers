import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention_dispatch import dispatch_attention_fn
from .attention_processor import Attention


def apply_rotary_emb(x, freqs_cis, use_real=True, **kwargs):
    # use_real=True path delegates to the shared diffusers implementation.
    # use_real=False (Lumina-style) uses explicit dim to handle 0-element tensors.
    if use_real:
        from .embeddings import apply_rotary_emb as _apply

        return _apply(x, freqs_cis, use_real=True, **kwargs)
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
    freqs_cis = freqs_cis.unsqueeze(2)
    return torch.view_as_real(x_rotated * freqs_cis).flatten(3).type_as(x)


def _prepare_attn_mask(attention_mask: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    """Reshape a bool padding mask ``[B, L]`` to the ``[B, 1, 1, L]`` form `dispatch_attention_fn` expects.

    The mask is always materialized (not dropped to ``None`` when no token is masked):
    the native backend rounds bf16 differently on its masked vs no-mask paths, and the
    Boogu checkpoints were trained with the mask applied.
    """
    if attention_mask is None:
        return None
    return attention_mask.bool().view(batch_size, 1, 1, -1)


class BooguImageDoubleStreamSelfAttnProcessor(nn.Module):
    """
    Double-stream self-attention processor.

    Instruction and image features are projected separately, concatenated
    (instruction first, then image) into a joint sequence, attended jointly via
    [`dispatch_attention_fn`], then split back so each stream gets its own output
    projection. The QKV / output projections live on this processor module, so the
    checkpoint keys are ``...processor.img_to_q`` / ``...processor.instruct_to_q`` /
    ``...processor.img_out`` / ``...processor.instruct_out``.

    Args:
        head_dim: Dimension of each attention head
        num_attention_heads: Number of attention heads for queries
        num_kv_heads: Number of key-value heads
        qkv_bias: Whether to use bias in QKV linear layers
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize the double-stream attention processor."""
        super().__init__()

        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads

        query_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_kv_heads

        # Separate Q/K/V projections for instruction and image streams.
        # Query uses num_attention_heads, Key/Value use num_kv_heads.
        self.img_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.img_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        # Separate output projections for instruction and image streams.
        self.instruct_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Xavier-uniform init for the projection weights, zeros for any biases."""
        for proj in (
            self.img_to_q,
            self.img_to_k,
            self.img_to_v,
            self.instruct_to_q,
            self.instruct_to_k,
            self.instruct_to_v,
            self.instruct_out,
            self.img_out,
        ):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _concat_instruction_image_features(
        self,
        img_hidden_states_list: List[torch.Tensor],
        instruct_hidden_states_list: List[torch.Tensor],
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
    ) -> List[torch.Tensor]:
        """
        Concatenate instruction (text & image) and reference image features (instruction first, then image).

        Args:
            img_hidden_states_list: List of image tensors [img_query, img_key, img_value]
            instruct_hidden_states_list: List of instruction tensors [instruct_query, instruct_key, instruct_value]
            encoder_seq_lengths: Instruction sequence lengths for each sample [B]
            seq_lengths: Total sequence lengths for each sample [B]

        Returns:
            List of concatenated tensors [query, key, value]
        """
        assert len(img_hidden_states_list) == len(instruct_hidden_states_list), (
            f"Length mismatch: img_list={len(img_hidden_states_list)}, instruct_list={len(instruct_hidden_states_list)}"
        )

        batch_size = img_hidden_states_list[0].shape[0]
        max_seq_len = max(seq_lengths)

        concatenated_list = []

        for img_tensor, instruct_tensor in zip(img_hidden_states_list, instruct_hidden_states_list):
            # Ensure tensors are on the same device
            device = img_tensor.device
            if instruct_tensor.device != device:
                instruct_tensor = instruct_tensor.to(device)

            # Create output tensor with proper shape [B, max_seq_len, feature_dim]
            feature_dim = img_tensor.shape[-1]
            concatenated = img_tensor.new_zeros(batch_size, max_seq_len, feature_dim)

            # Concatenate instruction first, then image for each sample
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                # Place instruction tokens first
                concatenated[i, :encoder_seq_len] = instruct_tensor[i, :encoder_seq_len]
                # Place image tokens after instruction
                concatenated[i, encoder_seq_len:seq_len] = img_tensor[i, : seq_len - encoder_seq_len]

            concatenated_list.append(concatenated)

        return concatenated_list

    def _split_instruction_image_features(
        self,
        hidden_states_list: List[torch.Tensor],
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Split concatenated features back to instruction and image features.
        Inverse operation of _concat_instruction_image_features.

        Args:
            hidden_states_list: List of concatenated tensors (usually just one element)
            encoder_seq_lengths: Instruction sequence lengths for each sample [B]
            seq_lengths: Total sequence lengths for each sample [B]

        Returns:
            List of tuples, each containing (instruct_hidden_states, img_hidden_states)
        """
        result_list = []

        for hidden_states in hidden_states_list:
            batch_size = hidden_states.shape[0]
            feature_dim = hidden_states.shape[-1]

            # Get maximum lengths for instruction and image
            max_instruct_len = max(encoder_seq_lengths)
            max_img_len = max(
                seq_len - encoder_seq_len for seq_len, encoder_seq_len in zip(seq_lengths, encoder_seq_lengths)
            )

            # Create output tensors [B, max_len, feature_dim]
            instruct_hidden_states = hidden_states.new_zeros(batch_size, max_instruct_len, feature_dim)
            img_hidden_states = hidden_states.new_zeros(batch_size, max_img_len, feature_dim)

            # Split each sample back to instruction and image
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                img_len = seq_len - encoder_seq_len

                # Extract instruction portion
                instruct_hidden_states[i, :encoder_seq_len] = hidden_states[i, :encoder_seq_len]
                # Extract image portion
                img_hidden_states[i, :img_len] = hidden_states[i, encoder_seq_len:seq_len]

            result_list.append((instruct_hidden_states, img_hidden_states))

        return result_list

    def __call__(
        self,
        attn: Attention,
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        joint_attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        encoder_seq_lengths: List[int] = None,  # [B] - Instruction sequence lengths for each sample
        seq_lengths: List[int] = None,  # [B] - Total sequence lengths for each sample
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process double-stream self-attention.

        Args:
            attn: Attention module
            img_hidden_states: Image hidden states tensor [B, L_img, D]
            instruct_hidden_states: Instruction hidden states tensor [B, L_instruct, D]
            joint_attention_mask: Combined padding mask [B, L_total]
            rotary_emb: Rotary embeddings for the joint sequence
            encoder_seq_lengths: Instruction sequence lengths for each sample [B]
            seq_lengths: Total sequence lengths for each sample [B]
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size = img_hidden_states.shape[0]

        # Generate Q, K, V for image and instruction streams (NO head reshaping yet)
        img_query = self.img_to_q(img_hidden_states)  # [B, L_img, query_dim]
        img_key = self.img_to_k(img_hidden_states)  # [B, L_img, kv_dim]
        img_value = self.img_to_v(img_hidden_states)  # [B, L_img, kv_dim]

        instruct_query = self.instruct_to_q(instruct_hidden_states)  # [B, L_instruct, query_dim]
        instruct_key = self.instruct_to_k(instruct_hidden_states)  # [B, L_instruct, kv_dim]
        instruct_value = self.instruct_to_v(instruct_hidden_states)  # [B, L_instruct, kv_dim]

        # Concatenate QKV across streams (instruction first, then image)
        img_list = [img_query, img_key, img_value]  # [B, L_img, feature_dim] each
        instruct_list = [instruct_query, instruct_key, instruct_value]  # [B, L_instruct, feature_dim] each
        query, key, value = self._concat_instruction_image_features(
            img_list, instruct_list, encoder_seq_lengths, seq_lengths
        )  # [B, max_seq_len, feature_dim] each

        sequence_length = max(seq_lengths)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        dtype = query.dtype

        # Reshape to [B, L, H, head_dim] (the layout dispatch_attention_fn expects)
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb, use_real=False)
            key = apply_rotary_emb(key, rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=_prepare_attn_mask(joint_attention_mask, batch_size),
            scale=softmax_scale,
            enable_gqa=kv_heads < attn.heads,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        # Split back to instruction / image, apply separate output projections, then merge.
        split_results = self._split_instruction_image_features([hidden_states], encoder_seq_lengths, seq_lengths)
        instruct_hidden_states, img_hidden_states = split_results[0]

        instruct_projected = self.instruct_out(instruct_hidden_states)  # [B, max_instruct_len, feature_dim]
        img_projected = self.img_out(img_hidden_states)  # [B, max_img_len, feature_dim]

        merged_list = self._concat_instruction_image_features(
            [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths
        )
        hidden_states = merged_list[0]  # [B, max_seq_len, feature_dim]

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class BooguImageAttnProcessor:
    """
    Single-stream self-attention processor.

    Projects Q/K/V from the (shared) `Attention` module, applies QK-norm and RoPE,
    and attends via [`dispatch_attention_fn`]. Used for the refiner / single-stream
    blocks and the image self-attention of the double-stream block.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process single-stream self-attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor (same as hidden_states for self-attention)
            attention_mask: Optional bool padding mask [B, L]
            image_rotary_emb: Optional rotary embeddings
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        dtype = query.dtype

        # Reshape to [B, L, H, head_dim] (the layout dispatch_attention_fn expects)
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=_prepare_attn_mask(attention_mask, batch_size),
            scale=softmax_scale,
            enable_gqa=kv_heads < attn.heads,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
