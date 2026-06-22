import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.import_utils import is_flash_attn_available


if is_flash_attn_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
else:
    warnings.warn("Cannot import flash_attn, install flash_attn to use Flash2Varlen attention for better performance")


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


class BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen(nn.Module):
    """
    Double-stream self-attention processor with flash attention and variable length sequences.

    This processor implements double-stream attention where:
    - Instruction and image features are processed separately to generate QKV
    - QKV are concatenated and processed together for cross-modal attention
    - Uses flash attention for efficient computation
    - Supports both standard and causal attention masks

    Args:
        head_dim: Dimension of each attention head
        num_attention_heads: Number of attention heads for queries
        num_kv_heads: Number of key-value heads
        qkv_bias: Whether to use bias in QKV linear layers
    """

    def __init__(
        self,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize the double-stream attention processor."""
        super().__init__()
        if not is_flash_attn_available():
            raise ImportError(
                "BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen requires flash_attn. Please install flash_attn."
            )

        # Calculate dimensions
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads

        query_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_kv_heads

        # Initialize separate Q, K, V linear layers for instruction and image
        # Query uses num_attention_heads, Key/Value use num_kv_heads
        self.img_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.img_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        # Additional output projection layers for instruction and image streams
        self.instruct_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the double-stream attention processor.

        Uses Xavier uniform initialization for linear layers and zero initialization for biases.
        """
        # Initialize image stream QKV projection layers
        nn.init.xavier_uniform_(self.img_to_q.weight)
        nn.init.xavier_uniform_(self.img_to_k.weight)
        nn.init.xavier_uniform_(self.img_to_v.weight)

        # Initialize instruction stream QKV projection layers
        nn.init.xavier_uniform_(self.instruct_to_q.weight)
        nn.init.xavier_uniform_(self.instruct_to_k.weight)
        nn.init.xavier_uniform_(self.instruct_to_v.weight)

        # Initialize separate output projection layers
        nn.init.xavier_uniform_(self.instruct_out.weight)
        nn.init.xavier_uniform_(self.img_out.weight)

        # Initialize biases if they exist
        if self.img_to_q.bias is not None:
            nn.init.zeros_(self.img_to_q.bias)
            nn.init.zeros_(self.img_to_k.bias)
            nn.init.zeros_(self.img_to_v.bias)
            nn.init.zeros_(self.instruct_to_q.bias)
            nn.init.zeros_(self.instruct_to_k.bias)
            nn.init.zeros_(self.instruct_to_v.bias)
            nn.init.zeros_(self.instruct_out.bias)
            nn.init.zeros_(self.img_out.bias)

    def _upad_input(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        num_heads: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[int, int],
    ]:
        """
        Unpad the input tensors for flash attention.
        Same implementation as BooguImageAttnProcessorFlash2Varlen.
        """

        def _get_unpad_data(
            attention_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, int]:
            """Helper function to get unpadding data from attention mask."""
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # Unpad key and value layers
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )

        # Handle different query length cases
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

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
        Process double-stream self-attention computation with flash attention.

        Args:
            attn: Attention module
            img_hidden_states: Image hidden states tensor [B, L_img, D]
            instruct_hidden_states: Instruction hidden states tensor [B, L_instruct, D]
            joint_attention_mask: Combined attention mask [B, L_total]
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

        # Use helper function to concatenate QKV (instruction first, then image)
        img_list = [img_query, img_key, img_value]  # [B, L_img, feature_dim] each
        instruct_list = [
            instruct_query,
            instruct_key,
            instruct_value,
        ]  # [B, L_instruct, feature_dim] each
        concatenated_list = self._concat_instruction_image_features(
            img_list, instruct_list, encoder_seq_lengths, seq_lengths
        )
        query, key, value = concatenated_list  # [B, max_seq_len, feature_dim] each

        # From here, follow exactly the same logic as BooguImageAttnProcessorFlash2Varlen
        sequence_length = max(seq_lengths)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb, use_real=False)
            key = apply_rotary_emb(key, rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # Detect if we have a causal mask
        is_causal = False
        if joint_attention_mask is not None and joint_attention_mask.dim() == 3:
            # Check if it's a lower triangular causal mask
            # For efficiency, we only check the first sample
            mask_sample = joint_attention_mask[0]  # [seq_len, seq_len]
            is_causal = torch.allclose(mask_sample, torch.tril(torch.ones_like(mask_sample)))

        # Unpad input for flash attention
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = self._upad_input(query, key, value, joint_attention_mask, sequence_length, attn.heads)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        # Handle different number of heads
        if kv_heads < attn.heads:
            key_states = key_states.repeat_interleave(attn.heads // kv_heads, dim=1)
            value_states = value_states.repeat_interleave(attn.heads // kv_heads, dim=1)

        # Apply flash attention with causal parameter
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=0.0,
            causal=is_causal,  # Use detected causal setting
            softmax_scale=softmax_scale,
        )

        # Pad output and apply final transformations
        hidden_states = pad_input(attn_output_unpad, indices_q, batch_size, sequence_length)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = hidden_states.type_as(query)

        # Split hidden_states back to instruction and image, apply separate output projections, then merge
        split_results = self._split_instruction_image_features([hidden_states], encoder_seq_lengths, seq_lengths)
        instruct_hidden_states, img_hidden_states = split_results[
            0
        ]  # [B, max_instruct_len, feature_dim], [B, max_img_len, feature_dim]

        # Apply separate output projections for instruction and image
        instruct_projected = self.instruct_out(instruct_hidden_states)  # [B, max_instruct_len, feature_dim]
        img_projected = self.img_out(img_hidden_states)  # [B, max_img_len, feature_dim]

        # Merge back to joint representation
        merged_list = self._concat_instruction_image_features(
            [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths
        )
        hidden_states = merged_list[0]  # [B, max_seq_len, feature_dim]

        # Apply final output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # rank, world_size, worker, num_workers = pytorch_worker_info(None)

        return hidden_states


class BooguImageDoubleStreamSelfAttnProcessor(nn.Module):
    """
    Double-stream self-attention processor without flash attention.

    This processor implements double-stream attention where:
    - Instruction and image features are processed separately to generate QKV
    - QKV are concatenated and processed together for cross-modal attention
    - Uses PyTorch's scaled_dot_product_attention for computation
    - Supports both standard and causal attention masks

    Args:
        head_dim: Dimension of each attention head
        num_attention_heads: Number of attention heads for queries
        num_kv_heads: Number of key-value heads
        qkv_bias: Whether to use bias in QKV linear layers
    """

    def __init__(
        self,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize the double-stream attention processor."""
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "BooguImageDoubleStreamSelfAttnProcessor requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or later."
            )

        # Calculate dimensions
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads

        query_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_kv_heads

        # Initialize separate Q, K, V linear layers for instruction and image
        # Query uses num_attention_heads, Key/Value use num_kv_heads
        self.img_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.img_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        # Additional output projection layers for instruction and image streams
        self.instruct_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the double-stream attention processor.

        Uses Xavier uniform initialization for linear layers and zero initialization for biases.
        """
        # Initialize image stream QKV projection layers
        nn.init.xavier_uniform_(self.img_to_q.weight)
        nn.init.xavier_uniform_(self.img_to_k.weight)
        nn.init.xavier_uniform_(self.img_to_v.weight)

        # Initialize instruction stream QKV projection layers
        nn.init.xavier_uniform_(self.instruct_to_q.weight)
        nn.init.xavier_uniform_(self.instruct_to_k.weight)
        nn.init.xavier_uniform_(self.instruct_to_v.weight)

        # Initialize separate output projection layers
        nn.init.xavier_uniform_(self.instruct_out.weight)
        nn.init.xavier_uniform_(self.img_out.weight)

        # Initialize biases if they exist
        if self.img_to_q.bias is not None:
            nn.init.zeros_(self.img_to_q.bias)
            nn.init.zeros_(self.img_to_k.bias)
            nn.init.zeros_(self.img_to_v.bias)
            nn.init.zeros_(self.instruct_to_q.bias)
            nn.init.zeros_(self.instruct_to_k.bias)
            nn.init.zeros_(self.instruct_to_v.bias)
            nn.init.zeros_(self.instruct_out.bias)
            nn.init.zeros_(self.img_out.bias)

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
        Process double-stream self-attention computation with PyTorch's scaled_dot_product_attention.

        Args:
            attn: Attention module
            img_hidden_states: Image hidden states tensor [B, L_img, D]
            instruct_hidden_states: Instruction hidden states tensor [B, L_instruct, D]
            joint_attention_mask: Combined attention mask [B, L_total]
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

        # Use helper function to concatenate QKV (instruction first, then image)
        img_list = [img_query, img_key, img_value]  # [B, L_img, feature_dim] each
        instruct_list = [
            instruct_query,
            instruct_key,
            instruct_value,
        ]  # [B, L_instruct, feature_dim] each
        concatenated_list = self._concat_instruction_image_features(
            img_list, instruct_list, encoder_seq_lengths, seq_lengths
        )
        query, key, value = concatenated_list  # [B, max_seq_len, feature_dim] each

        # From here, follow exactly the same logic as BooguImageAttnProcessor
        sequence_length = max(seq_lengths)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb, use_real=False)
            key = apply_rotary_emb(key, rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        if joint_attention_mask is not None:
            joint_attention_mask = joint_attention_mask.bool()
            if joint_attention_mask.dim() == 2:
                # Standard mask [B, seq_len] -> [B, 1, 1, seq_len]
                joint_attention_mask = joint_attention_mask.view(batch_size, 1, 1, -1)
            elif joint_attention_mask.dim() == 3:
                # Causal mask [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
                joint_attention_mask = joint_attention_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unsupported joint_attention_mask shape: {joint_attention_mask.shape}")

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # explicitly repeat key and value to match query length, otherwise using enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=joint_attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)

        # Split hidden_states back to instruction and image, apply separate output projections, then merge
        split_results = self._split_instruction_image_features([hidden_states], encoder_seq_lengths, seq_lengths)
        instruct_hidden_states, img_hidden_states = split_results[
            0
        ]  # [B, max_instruct_len, feature_dim], [B, max_img_len, feature_dim]

        # Apply separate output projections for instruction and image
        instruct_projected = self.instruct_out(instruct_hidden_states)  # [B, max_instruct_len, feature_dim]
        img_projected = self.img_out(img_hidden_states)  # [B, max_img_len, feature_dim]

        # Merge back to joint representation
        merged_list = self._concat_instruction_image_features(
            [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths
        )
        hidden_states = merged_list[0]  # [B, max_seq_len, feature_dim]

        # Apply final output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class BooguImageAttnProcessorFlash2Varlen:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling

    Args:
        None
    """

    def __init__(self) -> None:
        """Initialize the attention processor."""
        if not is_flash_attn_available():
            raise ImportError("BooguImageAttnProcessorFlash2Varlen requires flash_attn. Please install flash_attn.")

    def _upad_input(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        num_heads: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[int, int],
    ]:
        """
        Unpad the input tensors for flash attention.

        Args:
            query_layer: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            key_layer: Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            value_layer: Value tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            attention_mask: Attention mask tensor of shape (batch_size, seq_len) or (batch_size, seq_len, seq_len) for causal
            query_length: Length of the query sequence
            num_heads: Number of attention heads

        Returns:
            Tuple containing:
                - Unpadded query tensor
                - Unpadded key tensor
                - Unpadded value tensor
                - Query indices
                - Tuple of cumulative sequence lengths for query and key
                - Tuple of maximum sequence lengths for query and key
        """

        def _get_unpad_data(
            mask_2d: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, int]:
            """Helper function to get unpadding data from a 2D attention mask [B, L]."""
            seqlens_in_batch = mask_2d.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(mask_2d.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch

        # Normalize attention mask: if a causal 3D mask is provided [B, L, L],
        # convert it to a standard 2D padding mask [B, L] with True for valid tokens.
        if attention_mask is not None and attention_mask.dim() == 3:
            B, L, _ = attention_mask.shape
            # For a proper lower-triangular causal mask, all first L positions are valid per sample.
            # However, to be robust, infer per-sample effective lengths from the diagonal.
            diag_valid = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
            lengths = diag_valid.sum(dim=-1, dtype=torch.int32)  # [B]
            mask_2d = torch.zeros(B, L, dtype=torch.bool, device=attention_mask.device)
            for i in range(B):
                if lengths[i].item() > 0:
                    mask_2d[i, : int(lengths[i].item())] = True
        else:
            mask_2d = attention_mask  # already [B, L]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(mask_2d)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # Unpad key and value layers (shared path for both standard and causal cases)
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )

        # Handle different query length cases
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # Use the last query_length positions of the 2D mask
            q_mask = mask_2d[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, q_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

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
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """

        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # Detect if we have a causal mask
        is_causal = False
        if attention_mask is not None and attention_mask.dim() == 3:
            # Check if it's a lower triangular causal mask
            # For efficiency, we only check the first sample
            mask_sample = attention_mask[0]  # [seq_len, seq_len]
            is_causal = torch.allclose(mask_sample, torch.tril(torch.ones_like(mask_sample)))

        # Unpad input for flash attention
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = self._upad_input(query, key, value, attention_mask, sequence_length, attn.heads)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        # Handle different number of heads
        if kv_heads < attn.heads:
            key_states = key_states.repeat_interleave(attn.heads // kv_heads, dim=1)
            value_states = value_states.repeat_interleave(attn.heads // kv_heads, dim=1)

        # Apply flash attention with causal parameter
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=0.0,
            causal=is_causal,  # Use detected causal setting
            softmax_scale=softmax_scale,
        )

        # Pad output and apply final transformations
        hidden_states = pad_input(attn_output_unpad, indices_q, batch_size, sequence_length)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class BooguImageAttnProcessor:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor is optimized for PyTorch 2.0 and implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling

    Args:
        None

    Raises:
        ImportError: If PyTorch version is less than 2.0
    """

    def __init__(self) -> None:
        """Initialize the attention processor."""
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "BooguImageAttnProcessorFlash2Varlen requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or later."
            )

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
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # sdpa expects attn_mask with shape (B, H, Q, K) as boolean (True keeps, False masks)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2:
                # Standard padding mask [B, L] -> [B, 1, 1, L]
                attention_mask = attention_mask.view(batch_size, 1, 1, -1)
            elif attention_mask.dim() == 3:
                # Robust causal + padding mask construction
                # Infer valid lengths from diagonal, then build lower-triangular mask within valid lengths
                B, L, _ = attention_mask.shape
                diag_valid = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
                lengths = diag_valid.sum(dim=-1)  # [B]
                arange_L = torch.arange(L, device=attention_mask.device)
                # Padding masks for queries and keys: shape [B, L]
                q_valid = arange_L.unsqueeze(0) < lengths.unsqueeze(1)
                k_valid = q_valid  # same lengths assumed
                # Lower-triangular causal mask [L, L]
                causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=attention_mask.device))
                # Combine: [B, L, L]
                combined = causal & q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)
                attention_mask = combined.unsqueeze(1)  # [B, 1, L, L]
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # explicitly repeat key and value to match query length, otherwise using enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
