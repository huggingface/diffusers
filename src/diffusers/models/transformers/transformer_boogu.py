"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)


logger = logging.get_logger(__name__)


# ----------------------------- RoPE -----------------------------
def get_freqs_cis(axes_dim: Tuple[int, int, int], axes_lens: Tuple[int, int, int], theta: int) -> List[torch.Tensor]:
    """Build Boogu's rotary-position-embedding frequency table (one entry per axis)."""
    freqs_cis = []
    freqs_dtype = torch.float32
    for d, e in zip(axes_dim, axes_lens):
        emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
        freqs_cis.append(emb)
    return freqs_cis


class BooguImageDoubleStreamRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: Tuple[int, int, int],
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis,
        attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # Create position IDs
        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add text position ids
            position_ids[i, :cap_seq_len] = (
                torch.arange(cap_seq_len, dtype=torch.int32, device=device).unsqueeze(1).expand(-1, 3)
            )

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    if ref_H_tokens * ref_W_tokens != ref_img_len:
                        raise ValueError(
                            f"Reference image token count mismatch: {ref_H_tokens * ref_W_tokens} != {ref_img_len}."
                        )
                    # add image position ids

                    row_ids = (
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device)
                        .unsqueeze(1)
                        .expand(ref_H_tokens, ref_W_tokens)
                        .flatten()
                    )
                    col_ids = (
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device)
                        .unsqueeze(0)
                        .expand(ref_H_tokens, ref_W_tokens)
                        .flatten()
                    )
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            if H_tokens * W_tokens != l_effective_img_len[i]:
                raise ValueError(f"Image token count mismatch: {H_tokens * W_tokens} != {l_effective_img_len[i]}.")

            row_ids = (
                torch.arange(H_tokens, dtype=torch.int32, device=device)
                .unsqueeze(1)
                .expand(H_tokens, W_tokens)
                .flatten()
            )
            col_ids = (
                torch.arange(W_tokens, dtype=torch.int32, device=device)
                .unsqueeze(0)
                .expand(H_tokens, W_tokens)
                .flatten()
            )

            if pe_shift_len + l_effective_img_len[i] != seq_len:
                raise ValueError(
                    f"RoPE position length mismatch: {pe_shift_len + l_effective_img_len[i]} != {seq_len}."
                )
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            batch_size,
            encoder_seq_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size,
            max_ref_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        img_freqs_cis = torch.zeros(
            batch_size,
            max_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        # Calculate combined image sequence lengths (ref_img + img) for each sample
        combined_img_seq_lengths = [
            sum(ref_img_len) + img_len for ref_img_len, img_len in zip(l_effective_ref_img_len, l_effective_img_len)
        ]
        max_combined_img_len = max(combined_img_seq_lengths)

        # Create combined image rotary embeddings
        combined_img_freqs_cis = torch.zeros(
            batch_size,
            max_combined_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(
                l_effective_cap_len,
                l_effective_ref_img_len,
                l_effective_img_len,
                seq_lengths,
            )
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i,
                cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len,
            ]

            # Combined image rotary embeddings: ref_img + img (same order as img_patch_embed_and_refine)
            combined_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            combined_img_freqs_cis[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = freqs_cis[
                i,
                cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len,
            ]

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
            combined_img_freqs_cis,
            combined_img_seq_lengths,
        )


# --------------- Norm / FeedForward / Embedding ----------------
def _torch_swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


swiglu = _torch_swiglu
torch_swiglu = _torch_swiglu


class BooguImageRMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_eps: float,
        norm_elementwise_affine: bool,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )

        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class BooguImageLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


class BooguImageFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        hidden_size (`int`):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        intermediate_size (`int`): The intermediate dimension of the feedforward layer.
        multiple_of (`int`, *optional*): Value to ensure hidden dimension is a multiple
            of this value.
        ffn_dim_multiplier (float, *optional*): Custom multiplier for hidden
            dimension. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.swiglu = swiglu

        # custom hidden_size factor multiplier
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            inner_dim,
            dim,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )

    def forward(self, x):
        h1, h2 = self.linear_1(x), self.linear_3(x)
        swiglu_fn = torch_swiglu if torch.compiler.is_compiling() else self.swiglu
        return self.linear_2(swiglu_fn(h1, h2))


class BooguImageCombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        instruction_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )

        self.caption_embedder = nn.Sequential(
            RMSNorm(instruction_feat_dim, eps=norm_eps),
            nn.Linear(instruction_feat_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(instruction_hidden_states)
        return time_embed, caption_embed


# ----------------------- Attention processors ------------------
def apply_rotary_emb(x, freqs_cis, use_real=True, **kwargs):
    # use_real=True path delegates to the shared diffusers implementation.
    # use_real=False (Lumina-style) uses explicit dim to handle 0-element tensors.
    if use_real:
        from diffusers.models.embeddings import apply_rotary_emb as _apply

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


class BooguImageAttention(torch.nn.Module, AttentionModuleMixin):
    """
    Attention module for Boogu-Image. Holds the per-head layers (``to_q`` / ``to_k`` / ``to_v`` / ``norm_q`` /
    ``norm_k`` / ``to_out``) and defers the computation to a stateless processor.

    Two layouts are supported:

    - Single-stream self-attention (``has_qkv=True``, the default): owns ``to_q`` / ``to_k`` / ``to_v``. Paired with
      [`BooguImageAttnProcessor`].
    - Double-stream joint attention (``has_qkv=False``): the per-stream QKV / output projections live on the
      stateful [`BooguImageDoubleStreamSelfAttnProcessor`] (checkpoint keys ``...processor.img_to_q`` /
      ``...processor.instruct_to_q`` / ``...processor.img_out`` / ``...processor.instruct_out``), so this module only
      holds the shared ``norm_q`` / ``norm_k`` / ``to_out``.
    """

    _default_processor_cls = None
    _available_processors = []

    def __init__(
        self,
        query_dim: int,
        heads: int,
        kv_heads: int,
        dim_head: int,
        eps: float = 1e-5,
        bias: bool = False,
        out_bias: bool = False,
        has_qkv: bool = True,
        processor=None,
    ) -> None:
        super().__init__()

        self.inner_dim = dim_head * heads
        self.inner_kv_dim = dim_head * kv_heads
        self.query_dim = query_dim
        self.out_dim = query_dim
        self.heads = heads
        self.scale = dim_head**-0.5

        if has_qkv:
            self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
            self.to_k = torch.nn.Linear(query_dim, self.inner_kv_dim, bias=bias)
            self.to_v = torch.nn.Linear(query_dim, self.inner_kv_dim, bias=bias)

        # QK norm reproduces ``Attention(qk_norm="rms_norm")`` exactly (diffusers' float32-upcasting RMSNorm).
        self.norm_q = DiffusersRMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = DiffusersRMSNorm(dim_head, eps=eps, elementwise_affine=True)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(0.0))

        self.set_processor(processor)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.processor(self, *args, **kwargs)


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
        if len(img_hidden_states_list) != len(instruct_hidden_states_list):
            raise ValueError(
                f"Length mismatch: img_list={len(img_hidden_states_list)}, "
                f"instruct_list={len(instruct_hidden_states_list)}"
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
        attn: "BooguImageAttention",
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        joint_attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        encoder_seq_lengths: List[int] = None,  # [B] - Instruction sequence lengths for each sample
        seq_lengths: List[int] = None,  # [B] - Total sequence lengths for each sample
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

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=_prepare_attn_mask(joint_attention_mask, batch_size),
            scale=attn.scale,
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
        attn: "BooguImageAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process single-stream self-attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor (same as hidden_states for self-attention)
            attention_mask: Optional bool padding mask [B, L]
            image_rotary_emb: Optional rotary embeddings

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size = hidden_states.shape[0]

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

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=_prepare_attn_mask(attention_mask, batch_size),
            scale=attn.scale,
            enable_gqa=kv_heads < attn.heads,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# Registered here (after both processors are defined) so `BooguImageAttention()` defaults to the single-stream
# stateless processor, while the double-stream processor remains available.
BooguImageAttention._default_processor_cls = BooguImageAttnProcessor
BooguImageAttention._available_processors = [
    BooguImageAttnProcessor,
    BooguImageDoubleStreamSelfAttnProcessor,
]


class BooguImageTransformerBlock(nn.Module):
    """
    Basic Boogu-Image transformer block: attention + MLP + RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        # Initialize attention layer
        self.attn = BooguImageAttention(
            query_dim=dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            has_qkv=True,
            processor=BooguImageAttnProcessor(),
        )

        # Initialize feed-forward network
        self.feed_forward = BooguImageFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        # Initialize normalization layers
        if modulation:
            self.norm1 = BooguImageRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            hidden_states: Input hidden states tensor
            attention_mask: Attention mask tensor
            image_rotary_emb: Rotary embeddings for image tokens
            temb: Optional timestep embedding tensor

        Returns:
            torch.Tensor: Output hidden states after transformer block processing
        """
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class BooguImageDoubleStreamTransformerBlock(nn.Module):
    """
    Boogu-Image double-stream block.
    Here "double-stream" is the same idea as a "dual-stream" layer:
    instruction tokens and image tokens are processed in parallel streams.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the double stream transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.modulation = modulation
        self.hidden_size = dim

        double_stream_processor = BooguImageDoubleStreamSelfAttnProcessor(
            head_dim=self.head_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=False,
        )

        # Image stream components.
        # Joint instruction<->image attention: the per-stream QKV / output projections live on the processor,
        # so this module carries only the shared norm_q / norm_k / to_out (has_qkv=False).
        self.img_instruct_attn = BooguImageAttention(
            query_dim=dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            has_qkv=False,
            processor=double_stream_processor,
        )

        self.img_self_attn = BooguImageAttention(
            query_dim=dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            has_qkv=True,
            processor=BooguImageAttnProcessor(),
        )

        self.img_feed_forward = BooguImageFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Image modulation terms: cross-attn, MLP, self-attn.
            self.img_norm1 = BooguImageRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
            self.img_norm2 = BooguImageRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
            self.img_norm3 = BooguImageRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.img_norm1 = RMSNorm(dim, eps=norm_eps)
            self.img_norm2 = RMSNorm(dim, eps=norm_eps)
            self.img_norm3 = RMSNorm(dim, eps=norm_eps)

        self.img_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.img_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        # Instruction stream components.
        self.instruct_feed_forward = BooguImageFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Instruction modulation terms: cross-attn, MLP.
            self.instruct_norm1 = BooguImageRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
            self.instruct_norm2 = BooguImageRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
        else:
            self.instruct_norm1 = RMSNorm(dim, eps=norm_eps)
            self.instruct_norm2 = RMSNorm(dim, eps=norm_eps)

        self.instruct_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        img_hidden_states: torch.Tensor,  # [B, L_img, D] - Image tokens (ref_img + noise_img)
        instruct_hidden_states: torch.Tensor,  # [B, L_instruct, D] - Instruction tokens
        img_attention_mask: torch.Tensor,  # [B, L_img] - Attention mask for [ref_img + noise_img]
        joint_attention_mask: torch.Tensor,  # [B, L_total] - Combined attention mask for [instruct + img]
        image_rotary_emb: torch.Tensor,  # [B, L_img, head_dim] - Rotary embeddings for [ref_img + noise_img]
        rotary_emb: torch.Tensor,  # [B, L_total, head_dim] - Rotary embeddings for [instruct + img]
        temb: Optional[torch.Tensor] = None,  # [B, 1024] - Timestep embeddings
        encoder_seq_lengths: List[int] = None,  # [B] - Instruction sequence lengths for each sample
        seq_lengths: List[int] = None,  # [B] - Total sequence lengths for each sample
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one dual-stream (double-stream) block step.
        Returns updated `(img_hidden_states, instruct_hidden_states)`.
        """
        if self.modulation and temb is None:
            raise ValueError("temb must be provided when modulation is enabled")

        # Extract dimensions
        batch_size = img_hidden_states.shape[0]
        L_instruct = instruct_hidden_states.shape[1]  # Instruction sequence length
        L_img = img_hidden_states.shape[1]  # Image sequence length (ref_img + noise_img)

        if self.modulation:
            # Step 1: modulation for both streams.
            img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img_hidden_states, temb)
            img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
            img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

            (
                instruct_norm1_out,
                instruct_gate_msa,
                instruct_scale_mlp,
                instruct_gate_mlp,
            ) = self.instruct_norm1(instruct_hidden_states, temb)
            instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)

            # Step 2: joint attention on [instruct + img]. The module dispatches to its double-stream processor.
            joint_attn_out = self.img_instruct_attn(
                img_hidden_states=img_norm1_out,
                instruct_hidden_states=instruct_norm1_out,
                joint_attention_mask=joint_attention_mask,
                rotary_emb=rotary_emb,
                encoder_seq_lengths=encoder_seq_lengths,
                seq_lengths=seq_lengths,
            )

            # Split back into instruction/image segments.
            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            # Step 3: image self-attention.
            img_self_attn_out = self.img_self_attn(
                hidden_states=img_norm3_out,
                encoder_hidden_states=img_norm3_out,
                attention_mask=img_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

            # Step 4: residual updates.
            img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(1).tanh() * self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(1).tanh() * self.img_self_attn_norm(
                img_self_attn_out
            )

            img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
            img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(1).tanh() * self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + instruct_gate_msa.unsqueeze(
                1
            ).tanh() * self.instruct_attn_norm(instruct_attn_out)

            instruct_mlp_input = (
                1 + instruct_scale_mlp.unsqueeze(1)
            ) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
            instruct_hidden_states = instruct_hidden_states + instruct_gate_mlp.unsqueeze(
                1
            ).tanh() * self.instruct_ffn_norm2(instruct_mlp_out)

        else:
            # Non-modulated branch used by context-style blocks.
            img_norm1_out = self.img_norm1(img_hidden_states)
            img_norm3_out = self.img_norm3(img_hidden_states)
            instruct_norm1_out = self.instruct_norm1(instruct_hidden_states)

            # Same processor path as above.
            joint_attn_out = self.img_instruct_attn(
                img_hidden_states=img_norm1_out,
                instruct_hidden_states=instruct_norm1_out,
                joint_attention_mask=joint_attention_mask,
                rotary_emb=rotary_emb,
                encoder_seq_lengths=encoder_seq_lengths,
                seq_lengths=seq_lengths,
            )

            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            img_self_attn_out = self.img_self_attn(
                hidden_states=img_norm3_out,
                encoder_hidden_states=img_norm3_out,
                attention_mask=img_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

            img_hidden_states = img_hidden_states + self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + self.img_self_attn_norm(img_self_attn_out)
            img_norm2_out = self.img_norm2(img_hidden_states)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_norm2_out))
            img_hidden_states = img_hidden_states + self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + self.instruct_attn_norm(instruct_attn_out)
            instruct_norm2_out = self.instruct_norm2(instruct_hidden_states)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_norm2_out))
            instruct_hidden_states = instruct_hidden_states + self.instruct_ffn_norm2(instruct_mlp_out)

        return img_hidden_states, instruct_hidden_states


class BooguImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Boogu-Image transformer with mixed stream topology.
    Early layers use double-stream (aka dual-stream) processing, then switch
    to single-stream joint processing.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "BooguImageTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
    ]
    _repeated_blocks = [
        "BooguImageTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
    ]
    _skip_layerwise_casting_patterns = ["x_embedder", "norm", "embedding"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_double_stream_layers: int = 2,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: Tuple[int, int, int] = (40, 40, 40),
        axes_lens: Tuple[int, int, int] = (2048, 1664, 1664),
        instruction_feature_configs: Dict[str, Any] = {
            "instruction_feat_dim": 1024,
            "reduce_type": "mean",
            "num_instruction_feat_layers": 1,
        },
        timestep_scale: float = 1.0,
    ) -> None:
        """Initialize the Boogu-Image mixed single-double stream transformer model."""
        super().__init__()

        # Validate configuration
        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )

        if num_double_stream_layers > num_layers:
            raise ValueError(
                f"num_double_stream_layers ({num_double_stream_layers}) cannot be greater than "
                f"num_layers ({num_layers})"
            )

        self.out_channels = out_channels or in_channels
        self.num_double_stream_layers = num_double_stream_layers
        self.num_single_stream_layers = num_layers - num_double_stream_layers
        self.instruction_feature_configs = instruction_feature_configs
        self.preprocessed_instruction_feat_dim = self.cal_preprocessed_instruction_feat_dim(
            instruction_feature_configs
        )

        # Initialize embeddings
        self.rope_embedder = BooguImageDoubleStreamRotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = BooguImageCombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            instruction_feat_dim=self.preprocessed_instruction_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        # Refiner layers.
        self.noise_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.ref_image_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # Mixed architecture: dual-stream first, then single-stream.
        # Here "double-stream" and "dual-stream" mean the same thing.
        self.double_stream_layers = nn.ModuleList(
            [
                BooguImageDoubleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_double_stream_layers)
            ]
        )

        # Single-stream layers process the fused sequence; they reuse BooguImageTransformerBlock.
        self.single_stream_layers = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(self.num_single_stream_layers)
            ]
        )

        # Output norm and projection.
        self.norm_out = BooguImageLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Distinguish multiple reference images.
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))  # support max 5 ref images

        self.gradient_checkpointing = False

    def img_patch_embed_and_refine(
        self,
        hidden_states,
        ref_image_hidden_states,
        padded_img_mask,
        padded_ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    ):
        """Embed image patches and run the refiner blocks."""
        batch_size = len(hidden_states)
        max_combined_img_len = max(
            [img_len + sum(ref_img_len) for img_len, ref_img_len in zip(l_effective_img_len, l_effective_ref_img_len)]
        )

        hidden_states = self.x_embedder(hidden_states)
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)

        for i in range(batch_size):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift : shift + ref_img_len, :] = (
                    ref_image_hidden_states[i, shift : shift + ref_img_len, :] + self.image_index_embedding[j]
                )
                shift += ref_img_len

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb)

        flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))
        num_ref_images = len(flat_l_effective_ref_img_len)
        max_ref_img_len = max(flat_l_effective_ref_img_len)

        batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images, max_ref_img_len, dtype=torch.bool)
        batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, self.config.hidden_size
        )
        batch_ref_img_rotary_emb = hidden_states.new_zeros(
            num_ref_images,
            max_ref_img_len,
            ref_img_rotary_emb.shape[-1],
            dtype=ref_img_rotary_emb.dtype,
        )
        batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)

        # Flatten reference images into a temporary batch.
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                batch_ref_img_mask[idx, :ref_img_len] = True
                batch_ref_image_hidden_states[idx, :ref_img_len] = ref_image_hidden_states[
                    i, shift : shift + ref_img_len
                ]
                batch_ref_img_rotary_emb[idx, :ref_img_len] = ref_img_rotary_emb[i, shift : shift + ref_img_len]
                batch_temb[idx] = temb[i]
                shift += ref_img_len
                idx += 1

        # Refine each reference-image sample.
        for layer in self.ref_image_refiner:
            batch_ref_image_hidden_states = layer(
                batch_ref_image_hidden_states,
                batch_ref_img_mask,
                batch_ref_img_rotary_emb,
                batch_temb,
            )

        # Restore reference-image sequence layout.
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                ref_image_hidden_states[i, shift : shift + ref_img_len] = batch_ref_image_hidden_states[
                    idx, :ref_img_len
                ]
                shift += ref_img_len
                idx += 1

        combined_img_hidden_states = hidden_states.new_zeros(batch_size, max_combined_img_len, self.config.hidden_size)
        for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
            combined_img_hidden_states[i, : sum(ref_img_len)] = ref_image_hidden_states[i, : sum(ref_img_len)]
            combined_img_hidden_states[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = hidden_states[i, :img_len]

        return combined_img_hidden_states

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        """Flatten patch tokens and pad to batched sequences."""
        batch_size = len(hidden_states)
        p = self.config.patch_size
        device = hidden_states[0].device

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None:
            ref_img_sizes = [
                [(img.size(1), img.size(2)) for img in imgs] if imgs is not None else None
                for imgs in ref_image_hidden_states
            ]
            l_effective_ref_img_len = [
                [(ref_img_size[0] // p) * (ref_img_size[1] // p) for ref_img_size in _ref_img_sizes]
                if _ref_img_sizes is not None
                else [0]
                for _ref_img_sizes in ref_img_sizes
            ]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # Reference-image patch embeddings.
        flat_ref_img_hidden_states = []
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                imgs = []
                for ref_img in ref_image_hidden_states[i]:
                    C, H, W = ref_img.size()
                    # "c (h p1) (w p2) -> (h w) (p1 p2 c)"
                    ref_img = ref_img.reshape(C, H // p, p, W // p, p)
                    ref_img = ref_img.permute(1, 3, 2, 4, 0)
                    ref_img = ref_img.reshape((H // p) * (W // p), p * p * C)
                    imgs.append(ref_img)

                img = torch.cat(imgs, dim=0)
                flat_ref_img_hidden_states.append(img)
            else:
                flat_ref_img_hidden_states.append(None)

        # Noise-image patch embeddings.
        flat_hidden_states = []
        for i in range(batch_size):
            img = hidden_states[i]
            C, H, W = img.size()

            # "c (h p1) (w p2) -> (h w) (p1 p2 c)"
            img = img.reshape(C, H // p, p, W // p, p)
            img = img.permute(1, 3, 2, 4, 0)
            img = img.reshape((H // p) * (W // p), p * p * C)
            flat_hidden_states.append(img)

        padded_ref_img_hidden_states = torch.zeros(
            batch_size,
            max_ref_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_ref_img_mask = torch.zeros(batch_size, max_ref_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                padded_ref_img_hidden_states[i, : sum(l_effective_ref_img_len[i])] = flat_ref_img_hidden_states[i]
                padded_ref_img_mask[i, : sum(l_effective_ref_img_len[i])] = True

        padded_hidden_states = torch.zeros(
            batch_size,
            max_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_img_mask = torch.zeros(batch_size, max_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            padded_hidden_states[i, : l_effective_img_len[i]] = flat_hidden_states[i]
            padded_img_mask[i, : l_effective_img_len[i]] = True

        return (
            padded_hidden_states,
            padded_ref_img_hidden_states,
            padded_img_mask,
            padded_ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        )

    def cal_preprocessed_instruction_feat_dim(self, instruction_feature_configs: Dict[str, Any]):
        num_instruction_feat_layers = max(instruction_feature_configs.get("num_instruction_feat_layers", 1), 1)
        instruction_feat_dim = instruction_feature_configs.get("instruction_feat_dim", 4096)
        reduce_type = instruction_feature_configs.get("reduce_type", "concat")
        if "cat" in reduce_type.lower():
            return num_instruction_feat_layers * instruction_feat_dim
        elif "mean" in reduce_type.lower():
            return instruction_feat_dim
        else:
            raise ValueError(f"Invalid reduce_type: {reduce_type}")

    def preprocess_instruction_hidden_states(
        self, raw_instruction_hidden_states, instruction_feature_configs: Dict[str, Any]
    ):
        num_instruction_feat_layers = max(instruction_feature_configs.get("num_instruction_feat_layers", 1), 1)
        reduce_type = instruction_feature_configs.get("reduce_type", "concat")

        instruction_hidden_states = None
        if isinstance(raw_instruction_hidden_states, torch.Tensor):
            instruction_hidden_states = raw_instruction_hidden_states
        elif isinstance(raw_instruction_hidden_states, (list, tuple)):
            if len(raw_instruction_hidden_states) != num_instruction_feat_layers:
                raise ValueError(
                    f"Expected {num_instruction_feat_layers} instruction-feature layers, "
                    f"got {len(raw_instruction_hidden_states)}."
                )
            if "cat" in reduce_type.lower():
                instruction_hidden_states = torch.cat(raw_instruction_hidden_states, dim=-1)
            elif "mean" in reduce_type.lower():
                instruction_hidden_states = torch.mean(torch.stack(raw_instruction_hidden_states), dim=0)
            else:
                raise ValueError(f"Invalid reduce_type: {reduce_type}")
        else:
            raise ValueError(
                f"Invalid type of raw_instruction_hidden_states, expected torch.Tensor or list, but got {type(raw_instruction_hidden_states)}"
            )

        if self.preprocessed_instruction_feat_dim != instruction_hidden_states.shape[-1]:
            raise ValueError(
                f"Instruction feature dim mismatch: expected {self.preprocessed_instruction_feat_dim}, "
                f"got {instruction_hidden_states.shape[-1]}."
            )

        return instruction_hidden_states

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass:
        context/refiner -> dual-stream (double-stream) -> fusion -> single-stream -> projection.
        """
        instruction_hidden_states = self.preprocess_instruction_hidden_states(
            instruction_hidden_states, self.instruction_feature_configs
        )

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

        # === 1. Initial processing (same as original Boogu-Image) ===
        batch_size = len(hidden_states)
        is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

        if is_hidden_states_tensor:
            if hidden_states.ndim != 4:
                raise ValueError(f"Expected hidden_states with 4 dims [B, C, H, W], got ndim={hidden_states.ndim}.")
            hidden_states = list(hidden_states)

        device = hidden_states[0].device

        # Timestep and instruction embedding.
        temb, instruction_hidden_states = self.time_caption_embed(
            timestep, instruction_hidden_states, hidden_states[0].dtype
        )

        # Flatten and pad token sequences.
        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        # Build rotary embeddings and sequence lengths.
        (
            context_rotary_emb,
            ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
            combined_img_rotary_emb,
            combined_img_seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            instruction_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        # Context refinement.
        for layer in self.context_refiner:
            instruction_hidden_states = layer(
                instruction_hidden_states,
                instruction_attention_mask,
                context_rotary_emb,
            )

        # Image patch embedding and refinement.
        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        # Dual-stream (double-stream) stage.
        instruct_hidden_states = instruction_hidden_states
        img_hidden_states = combined_img_hidden_states

        # Joint mask for [instruct + image].
        max_seq_len = max(seq_lengths)
        joint_attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            joint_attention_mask[i, :seq_len] = True

        # Run dual-stream blocks.
        if self.num_double_stream_layers > 0:
            # Image-only mask for [ref + noise].
            max_img_len = max(combined_img_seq_lengths)
            img_attention_mask = hidden_states.new_zeros(batch_size, max_img_len, dtype=torch.bool)
            for i, img_seq_len in enumerate(combined_img_seq_lengths):
                img_attention_mask[i, :img_seq_len] = True

            for layer in self.double_stream_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    img_hidden_states, instruct_hidden_states = self._gradient_checkpointing_func(
                        layer,
                        img_hidden_states,
                        instruct_hidden_states,
                        img_attention_mask,
                        joint_attention_mask,
                        combined_img_rotary_emb,
                        rotary_emb,
                        temb,
                        encoder_seq_lengths,
                        seq_lengths,
                    )
                else:
                    img_hidden_states, instruct_hidden_states = layer(
                        img_hidden_states,
                        instruct_hidden_states,
                        img_attention_mask,
                        joint_attention_mask,
                        combined_img_rotary_emb,
                        rotary_emb,
                        temb,
                        encoder_seq_lengths,
                        seq_lengths,
                    )

        # Fuse streams to joint sequence.
        joint_hidden_states = hidden_states.new_zeros(batch_size, max(seq_lengths), self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            joint_hidden_states[i, :encoder_seq_len] = instruct_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = img_hidden_states[i, : seq_len - encoder_seq_len]

        # Single-stream stage.
        hidden_states = joint_hidden_states

        for layer in self.single_stream_layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer, hidden_states, joint_attention_mask, rotary_emb, temb
                )
            else:
                hidden_states = layer(hidden_states, joint_attention_mask, rotary_emb, temb)

        # Output projection.
        hidden_states = self.norm_out(hidden_states, temb)

        # Reshape back to image format.
        p = self.config.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
            height, width = img_size
            img_tokens = hidden_states[i][seq_len - img_len : seq_len]
            # "(h w) (p1 p2 c) -> c (h p1) (w p2)"
            h, w = height // p, width // p
            c = img_tokens.shape[-1] // (p * p)
            img_output = img_tokens.reshape(h, w, p, p, c)
            img_output = img_output.permute(4, 0, 2, 1, 3)
            img_output = img_output.reshape(c, h * p, w * p)
            output.append(img_output)

        if is_hidden_states_tensor:
            output = torch.stack(output, dim=0)

        # Reset LoRA scaling.
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)
