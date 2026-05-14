import inspect
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..activations import get_activation
from ..attention import AttentionModuleMixin
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Based on transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Based on transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Ultimately from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DreamAttnProcessor:
    _attention_backend = None

    def __call__(
        self,
        attn: "DreamAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        # TODO: can caching be implemented in diffusers like it is in the original code?
        # hidden_states shape: (batch_size, seq_len, hidden_dim) = [B, L, D]
        batch_size, query_len, _ = hidden_states.size()
        query = attn.to_q(hidden_states)  # [B, L, D]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)    # [B, L, D] --> [B, L, D_KV]
        value = attn.to_v(encoder_hidden_states)  # [B, L, D] --> [B, L, D_KV]

        # TODO: call attn.head_to_batch_dim here instead???
        # original code sends [batch_size, seq_len, hidden_dim] to [batch_size, num_heads, seq_len, head_dim]
        # batch_to_head_dim instead sends it to [batch_size // num_heads, seq_len, dim * heads]
        query = query.view(batch_size, query_len, attn.heads, attn.head_dim).transpose(1, 2)     # [B, N, L, H]
        key = key.view(batch_size, query_len, attn.kv_heads, attn.head_dim).transpose(1, 2)      # [B, N_KV, L, H]
        value = value.view(batch_size, query_len, attn.kv_heads, attn.head_dim).transpose(1, 2)  # [B, N_KV, L, H]

        if rotary_emb is not None:
            # TODO: rewrite in terms of embeddings.apply_rotary_emb???
            query, key = apply_rotary_pos_emb(query, key, rotary_emb[0], rotary_emb[1])

        # Repeat KV heads if attn.kv_heads < attn.heads
        key = repeat_kv(key, attn.kv_groups)  # [B, N_KV, L, H] --> [B, N, L, H]
        value = repeat_kv(value, attn.kv_groups)  # [B, N_KV, L, H] --> [B, N, L, H]

        # TODO: call dispatch_attention_fn here to dispatch the implementation to a backend? e.g. FlashAttn
        # hidden_states = dispatch_attention_fn(
        #     query, key, value, attn_mask=attention_mask, backend=self._attention_backend
        # )
        # TODO: call attn.get_attention_scores here instead???
        # For example, this would handle upcasting the attention operation for us
        attn_scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(attn.head_dim)  # [B, N, L, L]
        if attention_mask is not None:
            # Not matter the length, we just slice the attention mask
            # TODO: check shapes here, is attention_mask expected to be a causal (upper-triangular) mask of shape
            # [B, 1, L, L]????
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_scores = attn_scores + causal_mask

        # TODO: could use something like torch.autocast from torch AMP here
        if attn.upcast_softmax:
            original_dtype = attn_scores.dtype
            attn_scores = attn_scores.to(dtype=torch.float32)
        attn_scores = F.softmax(attn_scores, dim=-1)
        if attn.upcast_softmax:
            attn_scores = attn_scores.to(dtype=original_dtype)
        attn_scores = F.dropout(attn_scores, p=attn.dropout, training=attn.training)
        hidden_states = torch.matmul(attn_scores, value)  # [B, N, L, H]

        # TODO: call attn.batch_to_head_dim here instead????
        hidden_states = hidden_states.transpose(1, 2).contiguous()  # [B, L, N, H]
        hidden_states = hidden_states.reshape(batch_size, query_len, attn.inner_dim)  # [B, L, D]

        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class DreamSdpaAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "DreamAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        # TODO: can caching be implemented in diffusers like it is in the original code?
        # hidden_states shape: (batch_size, seq_len, hidden_dim) = [B, L, D]
        batch_size, query_len, _ = hidden_states.size()
        query = attn.to_q(hidden_states)  # [B, L, D]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)    # [B, L, D] --> [B, L, D_KV]
        value = attn.to_v(encoder_hidden_states)  # [B, L, D] --> [B, L, D_KV]

        # TODO: call attn.head_to_batch_dim here instead???
        # original code sends [batch_size, seq_len, hidden_dim] to [batch_size, num_heads, seq_len, head_dim]
        # batch_to_head_dim instead sends it to [batch_size // num_heads, seq_len, dim * heads]
        query = query.view(batch_size, query_len, attn.heads, attn.head_dim).transpose(1, 2)     # [B, N, L, H]
        key = key.view(batch_size, query_len, attn.kv_heads, attn.head_dim).transpose(1, 2)      # [B, N_KV, L, H]
        value = value.view(batch_size, query_len, attn.kv_heads, attn.head_dim).transpose(1, 2)  # [B, N_KV, L, H]

        if rotary_emb is not None:
            # TODO: rewrite in terms of embeddings.apply_rotary_emb???
            query, key = apply_rotary_pos_emb(query, key, rotary_emb[0], rotary_emb[1])

        # Repeat KV heads if attn.kv_heads < attn.heads
        key = repeat_kv(key, attn.kv_groups)  # [B, N_KV, L, H] --> [B, N, L, H]
        value = repeat_kv(value, attn.kv_groups)  # [B, N_KV, L, H] --> [B, N, L, H]

        # TODO: call dispatch_attention_fn here to dispatch the implementation to a backend? e.g. FlashAttn
        # hidden_states = dispatch_attention_fn(
        #     query, key, value, attn_mask=attention_mask, backend=self._attention_backend
        # )
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=attn.dropout if attn.training else 0.0,
            is_causal=False,  # hard-coded like in original code
        )

        # TODO: call attn.batch_to_head_dim here instead????
        hidden_states = hidden_states.transpose(1, 2).contiguous()  # [B, L, N, H]
        hidden_states = hidden_states.reshape(batch_size, query_len, attn.inner_dim)  # [B, L, D]

        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class DreamAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = DreamAttnProcessor
    _available_processors = [
        DreamAttnProcessor,
        DreamSdpaAttnProcessor,
    ]

    def __init__(
        self,
        query_dim: int,  # 3584 in Dream-7B???
        heads: int = 28,
        kv_heads: Optional[int] = 4,
        dim_head: int = 128,  # 3584 // 28 = 128
        dropout: float = 0.0,
        bias: bool = True,
        out_bias: bool = False,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        upcast_softmax: bool = True,
        processor=None,
    ):
        super().__init__()

        self.query_dim = query_dim
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads  # num_heads in original code
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.kv_inner_dim = dim_head * self.kv_heads
        self.kv_groups = self.heads // self.kv_heads  # num_key_value_groups

        self.dropout = dropout
        self.use_bias = bias
        self.upcast_softmax = upcast_softmax

        # q_proj, k_proj, v_proj in original code
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.kv_inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.kv_inner_dim, bias=bias)

        # o_proj in original code
        self.to_out = torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {}
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


# Based on transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Dream
class DreamRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float = 1000000.0,  # Not 10000.0 as is standard
    ):
        super().__init__()
        self.theta = theta

        # Default RoPE initialization
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2) / dim))  # [D // 2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: x is only used for its device and dtype and not its actual contents
        # position_ids shape: [B, S]
        # D --> dim --> attention_head_dim
        # TODO: rewrite in terms of get_1d_rotary_pos_embed?
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)  # [B, D // 2, 1]?
        position_ids_expanded = position_ids[:, None, :]  # [B, 1, S]?

        # Force to float32 following https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # [B, S, D // 2]?
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)

        return cos, sin


# Based on transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Dream
class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DreamRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Based on transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Dream
class DreamMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: Optional[int] = 4,  # mult is not an integer for Dream-7B - it's 18944 / 3584 = 37 / 7
        dropout: float = 0.0,  # dropout is actually not used in the Dream MLP
        activation_fn: str = "silu",
        inner_dim = 18944,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = dim
        if inner_dim is None:
            inner_dim = int(dim * mult)
        self.intermediate_size = inner_dim
        self.dim_out = dim_out if dim_out is not None else dim

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.act_fn = get_activation(activation_fn)

        self.down_proj = nn.Linear(self.intermediate_size, self.dim_out, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expanded_hidden_states = self.up_proj(hidden_states)

        gated_hidden_states = self.gate_proj(hidden_states)
        gated_hidden_states = self.act_fn(gated_hidden_states)

        hidden_states = gated_hidden_states * expanded_hidden_states
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class DreamTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_attention_kv_heads: Optional[int],
        attention_head_dim: int,
        ff_intermediate_dim: int = 18944,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Input LayerNorm
        self.norm1 = DreamRMSNorm(dim, eps=eps)

        self.attn = DreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_kv_heads,
            dim_head=attention_head_dim,
            processor=DreamSdpaAttnProcessor(),
        )

        # Post-attention LayerNorm
        self.norm2 = DreamRMSNorm(dim, eps=eps)
        self.ff = DreamMLP(dim=dim, dim_out=dim, inner_dim=ff_intermediate_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,  # temb is not used in Dream (time-invariant model)
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # hidden_states shape: [batch_size, seq_len, hidden_dim] = [B, L, D]
        residual = hidden_states

        # Input LayerNorm
        hidden_states = self.norm1(hidden_states)

        # Attention + shortcut connection
        hidden_states = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
        )
        hidden_states = residual + hidden_states

        # Fully-connected + shortcut connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DreamTransformer1DModel(
    ModelMixin,
    ConfigMixin,  # TODO: add other mixins as necessary
):
    """
    The diffusion transformer model used in the Dream-7B diffusion LLM.

    See https://hkunlp.github.io/blog/2025/dream/. The original transformers-style implementation is at
    https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/modeling_dream.py.

    Args:
        TODO
    """

    _supports_gradient_checkpointing = False
    _no_split_modules = ["DreamTransformerBlock"]
    _skip_layerwise_casting_patterns = ["embedding", "norm"]
    _repeated_blocks = ["DreamTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 28,
        num_attention_kv_heads: Optional[int] = 4,
        ff_intermediate_dim: int = 18944,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        vocab_size: int = 152064,
        pad_token_id: int = 151643,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim  # hidden_size = 3584 in original code
        self.pad_token_id = pad_token_id

        # TODO: can we replace this with a diffusers embedding module?
        self.token_embedding = nn.Embedding(vocab_size, self.inner_dim, self.pad_token_id)
        self.rotary_embedding = DreamRotaryEmbedding(dim=attention_head_dim, theta=rope_theta)

        self.transformer_blocks = nn.ModuleList(
            [
                DreamTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_attention_kv_heads=num_attention_kv_heads,
                    attention_head_dim=attention_head_dim,
                    ff_intermediate_dim=ff_intermediate_dim,
                    eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = DreamRMSNorm(self.inner_dim, eps=rms_norm_eps)
        self.lm_head = nn.Linear(self.inner_dim, vocab_size, bias=False)

    def embed_tokens(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(text_ids)

    def forward(
        self,
        text_ids: torch.Tensor = None,
        hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.LongTensor] = None,  # not used by Dream (time-invariant model)
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`DreamTransformer1DModel`] forward method.

        Args:
            text_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The indices of the input text tokens.
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                The already embedded hidden states for the transformer. This is analogous to `inputs_embeds` for a
                transformers model.
            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The indices of the positions of each token within the input. Will be created if not supplied.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step. Not used currently as Dream is a time-invariant model.
            attention_mask (`torch.Tensor`, *optional*):
                An optional attention mask. This is mainly useful for training, as Dream is trained with an attention
                mask annealing strategy.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # text_ids shape: [B, L]
        if hidden_states is None:
            # Embed text tokens
            hidden_states = self.token_embedding(text_ids)  # [B, L] --> [B, L, D]

        # Create position_ids if not supplied
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0)  # [L] --> [1, L]
        # Get RoPE embeddings (shared across all layers)
        rotary_emb = self.rotary_embedding(hidden_states, position_ids)

        # Transformer decoder layers
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,)

        # TODO: arguably the input is not 2D here since it is of shape (batch_size, seq_len, vocab_size)
        return Transformer2DModelOutput(sample=logits)
