# Copyright 2025 Nucleus-Image Team, The HuggingFace Team. All rights reserved.
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

import functools
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import AttentionMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, RMSNorm


logger = logging.get_logger(__name__)


# Copied from diffusers.models.transformers.transformer_qwenimage.apply_rotary_emb_qwen with qwen->nucleus
def _apply_rotary_emb_nucleus(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def _compute_text_seq_len_from_mask(
    encoder_hidden_states: torch.Tensor, encoder_hidden_states_mask: torch.Tensor | None
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    batch_size, text_seq_len = encoder_hidden_states.shape[:2]
    if encoder_hidden_states_mask is None:
        return text_seq_len, None, None

    if encoder_hidden_states_mask.shape[:2] != (batch_size, text_seq_len):
        raise ValueError(
            f"`encoder_hidden_states_mask` shape {encoder_hidden_states_mask.shape} must match "
            f"(batch_size, text_seq_len)=({batch_size}, {text_seq_len})."
        )

    if encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)

    position_ids = torch.arange(text_seq_len, device=encoder_hidden_states.device, dtype=torch.long)
    active_positions = torch.where(encoder_hidden_states_mask, position_ids, position_ids.new_zeros(()))
    has_active = encoder_hidden_states_mask.any(dim=1)
    per_sample_len = torch.where(
        has_active,
        active_positions.max(dim=1).values + 1,
        torch.as_tensor(text_seq_len, device=encoder_hidden_states.device),
    )
    return text_seq_len, per_sample_len, encoder_hidden_states_mask


class NucleusMoETimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=embedding_dim, time_embed_dim=4 * embedding_dim, out_dim=embedding_dim
        )
        self.norm = RMSNorm(embedding_dim, eps=1e-6)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return self.norm(conditioning)


class NucleusMoEEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),
                self._rope_params(pos_index, self.axes_dim[1], self.theta),
                self._rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(neg_index, self.axes_dim[0], self.theta),
                self._rope_params(neg_index, self.axes_dim[1], self.theta),
                self._rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    @staticmethod
    def _rope_params(index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        device: torch.device = None,
        max_txt_seq_len: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw (`tuple[int, int, int]` or `list[tuple[int, int, int]]`):
                A list of 3 integers [frame, height, width] representing the shape of the video.
            device: (`torch.device`, *optional*):
                The device on which to perform the RoPE computation.
            max_txt_seq_len (`int` or `torch.Tensor`, *optional*):
                The maximum text sequence length for RoPE computation.
        """
        if max_txt_seq_len is None:
            raise ValueError("Either `max_txt_seq_len` must be provided.")

        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            first_fhw = video_fhw[0]
            if not all(fhw == first_fhw for fhw in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in NucleusMoEEmbedRope. "
                    "All images in the batch should have the same dimensions (frame, height, width). "
                    f"Detected sizes: {video_fhw}. Using the first image's dimensions {first_fhw} "
                    "for RoPE computation, which may lead to incorrect results for other images in the batch."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            max_txt_seq_len_int = int(max_txt_seq_len)
            if self.scale_rope:
                max_vid_index = torch.maximum(
                    torch.tensor(height // 2, device=device, dtype=torch.long),
                    torch.tensor(width // 2, device=device, dtype=torch.long),
                )
            else:
                max_vid_index = torch.maximum(
                    torch.tensor(height, device=device, dtype=torch.long),
                    torch.tensor(width, device=device, dtype=torch.long),
                )

        txt_freqs = self.pos_freqs.to(device)[max_vid_index + torch.arange(max_txt_seq_len_int, device=device)]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0, device: torch.device = None
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs = self.pos_freqs.to(device) if device is not None else self.pos_freqs
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class NucleusMoEAttnProcessor2_0:
    """
    Attention processor for the NucleusMoE architecture. Image queries attend to concatenated image+text keys/values
    (cross-attention style, no text query). Supports grouped-query attention (GQA) when num_key_value_heads is set on
    the Attention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "NucleusMoEAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        cached_txt_key: torch.FloatTensor | None = None,
        cached_txt_value: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        head_dim = attn.inner_dim // attn.heads
        num_kv_heads = attn.inner_kv_dim // head_dim
        num_kv_groups = attn.heads // num_kv_heads

        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = attn.to_k(hidden_states).unflatten(-1, (num_kv_heads, -1))
        img_value = attn.to_v(hidden_states).unflatten(-1, (num_kv_heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = _apply_rotary_emb_nucleus(img_query, img_freqs, use_real=False)
            img_key = _apply_rotary_emb_nucleus(img_key, img_freqs, use_real=False)

        if cached_txt_key is not None and cached_txt_value is not None:
            txt_key, txt_value = cached_txt_key, cached_txt_value
            joint_key = torch.cat([img_key, txt_key], dim=1)
            joint_value = torch.cat([img_value, txt_value], dim=1)
        elif encoder_hidden_states is not None:
            txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (num_kv_heads, -1))
            txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (num_kv_heads, -1))

            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)

            if image_rotary_emb is not None:
                txt_key = _apply_rotary_emb_nucleus(txt_key, txt_freqs, use_real=False)

            joint_key = torch.cat([img_key, txt_key], dim=1)
            joint_value = torch.cat([img_value, txt_value], dim=1)
        else:
            joint_key = img_key
            joint_value = img_value

        if num_kv_groups > 1:
            joint_key = joint_key.repeat_interleave(num_kv_groups, dim=2)
            joint_value = joint_value.repeat_interleave(num_kv_groups, dim=2)

        hidden_states = dispatch_attention_fn(
            img_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(img_query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def _is_moe_layer(strategy: str, layer_idx: int, num_layers: int) -> bool:
    if strategy == "leave_first_three_and_last_block_dense":
        return layer_idx >= 3 and layer_idx < num_layers - 1
    elif strategy == "leave_first_three_blocks_dense":
        return layer_idx >= 3
    elif strategy == "leave_first_block_dense":
        return layer_idx >= 1
    elif strategy == "all_moe":
        return True
    elif strategy == "all_dense":
        return False
    return True


class SwiGLUExperts(nn.Module):
    """
    Packed SwiGLU feed-forward experts for MoE: ``gate, up = (x @ gate_up_proj).chunk(2); out = (silu(gate) * up) @
    down_proj``.

    Gate and up projections are fused into a single weight ``gate_up_proj`` so that only two grouped matmuls are needed
    at runtime (gate+up combined, then down).

    Weights are stored pre-transposed relative to the standard linear-layer convention so that matmuls can be issued
    without a transpose at runtime.

    Weight shapes:
        gate_up_proj: (num_experts, hidden_size, 2 * moe_intermediate_dim) -- fused gate + up projection down_proj:
        (num_experts, moe_intermediate_dim, hidden_size) -- down projection
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_dim: int,
        num_experts: int,
        use_grouped_mm: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.moe_intermediate_dim = moe_intermediate_dim
        self.hidden_size = hidden_size
        self.use_grouped_mm = use_grouped_mm

        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * moe_intermediate_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, moe_intermediate_dim, hidden_size))

    def _run_experts_for_loop(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SwiGLU MoE expert outputs using a sequential per-expert for loop.

        Tokens in ``x`` must be pre-sorted so that all tokens assigned to expert 0 come first, followed by expert 1,
        and so on — i.e. the layout produced by a standard token-permutation step (e.g. ``generate_permute_indices``).

        ``x`` may contain trailing padding rows appended by the permutation utility to reach a length that is a
        multiple of some alignment requirement. The padding rows are stripped before expert computation and re-appended
        as zeros so that the output shape matches ``x.shape``, keeping downstream scatter/gather indices valid.

        .. note::
            ``num_tokens_per_expert.tolist()`` synchronises the device with the host. This is acceptable for the loop
            path but means the method introduces a pipeline bubble. Use :meth:`forward` with ``use_grouped_mm=True``
            when a fully device-resident kernel is required (e.g. inside ``torch.compile``).

        SwiGLU formula::

            gate, up = (x @ gate_up_proj).chunk(2) out = (silu(gate) * up) @ down_proj

        Args:
            x (Tensor): Pre-permuted input tokens of shape
                ``(total_tokens_including_padding, hidden_dim)``.
            num_tokens_per_expert (Tensor): 1-D integer tensor of length
                ``num_experts`` giving the number of real (non-padding) tokens assigned to each expert. Values may
                differ across experts to support load-imbalanced routing.

        Returns:
            Tensor of shape ``(total_tokens_including_padding, hidden_dim)``. Positions corresponding to padding rows
            contain zeros.
        """
        # .tolist() triggers a host-device sync; see docstring note above.
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()

        # x may be padded to a larger buffer size by the permutation utility.
        # Track the padding count so we can restore the original buffer shape.
        num_real_tokens = sum(num_tokens_per_expert_list)
        num_padding = x.shape[0] - num_real_tokens

        # Split the real-token prefix of x into per-expert slices (variable length).
        x_per_expert = torch.split(
            x[:num_real_tokens],
            split_size_or_sections=num_tokens_per_expert_list,
            dim=0,
        )

        expert_outputs = []
        for expert_idx, x_expert in enumerate(x_per_expert):
            gate_up = torch.matmul(x_expert, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            out_expert = torch.matmul(F.silu(gate) * up, self.down_proj[expert_idx])
            expert_outputs.append(out_expert)

        # Concatenate real-token outputs, then re-append zero rows for the padding.
        out = torch.cat(expert_outputs, dim=0)
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        return out

    def _run_experts_grouped_mm(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SwiGLU MoE expert outputs using fused grouped GEMM kernels.

        Tokens in ``x`` must be pre-sorted so that all tokens assigned to expert 0 come first, followed by expert 1,
        and so on — the same layout required by :meth:`_run_experts_for_loop`.

        This method is fully device-resident (no host-device sync) and is compatible with ``torch.compile``.

        ``F.grouped_mm`` is called with *exclusive end* offsets: ``offsets[k]`` is the exclusive end index of expert
        ``k``'s token range in ``x`` (equivalently the inclusive start of expert ``k+1``'s range). This is the
        cumulative sum of ``num_tokens_per_expert``.

        SwiGLU formula::

            gate, up = (x @ gate_up_proj).chunk(2) out = (silu(gate) * up) @ down_proj

        Args:
            x (Tensor): Pre-permuted input tokens of shape
                ``(total_tokens, hidden_dim)``. No padding rows expected; ``total_tokens`` must equal
                ``num_tokens_per_expert.sum()``.
            num_tokens_per_expert (Tensor): 1-D integer tensor of length
                ``num_experts`` giving the number of tokens assigned to each expert.

        Returns:
            Tensor of shape ``(total_tokens, hidden_dim)`` with dtype matching ``x``.
        """
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up = F.grouped_mm(x, self.gate_up_proj, offs=offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        out = F.grouped_mm(F.silu(gate) * up, self.down_proj, offs=offsets)

        return out.type_as(x)

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_mm:
            return self._run_experts_grouped_mm(x, num_tokens_per_expert)
        return self._run_experts_for_loop(x, num_tokens_per_expert)


class NucleusMoELayer(nn.Module):
    """
    Mixture-of-Experts layer with expert-choice routing and a shared expert.

    Routed expert weights live in :class:`SwiGLUExperts`. The router concatenates a timestep embedding with the
    (unmodulated) hidden state to produce per-token affinity scores, then selects the top-C tokens per expert
    (expert-choice routing). A shared expert processes all tokens in parallel and its output is combined with the
    routed expert outputs via scatter-add.

    SwiGLU expert computation is implemented by :class:`SwiGLUExperts`.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_dim: int,
        num_experts: int,
        capacity_factor: float,
        use_sigmoid: bool,
        route_scale: float,
        use_grouped_mm: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.moe_intermediate_dim = moe_intermediate_dim
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        self.use_sigmoid = use_sigmoid
        self.route_scale = route_scale

        self.gate = nn.Linear(hidden_size * 2, num_experts, bias=False)

        self.experts = SwiGLUExperts(
            hidden_size=hidden_size,
            moe_intermediate_dim=moe_intermediate_dim,
            num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
        )

        self.shared_expert = FeedForward(
            dim=hidden_size,
            dim_out=hidden_size,
            inner_dim=moe_intermediate_dim,
            activation_fn="swiglu",
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_unmodulated: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, slen, dim = hidden_states.shape

        if timestep is not None:
            timestep_expanded = timestep.unsqueeze(1).expand(-1, slen, -1)
            router_input = torch.cat([timestep_expanded, hidden_states_unmodulated], dim=-1)
        else:
            router_input = hidden_states_unmodulated

        logits = self.gate(router_input)

        if self.use_sigmoid:
            scores = torch.sigmoid(logits.float()).to(logits.dtype)
        else:
            scores = F.softmax(logits.float(), dim=-1).to(logits.dtype)

        affinity = scores.transpose(1, 2)  # (B, E, S)
        capacity = max(1, math.ceil(self.capacity_factor * slen / self.num_experts))

        topk = torch.topk(affinity, k=capacity, dim=-1)
        top_indices = topk.indices  # (B, E, C)
        gating = affinity.gather(dim=-1, index=top_indices)  # (B, E, C)

        batch_offsets = torch.arange(bs, device=hidden_states.device, dtype=torch.long).view(bs, 1, 1) * slen
        global_token_indices = (batch_offsets + top_indices).transpose(0, 1).reshape(self.num_experts, -1).reshape(-1)
        gating_flat = gating.transpose(0, 1).reshape(self.num_experts, -1).reshape(-1)

        token_score_sums = torch.zeros(bs * slen, device=hidden_states.device, dtype=gating_flat.dtype)
        token_score_sums.scatter_add_(0, global_token_indices, gating_flat)
        gating_flat = gating_flat / (token_score_sums[global_token_indices] + 1e-12)
        gating_flat = gating_flat * self.route_scale

        x_flat = hidden_states.reshape(bs * slen, dim)
        routed_input = x_flat[global_token_indices]

        tokens_per_expert = bs * capacity
        num_tokens_per_expert = torch.full(
            (self.num_experts,),
            tokens_per_expert,
            device=hidden_states.device,
            dtype=torch.long,
        )
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        routed_output = (routed_output.float() * gating_flat.unsqueeze(-1)).to(hidden_states.dtype)

        out = self.shared_expert(hidden_states).reshape(bs * slen, dim)

        scatter_idx = global_token_indices.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=scatter_idx, src=routed_output)
        out = out.reshape(bs, slen, dim)

        return out


class NucleusMoEImageTransformerBlock(nn.Module):
    """
    Single-stream DiT block with optional Mixture-of-Experts MLP. Only the image stream receives adaptive modulation;
    the text context is projected per-block and used as cross-attention keys/values.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_key_value_heads: int | None = None,
        joint_attention_dim: int = 3584,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        mlp_ratio: float = 4.0,
        moe_enabled: bool = False,
        num_experts: int = 128,
        moe_intermediate_dim: int = 1344,
        capacity_factor: float = 8.0,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        use_grouped_mm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.moe_enabled = moe_enabled

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 4 * dim, bias=True),
        )

        self.encoder_proj = nn.Linear(joint_attention_dim, dim)

        self.pre_attn_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
        self.attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_key_value_heads,
            dim_head=attention_head_dim,
            added_kv_proj_dim=dim,
            added_proj_bias=False,
            out_dim=dim,
            out_bias=False,
            bias=False,
            processor=NucleusMoEAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
            context_pre_only=None,
        )

        self.pre_mlp_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)

        if moe_enabled:
            self.img_mlp = NucleusMoELayer(
                hidden_size=dim,
                moe_intermediate_dim=moe_intermediate_dim,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                use_sigmoid=use_sigmoid,
                route_scale=route_scale,
                use_grouped_mm=use_grouped_mm,
            )
        else:
            mlp_inner_dim = int(dim * mlp_ratio * 2 / 3) // 128 * 128
            self.img_mlp = FeedForward(
                dim=dim,
                dim_out=dim,
                inner_dim=mlp_inner_dim,
                activation_fn="swiglu",
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        scale1, gate1, scale2, gate2 = self.img_mod(temb).unsqueeze(1).chunk(4, dim=-1)

        gate1 = gate1.clamp(min=-2.0, max=2.0)
        gate2 = gate2.clamp(min=-2.0, max=2.0)

        attn_kwargs = attention_kwargs or {}
        context = None if attn_kwargs.get("cached_txt_key") is not None else self.encoder_proj(encoder_hidden_states)

        img_normed = self.pre_attn_norm(hidden_states)
        img_modulated = img_normed * (1 + scale1)

        img_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=context,
            image_rotary_emb=image_rotary_emb,
            **attn_kwargs,
        )

        hidden_states = hidden_states + gate1.tanh() * img_attn_output

        img_normed2 = self.pre_mlp_norm(hidden_states)
        img_modulated2 = img_normed2 * (1 + scale2)

        if self.moe_enabled:
            img_mlp_output = self.img_mlp(img_modulated2, img_normed2, timestep=temb)
        else:
            img_mlp_output = self.img_mlp(img_modulated2)

        hidden_states = hidden_states + gate2.tanh() * img_mlp_output

        if hidden_states.dtype == torch.float16:
            fp16_finfo = torch.finfo(torch.float16)
            hidden_states = hidden_states.clip(fp16_finfo.min, fp16_finfo.max)

        return hidden_states


class NucleusMoEImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """
    Nucleus MoE Transformer for image generation. Single-stream DiT with cross-attention to text and optional
    Mixture-of-Experts feed-forward layers.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `24`):
            The number of transformer blocks.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `16`):
            The number of attention heads to use.
        num_key_value_heads (`int`, *optional*):
            The number of key/value heads for grouped-query attention. Defaults to `num_attention_heads`.
        joint_attention_dim (`int`, defaults to `3584`):
            The embedding dimension of the encoder hidden states (text).
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
        mlp_ratio (`float`, defaults to `4.0`):
            Multiplier for the MLP hidden dimension in dense (non-MoE) blocks.
        moe_enabled (`bool`, defaults to `True`):
            Whether to use Mixture-of-Experts layers.
        dense_moe_strategy (`str`, defaults to ``"leave_first_three_and_last_block_dense"``):
            Strategy for choosing which layers are MoE vs dense.
        num_experts (`int`, defaults to `128`):
            Number of experts per MoE layer.
        moe_intermediate_dim (`int`, defaults to `1344`):
            Hidden dimension inside each expert.
        capacity_factors (`float | list[float]`, defaults to `8.0`):
            Expert-choice capacity factor per layer.
        use_sigmoid (`bool`, defaults to `False`):
            Use sigmoid instead of softmax for routing scores.
        route_scale (`float`, defaults to `2.5`):
            Scaling factor applied to routing weights.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["NucleusMoEImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["NucleusMoEImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 24,
        attention_head_dim: int = 128,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        joint_attention_dim: int = 3584,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        mlp_ratio: float = 4.0,
        moe_enabled: bool = True,
        dense_moe_strategy: str = "leave_first_three_and_last_block_dense",
        num_experts: int = 128,
        moe_intermediate_dim: int = 1344,
        capacity_factors: float | list[float] = 8.0,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        use_grouped_mm: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        capacity_factors = capacity_factors if isinstance(capacity_factors, list) else [capacity_factors] * num_layers

        self.pos_embed = NucleusMoEEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = NucleusMoETimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                NucleusMoEImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_key_value_heads=num_key_value_heads,
                    joint_attention_dim=joint_attention_dim,
                    mlp_ratio=mlp_ratio,
                    moe_enabled=moe_enabled and _is_moe_layer(dense_moe_strategy, idx, num_layers),
                    num_experts=num_experts,
                    moe_intermediate_dim=moe_intermediate_dim,
                    capacity_factor=capacity_factors[idx],
                    use_sigmoid=use_sigmoid,
                    route_scale=route_scale,
                    use_grouped_mm=use_grouped_mm,
                )
                for idx in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        img_shapes: tuple[int, int, int] | list[tuple[int, int, int]],
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`NucleusMoEImageTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            img_shapes (`list[tuple[int, int, int]]`, *optional*):
                Image shapes ``(frame, height, width)`` for RoPE computation.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Boolean mask for the encoder hidden states.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                Extra kwargs forwarded to the attention processor.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.transformer_2d.Transformer2DModelOutput`].

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)

        text_seq_len, _, encoder_hidden_states_mask = _compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([image_mask, encoder_hidden_states_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=block_attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
