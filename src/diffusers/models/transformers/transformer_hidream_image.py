from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.modeling_outputs import Transformer2DModelOutput
from ...models.modeling_utils import ModelMixin
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention
from ..embeddings import TimestepEmbedding, Timesteps


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HiDreamImageFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class HiDreamImagePooledEmbed(nn.Module):
    def __init__(self, text_emb_dim, hidden_size):
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(in_channels=text_emb_dim, time_embed_dim=hidden_size)

    def forward(self, pooled_embed: torch.Tensor) -> torch.Tensor:
        return self.pooled_embedder(pooled_embed)


class HiDreamImageTimestepEmbed(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size)

    def forward(self, timesteps: torch.Tensor, wdtype: Optional[torch.dtype] = None) -> torch.Tensor:
        t_emb = self.time_proj(timesteps).to(dtype=wdtype)
        t_emb = self.timestep_embedder(t_emb)
        return t_emb


class HiDreamImageOutEmbed(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class HiDreamImagePatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        out_channels=1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels * patch_size * patch_size, out_channels, bias=True)

    def forward(self, latent) -> torch.Tensor:
        latent = self.proj(latent)
        return latent


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    is_mps = pos.device.type == "mps"
    is_npu = pos.device.type == "npu"

    dtype = torch.float32 if (is_mps or is_npu) else torch.float64

    scale = torch.arange(0, dim, 2, dtype=dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class HiDreamImageEmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


@maybe_allow_in_graph
class HiDreamAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        single: bool = False,
    ):
        super(Attention, self).__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        self.to_q = nn.Linear(query_dim, self.inner_dim)
        self.to_k = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_v = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, self.out_dim)
        self.q_rms_norm = nn.RMSNorm(self.inner_dim, eps)
        self.k_rms_norm = nn.RMSNorm(self.inner_dim, eps)

        if not single:
            self.to_q_t = nn.Linear(query_dim, self.inner_dim)
            self.to_k_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_v_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_out_t = nn.Linear(self.inner_dim, self.out_dim)
            self.q_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)
            self.k_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)

        self.set_processor(processor)

    def forward(
        self,
        norm_hidden_states: torch.Tensor,
        hidden_states_masks: torch.Tensor = None,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states=norm_hidden_states,
            hidden_states_masks=hidden_states_masks,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )


class HiDreamAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: HiDreamAttention,
        hidden_states: torch.Tensor,
        hidden_states_masks: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(hidden_states)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(hidden_states)).to(dtype=dtype)
        value_i = attn.to_v(hidden_states)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if hidden_states_masks is not None:
            key_i = key_i * hidden_states_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(encoder_hidden_states)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(encoder_hidden_states)).to(dtype=dtype)
            value_t = attn.to_v_t(encoder_hidden_states)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        if query.shape[-1] == image_rotary_emb.shape[-3] * 2:
            query, key = apply_rope(query, key, image_rotary_emb)

        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, image_rotary_emb)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_routed_experts=4,
        num_activated_experts=2,
        aux_loss_alpha=0.01,
        _force_inference_output=False,
    ):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.randn(self.n_routed_experts, self.gating_dim) / embed_dim**0.5)

        self._force_inference_output = _force_inference_output

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0 and not self._force_inference_output:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)

                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.shared_experts = HiDreamImageFeedForwardSwiGLU(dim, hidden_dim // 2)
        self.experts = nn.ModuleList(
            [HiDreamImageFeedForwardSwiGLU(dim, hidden_dim) for i in range(num_routed_experts)]
        )
        self._force_inference_output = _force_inference_output
        self.gate = MoEGate(
            embed_dim=dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            _force_inference_output=_force_inference_output,
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training and not self._force_inference_output:
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape).to(dtype=wtype)
            # y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum")
        return expert_cache


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states


@maybe_allow_in_graph
class HiDreamImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor(),
            single=True,
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_masks: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        wtype = hidden_states.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = self.adaLN_modulation(temb)[
            :, None
        ].chunk(6, dim=-1)

        # 1. MM-Attention
        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
        attn_output_i = self.attn1(
            norm_hidden_states,
            hidden_states_masks,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = gate_msa_i * attn_output_i + hidden_states

        # 2. Feed-forward
        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states.to(dtype=wtype))
        hidden_states = ff_output_i + hidden_states
        return hidden_states


@maybe_allow_in_graph
class HiDreamImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim, bias=True))

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.norm1_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor(),
            single=False,
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)
        self.norm3_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_t = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_masks: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wtype = hidden_states.dtype
        (
            shift_msa_i,
            scale_msa_i,
            gate_msa_i,
            shift_mlp_i,
            scale_mlp_i,
            gate_mlp_i,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp_t,
            scale_mlp_t,
            gate_mlp_t,
        ) = self.adaLN_modulation(temb)[:, None].chunk(12, dim=-1)

        # 1. MM-Attention
        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
        norm_encoder_hidden_states = self.norm1_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_msa_t) + shift_msa_t

        attn_output_i, attn_output_t = self.attn1(
            norm_hidden_states,
            hidden_states_masks,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = gate_msa_i * attn_output_i + hidden_states
        encoder_hidden_states = gate_msa_t * attn_output_t + encoder_hidden_states

        # 2. Feed-forward
        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
        norm_encoder_hidden_states = self.norm3_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_mlp_t) + shift_mlp_t

        ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states)
        ff_output_t = gate_mlp_t * self.ff_t(norm_encoder_hidden_states)
        hidden_states = ff_output_i + hidden_states
        encoder_hidden_states = ff_output_t + encoder_hidden_states
        return hidden_states, encoder_hidden_states


class HiDreamBlock(nn.Module):
    def __init__(self, block: Union[HiDreamImageTransformerBlock, HiDreamImageSingleTransformerBlock]):
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_masks: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.block(
            hidden_states=hidden_states,
            hidden_states_masks=hidden_states_masks,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )


class HiDreamImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["HiDreamImageTransformerBlock", "HiDreamImageSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None,
        force_inference_output: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.t_embedder = HiDreamImageTimestepEmbed(self.inner_dim)
        self.p_embedder = HiDreamImagePooledEmbed(text_emb_dim, self.inner_dim)
        self.x_embedder = HiDreamImagePatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=self.inner_dim,
        )
        self.pe_embedder = HiDreamImageEmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamBlock(
                    HiDreamImageTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        num_routed_experts=num_routed_experts,
                        num_activated_experts=num_activated_experts,
                        _force_inference_output=force_inference_output,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamBlock(
                    HiDreamImageSingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        num_routed_experts=num_routed_experts,
                        num_activated_experts=num_activated_experts,
                        _force_inference_output=force_inference_output,
                    )
                )
                for _ in range(num_single_layers)
            ]
        )

        self.final_layer = HiDreamImageOutEmbed(self.inner_dim, patch_size, self.out_channels)

        caption_channels = [caption_channels[1]] * (num_layers + num_single_layers) + [caption_channels[0]]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

        self.gradient_checkpointing = False

    def unpatchify(self, x: torch.Tensor, img_sizes: List[Tuple[int, int]], is_training: bool) -> List[torch.Tensor]:
        if is_training and not self.config.force_inference_output:
            B, S, F = x.shape
            C = F // (self.config.patch_size * self.config.patch_size)
            x = (
                x.reshape(B, S, self.config.patch_size, self.config.patch_size, C)
                .permute(0, 4, 1, 2, 3)
                .reshape(B, C, S, self.config.patch_size * self.config.patch_size)
            )
        else:
            x_arr = []
            p1 = self.config.patch_size
            p2 = self.config.patch_size
            for i, img_size in enumerate(img_sizes):
                pH, pW = img_size
                t = x[i, : pH * pW].reshape(1, pH, pW, -1)
                F_token = t.shape[-1]
                C = F_token // (p1 * p2)
                t = t.reshape(1, pH, pW, p1, p2, C)
                t = t.permute(0, 5, 1, 3, 2, 4)
                t = t.reshape(1, C, pH * p1, pW * p2)
                x_arr.append(t)
            x = torch.cat(x_arr, dim=0)
        return x

    def patchify(self, hidden_states):
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.config.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size
        device = hidden_states.device
        dtype = hidden_states.dtype

        # create img_sizes
        img_sizes = torch.tensor([patch_height, patch_width], dtype=torch.int64, device=device).reshape(-1)
        img_sizes = img_sizes.unsqueeze(0).repeat(batch_size, 1)

        # create hidden_states_masks
        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            hidden_states_masks = torch.zeros((batch_size, self.max_seq), dtype=dtype, device=device)
            hidden_states_masks[:, : patch_height * patch_width] = 1.0
        else:
            hidden_states_masks = None

        # create img_ids
        img_ids = torch.zeros(patch_height, patch_width, 3, device=device)
        row_indices = torch.arange(patch_height, device=device)[:, None]
        col_indices = torch.arange(patch_width, device=device)[None, :]
        img_ids[..., 1] = img_ids[..., 1] + row_indices
        img_ids[..., 2] = img_ids[..., 2] + col_indices
        img_ids = img_ids.reshape(patch_height * patch_width, -1)

        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            # Handle non-square latents
            img_ids_pad = torch.zeros(self.max_seq, 3, device=device)
            img_ids_pad[: patch_height * patch_width, :] = img_ids
            img_ids = img_ids_pad.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            img_ids = img_ids.unsqueeze(0).repeat(batch_size, 1, 1)

        # patchify hidden_states
        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            # Handle non-square latents
            out = torch.zeros(
                (batch_size, channels, self.max_seq, patch_size * patch_size),
                dtype=dtype,
                device=device,
            )
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, patch_size, patch_width, patch_size
            )
            hidden_states = hidden_states.permute(0, 1, 2, 4, 3, 5)
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height * patch_width, patch_size * patch_size
            )
            out[:, :, 0 : patch_height * patch_width] = hidden_states
            hidden_states = out
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch_size, self.max_seq, patch_size * patch_size * channels
            )

        else:
            # Handle square latents
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, patch_size, patch_width, patch_size
            )
            hidden_states = hidden_states.permute(0, 2, 4, 3, 5, 1)
            hidden_states = hidden_states.reshape(
                batch_size, patch_height * patch_width, patch_size * patch_size * channels
            )

        return hidden_states, hidden_states_masks, img_sizes, img_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.LongTensor = None,
        encoder_hidden_states_t5: torch.Tensor = None,
        encoder_hidden_states_llama3: torch.Tensor = None,
        pooled_embeds: torch.Tensor = None,
        img_ids: Optional[torch.Tensor] = None,
        img_sizes: Optional[List[Tuple[int, int]]] = None,
        hidden_states_masks: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

        if encoder_hidden_states is not None:
            deprecation_message = "The `encoder_hidden_states` argument is deprecated. Please use `encoder_hidden_states_t5` and `encoder_hidden_states_llama3` instead."
            deprecate("encoder_hidden_states", "0.35.0", deprecation_message)
            encoder_hidden_states_t5 = encoder_hidden_states[0]
            encoder_hidden_states_llama3 = encoder_hidden_states[1]

        if img_ids is not None and img_sizes is not None and hidden_states_masks is None:
            deprecation_message = (
                "Passing `img_ids` and `img_sizes` with unpachified `hidden_states` is deprecated and will be ignored."
            )
            deprecate("img_ids", "0.35.0", deprecation_message)

        if hidden_states_masks is not None and (img_ids is None or img_sizes is None):
            raise ValueError("if `hidden_states_masks` is passed, `img_ids` and `img_sizes` must also be passed.")
        elif hidden_states_masks is not None and hidden_states.ndim != 3:
            raise ValueError(
                "if `hidden_states_masks` is passed, `hidden_states` must be a 3D tensors with shape (batch_size, patch_height * patch_width, patch_size * patch_size * channels)"
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

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # Patchify the input
        if hidden_states_masks is None:
            hidden_states, hidden_states_masks, img_sizes, img_ids = self.patchify(hidden_states)

        # Embed the hidden states
        hidden_states = self.x_embedder(hidden_states)

        # 0. time
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        temb = timesteps + p_embedder

        encoder_hidden_states = [encoder_hidden_states_llama3[k] for k in self.config.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            encoder_hidden_states_t5 = self.caption_projection[-1](encoder_hidden_states_t5)
            encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(encoder_hidden_states_t5)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1]
            + encoder_hidden_states[-2].shape[1]
            + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device,
            dtype=img_ids.dtype,
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids)

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat(
                [initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, initial_encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    hidden_states_masks,
                    cur_encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                hidden_states, initial_encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    hidden_states_masks=hidden_states_masks,
                    encoder_hidden_states=cur_encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if hidden_states_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=hidden_states_masks.device,
                dtype=hidden_states_masks.dtype,
            )
            hidden_states_masks = torch.cat([hidden_states_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    hidden_states_masks,
                    None,
                    temb,
                    image_rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    hidden_states_masks=hidden_states_masks,
                    encoder_hidden_states=None,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, temb)
        output = self.unpatchify(output, img_sizes, self.training)
        if hidden_states_masks is not None:
            hidden_states_masks = hidden_states_masks[:, :image_tokens_seq_len]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
