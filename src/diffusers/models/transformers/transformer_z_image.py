# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

import itertools
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    from torch.nn import RMSNorm

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention_processor import Attention
from ...models.modeling_utils import ModelMixin
from ...utils.torch_utils import maybe_allow_in_graph


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        x_cu_seqlens: Optional[torch.Tensor] = None,
        x_max_item_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        x_shard = hidden_states
        x_freqs_cis_shard = image_rotary_emb

        query = attn.to_q(x_shard)
        key = attn.to_k(x_shard)
        value = attn.to_v(x_shard)

        seqlen_shard = x_shard.shape[0]

        # Reshape to [seq_len, heads, head_dim]
        head_dim = query.shape[-1] // attn.heads
        query = query.view(seqlen_shard, attn.heads, head_dim)
        key = key.view(seqlen_shard, attn.heads, head_dim)
        value = value.view(seqlen_shard, attn.heads, head_dim)
        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(1)
                x_out = torch.view_as_real(x * freqs_cis).flatten(2)
                return x_out.type_as(x_in)

        if x_freqs_cis_shard is not None:
            query = apply_rotary_emb(query, x_freqs_cis_shard)
            key = apply_rotary_emb(key, x_freqs_cis_shard)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Flash Attention
        softmax_scale = math.sqrt(1 / head_dim)
        assert dtype in [torch.float16, torch.bfloat16]

        if x_cu_seqlens is None or x_max_item_seqlen is None:
            raise ValueError("x_cu_seqlens and x_max_item_seqlen are required for ZSingleStreamAttnProcessor")

        if flash_attn_varlen_func is not None:
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=x_cu_seqlens,
                cu_seqlens_k=x_cu_seqlens,
                max_seqlen_q=x_max_item_seqlen,
                max_seqlen_k=x_max_item_seqlen,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )
            output = output.flatten(-2)
        else:
            seqlens = (x_cu_seqlens[1:] - x_cu_seqlens[:-1]).cpu().tolist()

            q_split = torch.split(query, seqlens, dim=0)
            k_split = torch.split(key, seqlens, dim=0)
            v_split = torch.split(value, seqlens, dim=0)

            q_padded = torch.nn.utils.rnn.pad_sequence(q_split, batch_first=True)
            k_padded = torch.nn.utils.rnn.pad_sequence(k_split, batch_first=True)
            v_padded = torch.nn.utils.rnn.pad_sequence(v_split, batch_first=True)

            batch_size, max_seqlen, _, _ = q_padded.shape

            mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=query.device)
            for i, l in enumerate(seqlens):
                mask[i, :l] = True

            attn_mask = torch.zeros((batch_size, 1, 1, max_seqlen), dtype=query.dtype, device=query.device)
            attn_mask.masked_fill_(~mask[:, None, None, :], torch.finfo(query.dtype).min)

            q_padded = q_padded.transpose(1, 2)
            k_padded = k_padded.transpose(1, 2)
            v_padded = v_padded.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                q_padded,
                k_padded,
                v_padded,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=softmax_scale,
            )

            output = output.transpose(1, 2)

            out_list = []
            for i, l in enumerate(seqlens):
                out_list.append(output[i, :l])

            output = torch.cat(out_list, dim=0)
            output = output.flatten(-2)

        output = attn.to_out[0](output)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.w2.weight)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.w3.weight)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-6,
            bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )
            nn.init.zeros_(self.adaLN_modulation[0].weight)
            nn.init.zeros_(self.adaLN_modulation[0].bias)

    def forward(
        self,
        x_shard: torch.Tensor,
        x_src_ids_shard: torch.Tensor,
        x_freqs_cis_shard: torch.Tensor,
        x_cu_seqlens: torch.Tensor,
        x_max_item_seqlen: int,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            scale_gate_msa = (scale_msa, gate_msa)
            scale_gate_mlp = (scale_mlp, gate_mlp)
        else:
            scale_gate_msa = None
            scale_gate_mlp = None
            x_src_ids_shard = None

        x_shard = self.attn_forward(
            x_shard,
            x_freqs_cis_shard,
            x_cu_seqlens,
            x_max_item_seqlen,
            scale_gate_msa,
            x_src_ids_shard,
        )

        x_shard = self.ffn_forward(x_shard, scale_gate_mlp, x_src_ids_shard)

        return x_shard

    def attn_forward(
        self,
        x_shard,
        x_freqs_cis_shard,
        x_cu_seqlens,
        x_max_item_seqlen,
        scale_gate: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        x_src_ids_shard: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert scale_gate is not None and x_src_ids_shard is not None
            scale_msa, gate_msa = scale_gate

            # Pass extra args needed for ZSingleStreamAttnProcessor
            attn_out = self.attention(
                self.attention_norm1(x_shard) * scale_msa[x_src_ids_shard],
                image_rotary_emb=x_freqs_cis_shard,
                x_cu_seqlens=x_cu_seqlens,
                x_max_item_seqlen=x_max_item_seqlen,
            )

            x_shard = x_shard + gate_msa[x_src_ids_shard] * self.attention_norm2(attn_out)
        else:
            attn_out = self.attention(
                self.attention_norm1(x_shard),
                image_rotary_emb=x_freqs_cis_shard,
                x_cu_seqlens=x_cu_seqlens,
                x_max_item_seqlen=x_max_item_seqlen,
            )
            x_shard = x_shard + self.attention_norm2(attn_out)
        return x_shard

    def ffn_forward(
        self,
        x_shard,
        scale_gate: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        x_src_ids_shard: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert scale_gate is not None and x_src_ids_shard is not None
            scale_mlp, gate_mlp = scale_gate
            x_shard = x_shard + gate_mlp[x_src_ids_shard] * self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x_shard) * scale_mlp[x_src_ids_shard],
                )
            )

        else:
            x_shard = x_shard + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x_shard),
                )
            )
        return x_shard


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x_shard, x_src_ids_shard, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x_shard = self.norm_final(x_shard) * scale[x_src_ids_shard]
        x_shard = self.linear(x_shard)
        return x_shard


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.cuda() for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            nn.init.xavier_uniform_(x_embedder.weight)
            nn.init.constant_(x_embedder.bias, 0.0)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )
        nn.init.trunc_normal_(self.cap_embedder[1].weight, std=0.02)
        nn.init.zeros_(self.cap_embedder[1].bias)

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        nn.init.normal_(self.x_pad_token, std=0.02)
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))
        nn.init.normal_(self.cap_pad_token, std=0.02)

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x[i] = rearrange(
                x[i][:ori_len].view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels),
                "f h w pf ph pw c -> c (f pf) (h ph) (w pw)",
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, image in enumerate(all_image):
            ### LLM Text Encoder
            cap_ori_len = len(all_cap_feats[i])
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [all_cap_feats[i], all_cap_feats[i][-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = rearrange(image, "c f pf h ph w pw -> (f h w) (pf ph pw c)")

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF
            # padded_pos_ids

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        adaln_input = t

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_pad_mask,
            cap_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)
        x_cu_seqlens = F.pad(
            torch.cumsum(
                torch.tensor(x_item_seqlens, dtype=torch.int32, device=device),
                dim=0,
                dtype=torch.int32,
            ),
            (1, 0),
        )
        x_src_ids = [
            torch.full((count,), i, dtype=torch.int32, device=device) for i, count in enumerate(x_item_seqlens)
        ]
        x_freqs_cis = self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0)

        x_shard = torch.cat(x, dim=0)
        x_src_ids_shard = torch.cat(x_src_ids, dim=0)
        x_freqs_cis_shard = torch.cat(x_freqs_cis, dim=0)
        x_pad_mask_shard = torch.cat(x_pad_mask, dim=0)
        del x

        x_shard = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_shard)
        x_shard[x_pad_mask_shard] = self.x_pad_token
        for layer in self.noise_refiner:
            x_shard = layer(
                x_shard,
                x_src_ids_shard,
                x_freqs_cis_shard,
                x_cu_seqlens,
                x_max_item_seqlen,
                adaln_input,
            )
        x_flatten = x_shard

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)
        cap_cu_seqlens = F.pad(
            torch.cumsum(
                torch.tensor(cap_item_seqlens, dtype=torch.int32, device=device),
                dim=0,
                dtype=torch.int32,
            ),
            (1, 0),
        )
        cap_src_ids = [
            torch.full((count,), i, dtype=torch.int32, device=device) for i, count in enumerate(cap_item_seqlens)
        ]
        cap_freqs_cis = self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0)

        cap_shard = torch.cat(cap_feats, dim=0)
        cap_src_ids_shard = torch.cat(cap_src_ids, dim=0)
        cap_freqs_cis_shard = torch.cat(cap_freqs_cis, dim=0)
        cap_pad_mask_shard = torch.cat(cap_pad_mask, dim=0)
        del cap_feats

        cap_shard = self.cap_embedder(cap_shard)
        cap_shard[cap_pad_mask_shard] = self.cap_pad_token
        for layer in self.context_refiner:
            cap_shard = layer(
                cap_shard,
                cap_src_ids_shard,
                cap_freqs_cis_shard,
                cap_cu_seqlens,
                cap_max_item_seqlen,
            )
        cap_flatten = cap_shard

        # unified
        def merge_interleave(l1, l2):
            return list(itertools.chain(*zip(l1, l2)))

        unified = torch.cat(
            merge_interleave(
                cap_flatten.split(cap_item_seqlens, dim=0),
                x_flatten.split(x_item_seqlens, dim=0),
            ),
            dim=0,
        )
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert len(unified) == sum(unified_item_seqlens)
        unified_max_item_seqlen = max(unified_item_seqlens)
        unified_cu_seqlens = F.pad(
            torch.cumsum(
                torch.tensor(unified_item_seqlens, dtype=torch.int32, device=device),
                dim=0,
                dtype=torch.int32,
            ),
            (1, 0),
        )
        unified_src_ids = torch.cat(merge_interleave(cap_src_ids, x_src_ids))
        unified_freqs_cis = torch.cat(merge_interleave(cap_freqs_cis, x_freqs_cis))

        unified_shard = unified
        unified_src_ids_shard = unified_src_ids
        unified_freqs_cis_shard = unified_freqs_cis
        for layer in self.layers:
            unified_shard = layer(
                unified_shard,
                unified_src_ids_shard,
                unified_freqs_cis_shard,
                unified_cu_seqlens,
                unified_max_item_seqlen,
                adaln_input,
            )
        unified_shard = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified_shard, unified_src_ids_shard, adaln_input
        )
        unified = unified_shard.split(unified_item_seqlens, dim=0)
        x = [unified[i][cap_item_seqlens[i] :] for i in range(bsz)]
        assert all(len(x[i]) == x_item_seqlens[i] for i in range(bsz))

        x = self.unpatchify(x, x_size, patch_size, f_patch_size)

        return x, {}

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params
