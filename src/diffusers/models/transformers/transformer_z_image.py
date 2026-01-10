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

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention_processor import Attention
from ...models.modeling_utils import ModelMixin
from ...models.normalization import RMSNorm
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_outputs import Transformer2DModelOutput


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
X_PAD_DIM = 64


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

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
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)  # todo

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

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
            eps=1e-5,
            bias=False,
            out_bias=False,
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
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            seq_len = x.shape[1]

            if noise_mask is not None:
                # Per-token modulation: different modulation for noisy/clean tokens
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
                scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
                gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
                gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
            else:
                # Global modulation: same modulation for all tokens (avoid double select)
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            # Attention block
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            # Original global modulation
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x


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
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]
    _repeated_blocks = ["ZImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]  # precision sensitive layers

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
        siglip_feat_dim=None,  # Optional: set to enable SigLIP support for Omni
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
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
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
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))

        # Optional SigLIP components (for Omni variant)
        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=norm_eps), nn.Linear(siglip_feat_dim, dim, bias=True)
            )
            self.siglip_refiner = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        2000 + layer_id,
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
            self.siglip_pad_token = nn.Parameter(torch.empty((1, dim)))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

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

    def unpatchify(
        self,
        x: List[torch.Tensor],
        size: List[Tuple],
        patch_size,
        f_patch_size,
        x_pos_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz

        if x_pos_offsets is not None:
            # Omni: extract target image from unified sequence (cond_images + target)
            result = []
            for i in range(bsz):
                unified_x = x[i][x_pos_offsets[i][0] : x_pos_offsets[i][1]]
                cu_len = 0
                x_item = None
                for j in range(len(size[i])):
                    if size[i][j] is None:
                        ori_len = 0
                        pad_len = SEQ_MULTI_OF
                        cu_len += pad_len + ori_len
                    else:
                        F, H, W = size[i][j]
                        ori_len = (F // pF) * (H // pH) * (W // pW)
                        pad_len = (-ori_len) % SEQ_MULTI_OF
                        x_item = (
                            unified_x[cu_len : cu_len + ori_len]
                            .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                            .permute(6, 0, 3, 1, 4, 2, 5)
                            .reshape(self.out_channels, F, H, W)
                        )
                        cu_len += ori_len + pad_len
                result.append(x_item)  # Return only the last (target) image
            return result
        else:
            # Original mode: simple unpatchify
            for i in range(bsz):
                F, H, W = size[i]
                ori_len = (F // pF) * (H // pH) * (W // pW)
                # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
                x[i] = (
                    x[i][:ori_len]
                    .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, F, H, W)
                )
            return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify_image(self, image: torch.Tensor, patch_size: int, f_patch_size: int):
        """Patchify a single image tensor: (C, F, H, W) -> (num_patches, patch_dim)."""
        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F, H, W = image.size()
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return image, (F, H, W), (F_tokens, H_tokens, W_tokens)

    def _pad_with_ids(
        self,
        feat: torch.Tensor,
        pos_grid_size: Tuple,
        pos_start: Tuple,
        device: torch.device,
        noise_mask_val: Optional[int] = None,
    ):
        """Pad feature to SEQ_MULTI_OF, create position IDs and pad mask."""
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF
        total_len = ori_len + pad_len

        # Pos IDs
        ori_pos_ids = self.create_coordinate_grid(size=pos_grid_size, start=pos_start, device=device).flatten(0, 2)
        if pad_len > 0:
            pad_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                .flatten(0, 2)
                .repeat(pad_len, 1)
            )
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat(
                [
                    torch.zeros(ori_len, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device),
                ]
            )
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        noise_mask = [noise_mask_val] * total_len if noise_mask_val is not None else None  # token level
        return padded_feat, pos_ids, pad_mask, total_len, noise_mask

    def patchify_and_embed(
        self, all_image: List[torch.Tensor], all_cap_feats: List[torch.Tensor], patch_size: int, f_patch_size: int
    ):
        """Patchify for basic mode: single image per batch item."""
        device = all_image[0].device
        all_img_out, all_img_size, all_img_pos_ids, all_img_pad_mask = [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats):
            # Caption
            cap_out, cap_pos_ids, cap_pad_mask, cap_len, _ = self._pad_with_ids(
                cap_feat, (len(cap_feat) + (-len(cap_feat)) % SEQ_MULTI_OF, 1, 1), (1, 0, 0), device
            )
            all_cap_out.append(cap_out)
            all_cap_pos_ids.append(cap_pos_ids)
            all_cap_pad_mask.append(cap_pad_mask)

            # Image
            img_patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
            img_out, img_pos_ids, img_pad_mask, _, _ = self._pad_with_ids(
                img_patches, (F_t, H_t, W_t), (cap_len + 1, 0, 0), device
            )
            all_img_out.append(img_out)
            all_img_size.append(size)
            all_img_pos_ids.append(img_pos_ids)
            all_img_pad_mask.append(img_pad_mask)

        return (
            all_img_out,
            all_cap_out,
            all_img_size,
            all_img_pos_ids,
            all_cap_pos_ids,
            all_img_pad_mask,
            all_cap_pad_mask,
        )

    def patchify_and_embed_omni(
        self,
        all_x: List[List[torch.Tensor]],
        all_cap_feats: List[List[torch.Tensor]],
        all_siglip_feats: List[List[torch.Tensor]],
        patch_size: int,
        f_patch_size: int,
        images_noise_mask: List[List[int]],
    ):
        """Patchify for omni mode: multiple images per batch item with noise masks."""
        bsz = len(all_x)
        device = all_x[0][-1].device
        dtype = all_x[0][-1].dtype

        all_x_out, all_x_size, all_x_pos_ids, all_x_pad_mask, all_x_len, all_x_noise_mask = [], [], [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask, all_cap_len, all_cap_noise_mask = [], [], [], [], []
        all_sig_out, all_sig_pos_ids, all_sig_pad_mask, all_sig_len, all_sig_noise_mask = [], [], [], [], []

        for i in range(bsz):
            num_images = len(all_x[i])
            cap_feats_list, cap_pos_list, cap_mask_list, cap_lens, cap_noise = [], [], [], [], []
            cap_end_pos = []
            cap_cu_len = 1

            # Process captions
            for j, cap_item in enumerate(all_cap_feats[i]):
                noise_val = images_noise_mask[i][j] if j < len(images_noise_mask[i]) else 1
                cap_out, cap_pos, cap_mask, cap_len, cap_nm = self._pad_with_ids(
                    cap_item,
                    (len(cap_item) + (-len(cap_item)) % SEQ_MULTI_OF, 1, 1),
                    (cap_cu_len, 0, 0),
                    device,
                    noise_val,
                )
                cap_feats_list.append(cap_out)
                cap_pos_list.append(cap_pos)
                cap_mask_list.append(cap_mask)
                cap_lens.append(cap_len)
                cap_noise.extend(cap_nm)
                cap_cu_len += len(cap_item)
                cap_end_pos.append(cap_cu_len)
                cap_cu_len += 2  # for image vae and siglip tokens

            all_cap_out.append(torch.cat(cap_feats_list, dim=0))
            all_cap_pos_ids.append(torch.cat(cap_pos_list, dim=0))
            all_cap_pad_mask.append(torch.cat(cap_mask_list, dim=0))
            all_cap_len.append(cap_lens)
            all_cap_noise_mask.append(cap_noise)

            # Process images
            x_feats_list, x_pos_list, x_mask_list, x_lens, x_size, x_noise = [], [], [], [], [], []
            for j, x_item in enumerate(all_x[i]):
                noise_val = images_noise_mask[i][j]
                if x_item is not None:
                    x_patches, size, (F_t, H_t, W_t) = self._patchify_image(x_item, patch_size, f_patch_size)
                    x_out, x_pos, x_mask, x_len, x_nm = self._pad_with_ids(
                        x_patches, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), device, noise_val
                    )
                    x_size.append(size)
                else:
                    x_len = SEQ_MULTI_OF
                    x_out = torch.zeros((x_len, X_PAD_DIM), dtype=dtype, device=device)
                    x_pos = self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(x_len, 1)
                    x_mask = torch.ones(x_len, dtype=torch.bool, device=device)
                    x_nm = [noise_val] * x_len
                    x_size.append(None)
                x_feats_list.append(x_out)
                x_pos_list.append(x_pos)
                x_mask_list.append(x_mask)
                x_lens.append(x_len)
                x_noise.extend(x_nm)

            all_x_out.append(torch.cat(x_feats_list, dim=0))
            all_x_pos_ids.append(torch.cat(x_pos_list, dim=0))
            all_x_pad_mask.append(torch.cat(x_mask_list, dim=0))
            all_x_size.append(x_size)
            all_x_len.append(x_lens)
            all_x_noise_mask.append(x_noise)

            # Process siglip
            if all_siglip_feats[i] is None:
                all_sig_len.append([0] * num_images)
                all_sig_out.append(None)
            else:
                sig_feats_list, sig_pos_list, sig_mask_list, sig_lens, sig_noise = [], [], [], [], []
                for j, sig_item in enumerate(all_siglip_feats[i]):
                    noise_val = images_noise_mask[i][j]
                    if sig_item is not None:
                        sig_H, sig_W, sig_C = sig_item.size()
                        sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                        sig_out, sig_pos, sig_mask, sig_len, sig_nm = self._pad_with_ids(
                            sig_flat, (1, sig_H, sig_W), (cap_end_pos[j] + 1, 0, 0), device, noise_val
                        )
                        # Scale position IDs to match x resolution
                        if x_size[j] is not None:
                            sig_pos = sig_pos.float()
                            sig_pos[..., 1] = sig_pos[..., 1] / max(sig_H - 1, 1) * (x_size[j][1] - 1)
                            sig_pos[..., 2] = sig_pos[..., 2] / max(sig_W - 1, 1) * (x_size[j][2] - 1)
                            sig_pos = sig_pos.to(torch.int32)
                    else:
                        sig_len = SEQ_MULTI_OF
                        sig_out = torch.zeros((sig_len, self.config.siglip_feat_dim), dtype=dtype, device=device)
                        sig_pos = (
                            self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(sig_len, 1)
                        )
                        sig_mask = torch.ones(sig_len, dtype=torch.bool, device=device)
                        sig_nm = [noise_val] * sig_len
                    sig_feats_list.append(sig_out)
                    sig_pos_list.append(sig_pos)
                    sig_mask_list.append(sig_mask)
                    sig_lens.append(sig_len)
                    sig_noise.extend(sig_nm)

                all_sig_out.append(torch.cat(sig_feats_list, dim=0))
                all_sig_pos_ids.append(torch.cat(sig_pos_list, dim=0))
                all_sig_pad_mask.append(torch.cat(sig_mask_list, dim=0))
                all_sig_len.append(sig_lens)
                all_sig_noise_mask.append(sig_noise)

        # Compute x position offsets
        all_x_pos_offsets = [(sum(all_cap_len[i]), sum(all_cap_len[i]) + sum(all_x_len[i])) for i in range(bsz)]

        return (
            all_x_out,
            all_cap_out,
            all_sig_out,
            all_x_size,
            all_x_pos_ids,
            all_cap_pos_ids,
            all_sig_pos_ids,
            all_x_pad_mask,
            all_cap_pad_mask,
            all_sig_pad_mask,
            all_x_pos_offsets,
            all_x_noise_mask,
            all_cap_noise_mask,
            all_sig_noise_mask,
        )

    def _prepare_sequence(
        self,
        feats: List[torch.Tensor],
        pos_ids: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: torch.nn.Parameter,
        noise_mask: Optional[List[List[int]]] = None,
        device: torch.device = None,
    ):
        """Prepare sequence: apply pad token, RoPE embed, pad to batch, create attention mask."""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        # Pad token
        feats_cat = torch.cat(feats, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token
        feats = list(feats_cat.split(item_seqlens, dim=0))

        # RoPE
        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))

        # Pad to batch
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        # Noise mask
        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, : feats.shape[1]]

        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    def _build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: torch.Tensor,
        x_seqlens: List[int],
        x_noise_mask: Optional[List[List[int]]],
        cap: torch.Tensor,
        cap_freqs: torch.Tensor,
        cap_seqlens: List[int],
        cap_noise_mask: Optional[List[List[int]]],
        siglip: Optional[torch.Tensor],
        siglip_freqs: Optional[torch.Tensor],
        siglip_seqlens: Optional[List[int]],
        siglip_noise_mask: Optional[List[List[int]]],
        omni_mode: bool,
        device: torch.device,
    ):
        """Build unified sequence: x, cap, and optionally siglip.
        Basic mode order: [x, cap]; Omni mode order: [cap, x, siglip]
        """
        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []
        unified_noise_mask = []

        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]

            if omni_mode:
                # Omni: [cap, x, siglip]
                if siglip is not None and siglip_seqlens is not None:
                    sig_len = siglip_seqlens[i]
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len], siglip[i][:sig_len]]))
                    unified_freqs.append(
                        torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len], siglip_freqs[i][:sig_len]])
                    )
                    unified_noise_mask.append(
                        torch.tensor(
                            cap_noise_mask[i] + x_noise_mask[i] + siglip_noise_mask[i], dtype=torch.long, device=device
                        )
                    )
                else:
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len]]))
                    unified_freqs.append(torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len]]))
                    unified_noise_mask.append(
                        torch.tensor(cap_noise_mask[i] + x_noise_mask[i], dtype=torch.long, device=device)
                    )
            else:
                # Basic: [x, cap]
                unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
                unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))

        # Compute unified seqlens
        if omni_mode:
            if siglip is not None and siglip_seqlens is not None:
                unified_seqlens = [a + b + c for a, b, c in zip(cap_seqlens, x_seqlens, siglip_seqlens)]
            else:
                unified_seqlens = [a + b for a, b in zip(cap_seqlens, x_seqlens)]
        else:
            unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]

        max_seqlen = max(unified_seqlens)

        # Pad to batch
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_seqlens):
            attn_mask[i, :seq_len] = 1

        # Noise mask
        noise_mask_tensor = None
        if omni_mode:
            noise_mask_tensor = pad_sequence(unified_noise_mask, batch_first=True, padding_value=0)[
                :, : unified.shape[1]
            ]

        return unified, unified_freqs, attn_mask, noise_mask_tensor

    def forward(
        self,
        x: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        t,
        cap_feats: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        return_dict: bool = True,
        controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
        siglip_feats: Optional[List[List[torch.Tensor]]] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        """
        Flow: patchify -> t_embed -> x_embed -> x_refine -> cap_embed -> cap_refine
              -> [siglip_embed -> siglip_refine] -> build_unified -> main_layers -> final_layer -> unpatchify
        """
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        omni_mode = isinstance(x[0], list)
        device = x[0][-1].device if omni_mode else x[0].device

        if omni_mode:
            # Dual embeddings: noisy (t) and clean (t=1)
            t_noisy = self.t_embedder(t * self.t_scale).type_as(x[0][-1])
            t_clean = self.t_embedder(torch.ones_like(t) * self.t_scale).type_as(x[0][-1])
            adaln_input = None
        else:
            # Single embedding for all tokens
            adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
            t_noisy = t_clean = None

        # Patchify
        if omni_mode:
            (
                x,
                cap_feats,
                siglip_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                siglip_pos_ids,
                x_pad_mask,
                cap_pad_mask,
                siglip_pad_mask,
                x_pos_offsets,
                x_noise_mask,
                cap_noise_mask,
                siglip_noise_mask,
            ) = self.patchify_and_embed_omni(x, cap_feats, siglip_feats, patch_size, f_patch_size, image_noise_mask)
        else:
            (
                x,
                cap_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
            x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        # X embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))  # embed
        x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, x_noise_mask, device
        )

        for layer in self.noise_refiner:
            x = (
                self._gradient_checkpointing_func(
                    layer, x, x_mask, x_freqs, adaln_input, x_noise_tensor, t_noisy, t_clean
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(x, x_mask, x_freqs, adaln_input, x_noise_tensor, t_noisy, t_clean)
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))  # embed
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        )

        for layer in self.context_refiner:
            cap_feats = (
                self._gradient_checkpointing_func(layer, cap_feats, cap_mask, cap_freqs)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(cap_feats, cap_mask, cap_freqs)
            )

        # Siglip embed & refine
        siglip_seqlens = siglip_freqs = None
        if omni_mode and siglip_feats[0] is not None and self.siglip_embedder is not None:
            siglip_seqlens = [len(si) for si in siglip_feats]
            siglip_feats = self.siglip_embedder(torch.cat(siglip_feats, dim=0))  # embed
            siglip_feats, siglip_freqs, siglip_mask, _, _ = self._prepare_sequence(
                list(siglip_feats.split(siglip_seqlens, dim=0)),
                siglip_pos_ids,
                siglip_pad_mask,
                self.siglip_pad_token,
                None,
                device,
            )

            for layer in self.siglip_refiner:
                siglip_feats = (
                    self._gradient_checkpointing_func(layer, siglip_feats, siglip_mask, siglip_freqs)
                    if torch.is_grad_enabled() and self.gradient_checkpointing
                    else layer(siglip_feats, siglip_mask, siglip_freqs)
                )

        # Unified sequence
        unified, unified_freqs, unified_mask, unified_noise_tensor = self._build_unified_sequence(
            x,
            x_freqs,
            x_seqlens,
            x_noise_mask,
            cap_feats,
            cap_freqs,
            cap_seqlens,
            cap_noise_mask,
            siglip_feats,
            siglip_freqs,
            siglip_seqlens,
            siglip_noise_mask,
            omni_mode,
            device,
        )

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified = (
                self._gradient_checkpointing_func(
                    layer, unified, unified_mask, unified_freqs, adaln_input, unified_noise_tensor, t_noisy, t_clean
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(unified, unified_mask, unified_freqs, adaln_input, unified_noise_tensor, t_noisy, t_clean)
            )
            if controlnet_block_samples is not None and layer_idx in controlnet_block_samples:
                unified = unified + controlnet_block_samples[layer_idx]

        unified = (
            self.all_final_layer[f"{patch_size}-{f_patch_size}"](
                unified, noise_mask=unified_noise_tensor, c_noisy=t_noisy, c_clean=t_clean
            )
            if omni_mode
            else self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        )

        # Unpatchify
        x = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size, x_pos_offsets)

        return (x,) if not return_dict else Transformer2DModelOutput(sample=x)
