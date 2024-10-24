import collections
import functools
import itertools
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(itertools.repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def centers(start: float, stop, num, dtype=None, device=None):
    """linspace through bin centers.

    Args:
        start (float): Start of the range.
        stop (float): End of the range.
        num (int): Number of points.
        dtype (torch.dtype): Data type of the points.
        device (torch.device): Device of the points.

    Returns:
        centers (Tensor): Centers of the bins. Shape: (num,).
    """
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


@functools.lru_cache(maxsize=1)
def create_position_matrix(
    T: int,
    pH: int,
    pW: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    target_area: float = 36864,
):
    """
    Args:
        T: int - Temporal dimension
        pH: int - Height dimension after patchify
        pW: int - Width dimension after patchify

    Returns:
        pos: [T * pH * pW, 3] - position matrix
    """
    with torch.no_grad():
        # Create 1D tensors for each dimension
        t = torch.arange(T, dtype=dtype)

        # Positionally interpolate to area 36864.
        # (3072x3072 frame with 16x16 patches = 192x192 latents).
        # This automatically scales rope positions when the resolution changes.
        # We use a large target area so the model is more sensitive
        # to changes in the learned pos_frequencies matrix.
        scale = math.sqrt(target_area / (pW * pH))
        w = centers(-pW * scale / 2, pW * scale / 2, pW)
        h = centers(-pH * scale / 2, pH * scale / 2, pH)

        # Use meshgrid to create 3D grids
        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")

        # Stack and reshape the grids.
        pos = torch.stack([grid_t, grid_h, grid_w], dim=-1)  # [T, pH, pW, 3]
        pos = pos.view(-1, 3)  # [T * pH * pW, 3]
        pos = pos.to(dtype=dtype, device=device)

    return pos


def compute_mixed_rotation(
    freqs: torch.Tensor,
    pos: torch.Tensor,
):
    """
    Project each 3-dim position into per-head, per-head-dim 1D frequencies.

    Args:
        freqs: [3, num_heads, num_freqs] - learned rotation frequency (for t, row, col) for each head position
        pos: [N, 3] - position of each token
        num_heads: int

    Returns:
        freqs_cos: [N, num_heads, num_freqs] - cosine components freqs_sin: [N, num_heads, num_freqs] - sine components
    """
    with torch.autocast("cuda", enabled=False):
        assert freqs.ndim == 3
        freqs_sum = torch.einsum("Nd,dhf->Nhf", pos.to(freqs), freqs)
        freqs_cos = torch.cos(freqs_sum)
        freqs_sin = torch.sin(freqs_sum)
    return freqs_cos, freqs_sin


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        *,
        bias: bool = True,
        timestep_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.timestep_scale = timestep_scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        freqs.mul_(-math.log(max_period) / half).exp_()
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        if self.timestep_scale is not None:
            t = t * self.timestep_scale
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PooledCaptionEmbedder(nn.Module):
    def __init__(
        self,
        caption_feature_dim: int,
        hidden_size: int,
        *,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.caption_feature_dim = caption_feature_dim
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(caption_feature_dim, hidden_size, bias=bias, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias, device=device),
        )

    def forward(self, x):
        return self.mlp(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        # keep parameter count and computation constant compared to standard FFN
        hidden_size = int(2 * hidden_size / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_size = int(ffn_dim_multiplier * hidden_size)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)

        self.hidden_dim = hidden_size
        self.w1 = nn.Linear(in_features, 2 * hidden_size, bias=False, device=device)
        self.w2 = nn.Linear(hidden_size, in_features, bias=False, device=device)

    def forward(self, x):
        x, gate = self.w1(x).chunk(2, dim=-1)
        x = self.w2(F.silu(x) * gate)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        dynamic_img_pad: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            device=device,
        )
        assert norm_layer is None
        self.norm = norm_layer(embed_dim, device=device) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _C, T, H, W = x.shape
        if not self.dynamic_img_pad:
            assert (
                H % self.patch_size[0] == 0
            ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            assert (
                W % self.patch_size[1] == 0
            ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        else:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = rearrange(x, "B C T H W -> (B T) C H W", B=B, T=T)
        x = self.proj(x)

        # Flatten temporal and spatial dimensions.
        if not self.flatten:
            raise NotImplementedError("Must flatten output.")
        x = rearrange(x, "(B T) C H W -> B (T H W) C", B=B, T=T)

        x = self.norm(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device))
        self.register_parameter("bias", None)

    def forward(self, x):
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * self.weight).type_as(x)


def pool_tokens(x: torch.Tensor, mask: torch.Tensor, *, keepdim=False) -> torch.Tensor:
    """
    Pool tokens in x using mask.

    NOTE: We assume x does not require gradients.

    Args:
        x: (B, L, D) tensor of tokens.
        mask: (B, L) boolean tensor indicating which tokens are not padding.

    Returns:
        pooled: (B, D) tensor of pooled tokens.
    """
    assert x.size(1) == mask.size(1)  # Expected mask to have same length as tokens.
    assert x.size(0) == mask.size(0)  # Expected mask to have same batch size as tokens.
    mask = mask[:, :, None].to(dtype=x.dtype)
    mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
    pooled = (x * mask).sum(dim=1, keepdim=keepdim)
    return pooled


class AttentionPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            spatial_dim (int): Number of tokens in sequence length.
            embed_dim (int): Dimensionality of input tokens.
            num_heads (int): Number of attention heads.
            output_dim (int): Dimensionality of output tokens. Defaults to embed_dim.
        """
        super().__init__()
        self.num_heads = num_heads
        self.to_kv = nn.Linear(embed_dim, 2 * embed_dim, device=device)
        self.to_q = nn.Linear(embed_dim, embed_dim, device=device)
        self.to_out = nn.Linear(embed_dim, output_dim or embed_dim, device=device)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): (B, L, D) tensor of input tokens.
            mask (torch.Tensor): (B, L) boolean tensor indicating which tokens are not padding.

        NOTE: We assume x does not require gradients.

        Returns:
            x (torch.Tensor): (B, D) tensor of pooled tokens.
        """
        D = x.size(2)

        # Construct attention mask, shape: (B, 1, num_queries=1, num_keys=1+L).
        attn_mask = mask[:, None, None, :].bool()  # (B, 1, 1, L).
        attn_mask = F.pad(attn_mask, (1, 0), value=True)  # (B, 1, 1, 1+L).

        # Average non-padding token features. These will be used as the query.
        x_pool = pool_tokens(x, mask, keepdim=True)  # (B, 1, D)

        # Concat pooled features to input sequence.
        x = torch.cat([x_pool, x], dim=1)  # (B, L+1, D)

        # Compute queries, keys, values. Only the mean token is used to create a query.
        kv = self.to_kv(x)  # (B, L+1, 2 * D)
        q = self.to_q(x[:, 0])  # (B, D)

        # Extract heads.
        head_dim = D // self.num_heads
        kv = kv.unflatten(2, (2, self.num_heads, head_dim))  # (B, 1+L, 2, H, head_dim)
        kv = kv.transpose(1, 3)  # (B, H, 2, 1+L, head_dim)
        k, v = kv.unbind(2)  # (B, H, 1+L, head_dim)
        q = q.unflatten(1, (self.num_heads, head_dim))  # (B, H, head_dim)
        q = q.unsqueeze(2)  # (B, H, 1, head_dim)

        # Compute attention.
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)  # (B, H, 1, head_dim)

        # Concatenate heads and run output.
        x = x.squeeze(2).flatten(1, 2)  # (B, D = H * head_dim)
        x = self.to_out(x)
        return x


class ResidualTanhGatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_res, gate, eps=1e-6):
        # Convert to fp32 for precision
        x_res_fp32 = x_res.float()

        # Compute RMS
        mean_square = x_res_fp32.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(mean_square + eps)

        # Apply tanh to gate
        tanh_gate = torch.tanh(gate).unsqueeze(1)

        # Normalize and apply gated scaling
        x_normed = x_res_fp32 * scale * tanh_gate

        # Apply residual connection
        output = x + x_normed.type_as(x)

        return output


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    return ResidualTanhGatedRMSNorm.apply(x, x_res, gate, eps)


class ModulatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, eps=1e-6):
        # Convert to fp32 for precision
        x_fp32 = x.float()
        scale_fp32 = scale.float()

        # Compute RMS
        mean_square = x_fp32.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + eps)

        # Normalize and modulate
        x_normed = x_fp32 * inv_rms
        x_modulated = x_normed * (1 + scale_fp32.unsqueeze(1))

        return x_modulated.type_as(x)


def modulated_rmsnorm(x, scale, eps=1e-6):
    return ModulatedRMSNorm.apply(x, scale, eps)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AsymmetricJointBlock(nn.Module):
    def __init__(
        self,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens.
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens.
        update_y: bool = True,  # Whether to update text tokens in this block.
        device: Optional[torch.device] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mod_x = nn.Linear(hidden_size_x, 4 * hidden_size_x, device=device)
        if self.update_y:
            self.mod_y = nn.Linear(hidden_size_x, 4 * hidden_size_y, device=device)
        else:
            self.mod_y = nn.Linear(hidden_size_x, hidden_size_y, device=device)

        # Self-attention:
        self.attn = AsymmetricAttention(
            hidden_size_x,
            hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            device=device,
            **block_kwargs,
        )

        # MLP.
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=256,
            ffn_dim_multiplier=None,
            device=device,
        )

        # MLP for text not needed in last block.
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=256,
                ffn_dim_multiplier=None,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        **attn_kwargs,
    ):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block y: (B, L, dim) tensor of text tokens after block
        """
        N = x.size(1)

        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)

        mod_y = self.mod_y(c)
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y

        # Self-attention block.
        x_attn, y_attn = self.attn(
            x,
            y,
            scale_x=scale_msa_x,
            scale_y=scale_msa_y,
            **attn_kwargs,
        )

        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)
        breakpoint()

        # MLP block.
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)

        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)  # Sandwich norm
        breakpoint()
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        breakpoint()
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)  # Sandwich norm
        return y


class AsymmetricAttention(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        update_y: bool = True,
        out_bias: bool = True,
        softmax_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(f"dim_x={dim_x} should be divisible by num_heads={num_heads}")

        # Input layers.
        self.qkv_bias = qkv_bias
        self.qkv_x = nn.Linear(dim_x, 3 * dim_x, bias=qkv_bias, device=device)
        # Project text features to match visual features (dim_y -> dim_x)
        self.qkv_y = nn.Linear(dim_y, 3 * dim_x, bias=qkv_bias, device=device)

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = RMSNorm(self.head_dim, device=device)
        self.k_norm_x = RMSNorm(self.head_dim, device=device)
        self.q_norm_y = RMSNorm(self.head_dim, device=device)
        self.k_norm_y = RMSNorm(self.head_dim, device=device)

        # Output layers. y features go back down from dim_x -> dim_y.
        self.proj_x = nn.Linear(dim_x, dim_x, bias=out_bias, device=device)
        self.proj_y = nn.Linear(dim_x, dim_y, bias=out_bias, device=device) if update_y else nn.Identity()

    def run_qkv_y(self, y):
        qkv_y = self.qkv_y(y)
        qkv_y = qkv_y.view(qkv_y.size(0), qkv_y.size(1), 3, -1, self.head_dim)
        q_y, k_y, v_y = qkv_y.unbind(2)
        return q_y, k_y, v_y

        # cp_rank, cp_size = cp.get_cp_rank_size()
        # local_heads = self.num_heads // cp_size

        # if cp.is_cp_active():
        #     # Only predict local heads.
        #     assert not self.qkv_bias
        #     W_qkv_y = self.qkv_y.weight.view(
        #         3, self.num_heads, self.head_dim, self.dim_y
        #     )
        #     W_qkv_y = W_qkv_y.narrow(1, cp_rank * local_heads, local_heads)
        #     W_qkv_y = W_qkv_y.reshape(3 * local_heads * self.head_dim, self.dim_y)
        #     qkv_y = F.linear(y, W_qkv_y, None)  # (B, L, 3 * local_h * head_dim)
        # else:
        #     qkv_y = self.qkv_y(y)  # (B, L, 3 * dim)

        # qkv_y = qkv_y.view(qkv_y.size(0), qkv_y.size(1), 3, local_heads, self.head_dim)
        # q_y, k_y, v_y = qkv_y.unbind(2)
        # return q_y, k_y, v_y

    def prepare_qkv(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,
        scale_y: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        valid_token_indices: torch.Tensor = None,
    ):
        # Pre-norm for visual features
        x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size

        # Process visual features
        qkv_x = self.qkv_x(x)  # (B, M, 3 * dim_x)
        # assert qkv_x.dtype == torch.bfloat16
        # qkv_x = cp.all_to_all_collect_tokens(
        #     qkv_x, self.num_heads
        # )  # (3, B, N, local_h, head_dim)
        B, M, _ = qkv_x.size()
        qkv_x = qkv_x.view(B, M, 3, -1, 128)
        qkv_x = qkv_x.permute(2, 0, 1, 3, 4)

        # Process text features
        y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
        q_y, k_y, v_y = self.run_qkv_y(y)  # (B, L, local_heads, head_dim)
        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)

        # Split qkv_x into q, k, v
        q_x, k_x, v_x = qkv_x.unbind(0)  # (B, N, local_h, head_dim)
        q_x = self.q_norm_x(q_x)
        k_x = self.k_norm_x(k_x)
        
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)

        # Unite streams
        qkv = unify_streams(
            q_x,
            k_x,
            v_x,
            q_y,
            k_y,
            v_y,
            valid_token_indices,
        )

        return qkv

    @torch.compiler.disable()
    def run_attention(
        self,
        qkv: torch.Tensor,  # (total <= B * (N + L), 3, local_heads, head_dim)
        *,
        B: int,
        L: int,
        M: int,
        cu_seqlens: torch.Tensor = None,
        max_seqlen_in_batch: int = None,
        valid_token_indices: torch.Tensor = None,
    ):
        N = M
        local_heads = self.num_heads
        # local_dim = local_heads * self.head_dim
        # with torch.autocast("cuda", enabled=False):
        #     out: torch.Tensor = flash_attn_varlen_qkvpacked_func(
        #         qkv,
        #         cu_seqlens=cu_seqlens,
        #         max_seqlen=max_seqlen_in_batch,
        #         dropout_p=0.0,
        #         softmax_scale=self.softmax_scale,
        #     )  # (total, local_heads, head_dim)
        #     out = out.view(total, local_dim)

        q, k, v = qkv.unbind(1)
        q = q.permute(1, 0, 2).unsqueeze(0)
        k = k.permute(1, 0, 2).unsqueeze(0)
        v = v.permute(1, 0, 2).unsqueeze(0)
        
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).flatten(2, 3)

        # x, y = pad_and_split_xy(out, valid_token_indices, B, N, L, qkv.dtype)
        x, y = out.split_with_sizes((N, L), dim=1)
        # assert x.size() == (B, N, local_dim)
        # assert y.size() == (B, L, local_dim)

        # x = x.view(B, -1, local_heads, self.head_dim).flatten(2, 3)
        x = self.proj_x(x)  # (B, M, dim_x)

        # y = y.view(B, -1, local_heads, self.head_dim).flatten(2, 3)
        y = self.proj_y(y)  # (B, L, dim_y)
        return x, y

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,  # (B, dim_x), modulation for pre-RMSNorm.
        scale_y: torch.Tensor,  # (B, dim_y), modulation for pre-RMSNorm.
        packed_indices: Dict[str, torch.Tensor] = None,
        **rope_rotation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of asymmetric multi-modal attention.

        Args:
            x: (B, N, dim_x) tensor for visual tokens
            y: (B, L, dim_y) tensor of text token features
            packed_indices: Dict with keys for Flash Attention
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim_x) tensor of visual tokens after multi-modal attention y: (B, L, dim_y) tensor of text token
            features after multi-modal attention
        """
        B, L, _ = y.shape
        _, M, _ = x.shape

        # Predict a packed QKV tensor from visual and text features.
        # Don't checkpoint the all_to_all.
        qkv = self.prepare_qkv(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_rotation.get("rope_cos"),
            rope_sin=rope_rotation.get("rope_sin"),
            # valid_token_indices=packed_indices["valid_token_indices_kv"],
        )  # (total <= B * (N + L), 3, local_heads, head_dim)

        x, y = self.run_attention(
            qkv,
            B=B,
            L=L,
            M=M,
            # cu_seqlens=packed_indices["cu_seqlens_kv"],
            # max_seqlen_in_batch=packed_indices["max_seqlen_in_batch_kv"],
            # valid_token_indices=packed_indices["valid_token_indices_kv"],
        )
        return x, y


def apply_rotary_emb_qk_real(
    xqk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor without complex numbers.

    Args:
        xqk (torch.Tensor): Query and/or Key tensors to apply rotary embeddings. Shape: (B, S, *, num_heads, D)
                            Can be either just query or just key, or both stacked along some batch or * dim.
        freqs_cos (torch.Tensor): Precomputed cosine frequency tensor.
        freqs_sin (torch.Tensor): Precomputed sine frequency tensor.

    Returns:
        torch.Tensor: The input tensor with rotary embeddings applied.
    """
    # assert xqk.dtype == torch.bfloat16
    # Split the last dimension into even and odd parts
    xqk_even = xqk[..., 0::2]
    xqk_odd = xqk[..., 1::2]

    # Apply rotation
    cos_part = (xqk_even * freqs_cos - xqk_odd * freqs_sin).type_as(xqk)
    sin_part = (xqk_even * freqs_sin + xqk_odd * freqs_cos).type_as(xqk)

    # Interleave the results back into the original shape
    out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2)
    # assert out.dtype == torch.bfloat16
    return out


class PadSplitXY(torch.autograd.Function):
    """
    Merge heads, pad and extract visual and text tokens, and split along the sequence length.
    """

    @staticmethod
    def forward(
        ctx,
        xy: torch.Tensor,
        indices: torch.Tensor,
        B: int,
        N: int,
        L: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xy: Packed tokens. Shape: (total <= B * (N + L), num_heads * head_dim).
            indices: Valid token indices out of unpacked tensor. Shape: (total,)

        Returns:
            x: Visual tokens. Shape: (B, N, num_heads * head_dim). y: Text tokens. Shape: (B, L, num_heads * head_dim).
        """
        ctx.save_for_backward(indices)
        ctx.B, ctx.N, ctx.L = B, N, L
        D = xy.size(1)

        # Pad sequences to (B, N + L, dim).
        assert indices.ndim == 1
        output = torch.zeros(B * (N + L), D, device=xy.device, dtype=dtype)
        indices = indices.unsqueeze(1).expand(-1, D)  # (total,) -> (total, num_heads * head_dim)
        output.scatter_(0, indices, xy)
        xy = output.view(B, N + L, D)

        # Split visual and text tokens along the sequence length.
        return torch.tensor_split(xy, (N,), dim=1)


def pad_and_split_xy(xy, indices, B, N, L, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    return PadSplitXY.apply(xy, indices, B, N, L, dtype)


class UnifyStreams(torch.autograd.Function):
    """Unify visual and text streams."""

    @staticmethod
    def forward(
        ctx,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        v_x: torch.Tensor,
        q_y: torch.Tensor,
        k_y: torch.Tensor,
        v_y: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Args:
            q_x: (B, N, num_heads, head_dim)
            k_x: (B, N, num_heads, head_dim)
            v_x: (B, N, num_heads, head_dim)
            q_y: (B, L, num_heads, head_dim)
            k_y: (B, L, num_heads, head_dim)
            v_y: (B, L, num_heads, head_dim)
            indices: (total <= B * (N + L))

        Returns:
            qkv: (total <= B * (N + L), 3, num_heads, head_dim)
        """
        ctx.save_for_backward(indices)
        B, N, num_heads, head_dim = q_x.size()
        ctx.B, ctx.N, ctx.L = B, N, q_y.size(1)
        D = num_heads * head_dim

        q = torch.cat([q_x, q_y], dim=1)
        k = torch.cat([k_x, k_y], dim=1)
        v = torch.cat([v_x, v_y], dim=1)
        qkv = torch.stack([q, k, v], dim=2).view(B * (N + ctx.L), 3, D)

        # indices = indices[:, None, None].expand(-1, 3, D)
        # qkv = torch.gather(qkv, 0, indices)  # (total, 3, num_heads * head_dim)
        return qkv.unflatten(2, (num_heads, head_dim))


def unify_streams(q_x, k_x, v_x, q_y, k_y, v_y, indices) -> torch.Tensor:
    return UnifyStreams.apply(q_x, k_x, v_x, q_y, k_y, v_y, indices)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, device=device)
        self.mod = nn.Linear(hidden_size, 2 * hidden_size, device=device)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, device=device)

    def forward(self, x, c):
        c = F.silu(c)
        shift, scale = self.mod(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MochiTransformer3DModel(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    Ingests text embeddings instead of a label.
    """

    def __init__(
        self,
        *,
        patch_size=2,
        in_channels=4,
        hidden_size_x=1152,
        hidden_size_y=1152,
        depth=48,
        num_heads=16,
        mlp_ratio_x=8.0,
        mlp_ratio_y=4.0,
        t5_feat_dim: int = 4096,
        t5_token_length: int = 256,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        timestep_scale: Optional[float] = None,
        use_extended_posenc: bool = False,
        rope_theta: float = 10000.0,
        device: Optional[torch.device] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.head_dim = hidden_size_x // num_heads  # Head dimension and count is determined by visual.
        self.use_extended_posenc = use_extended_posenc
        self.t5_token_length = t5_token_length
        self.t5_feat_dim = t5_feat_dim
        self.rope_theta = rope_theta  # Scaling factor for frequency computation for temporal RoPE.

        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size_x,
            bias=patch_embed_bias,
            device=device,
        )
        # Conditionings
        # Timestep
        self.t_embedder = TimestepEmbedder(hidden_size_x, bias=timestep_mlp_bias, timestep_scale=timestep_scale)

        # Caption Pooling (T5)
        self.t5_y_embedder = AttentionPool(t5_feat_dim, num_heads=8, output_dim=hidden_size_x, device=device)

        # Dense Embedding Projection (T5)
        self.t5_yproj = nn.Linear(t5_feat_dim, hidden_size_y, bias=True, device=device)

        # Initialize pos_frequencies as an empty parameter.
        self.pos_frequencies = nn.Parameter(torch.empty(3, self.num_heads, self.head_dim // 2, device=device))

        # for depth 48:
        #  b =  0: AsymmetricJointBlock, update_y=True
        #  b =  1: AsymmetricJointBlock, update_y=True
        #  ...
        #  b = 46: AsymmetricJointBlock, update_y=True
        #  b = 47: AsymmetricJointBlock, update_y=False. No need to update text features.
        blocks = []
        for b in range(depth):
            # Joint multi-modal block
            update_y = b < depth - 1
            block = AsymmetricJointBlock(
                hidden_size_x,
                hidden_size_y,
                num_heads,
                mlp_ratio_x=mlp_ratio_x,
                mlp_ratio_y=mlp_ratio_y,
                update_y=update_y,
                device=device,
                **block_kwargs,
            )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(hidden_size_x, patch_size, self.out_channels, device=device)

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C=12, T, H, W) tensor of visual tokens

        Returns:
            x: (B, C=3072, N) tensor of visual tokens with positional embedding.
        """
        return self.x_embedder(x)  # Convert BcTHW to BCN

    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ):
        """Prepare input and conditioning embeddings."""
        with torch.profiler.record_function("x_emb_pe"):
            # Visual patch embeddings with positional encoding.
            T, H, W = x.shape[-3:]
            pH, pW = H // self.patch_size, W // self.patch_size
            x = self.embed_x(x)  # (B, N, D), where N = T * H * W / patch_size ** 2
            assert x.ndim == 3
            B = x.size(0)

        with torch.profiler.record_function("rope_cis"):
            # Construct position array of size [N, 3].
            # pos[:, 0] is the frame index for each location,
            # pos[:, 1] is the row index for each location, and
            # pos[:, 2] is the column index for each location.
            pH, pW = H // self.patch_size, W // self.patch_size
            N = T * pH * pW
            assert x.size(1) == N
            pos = create_position_matrix(T, pH=pH, pW=pW, device=x.device, dtype=torch.float32)  # (N, 3)
            rope_cos, rope_sin = compute_mixed_rotation(
                freqs=self.pos_frequencies, pos=pos
            )  # Each are (N, num_heads, dim // 2)

        with torch.profiler.record_function("t_emb"):
            # Global vector embedding for conditionings.
            c_t = self.t_embedder(1 - sigma)  # (B, D)

        with torch.profiler.record_function("t5_pool"):
            # Pool T5 tokens using attention pooler
            # Note y_feat[1] contains T5 token features.
            assert (
                t5_feat.size(1) == self.t5_token_length
            ), f"Expected L={self.t5_token_length}, got {t5_feat.shape} for y_feat."
            t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)  # (B, D)
            assert t5_y_pool.size(0) == B, f"Expected B={B}, got {t5_y_pool.shape} for t5_y_pool."

        c = c_t + t5_y_pool

        y_feat = self.t5_yproj(t5_feat)  # (B, L, t5_feat_dim) --> (B, L, D)

        return x, c, y_feat, rope_cos, rope_sin

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        packed_indices: Dict[str, torch.Tensor] = None,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
    ):
        """Forward pass of DiT.

        Args:
            x: (B, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
            sigma: (B,) tensor of noise standard deviations
            y_feat:
                List((B, L, y_feat_dim) tensor of caption token features. For SDXL text encoders: L=77,
                y_feat_dim=2048)
            y_mask: List((B, L) boolean tensor indicating which tokens are not padding)
            packed_indices: Dict with keys for Flash Attention. Result of compute_packed_indices.
        """
        B, _, T, H, W = x.shape

        x, c, y_feat, rope_cos, rope_sin = self.prepare(x, sigma, y_feat[0], y_mask[0])
        del y_mask

        for i, block in enumerate(self.blocks):
            x, y_feat = block(
                x,
                c,
                y_feat,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                packed_indices=packed_indices,
            )  # (B, M, D), (B, L, D)
            print(x.mean(), x.std())
        del y_feat  # Final layers don't use dense text features.

        x = self.final_layer(x, c)  # (B, M, patch_size ** 2 * out_channels)

        patch = x.size(2)
        # x = rearrange(x, "(G B) M P -> B (G M) P", G=1, P=patch)
        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )

        return x
