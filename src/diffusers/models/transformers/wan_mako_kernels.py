import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ============================================================================
# Triton Kernels
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BIAS, HAS_BIAS: tl.constexpr,
    ACT_GELU_TANH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    if ACT_GELU_TANH:
        # Fused GELU (tanh approximation)
        x = acc
        c0 = 0.7978845608028654 # sqrt(2/pi)
        c1 = 0.044715
        x3 = x * x * x
        inner = c0 * (x + c1 * x3)
        e2 = tl.exp(2.0 * inner)
        tanh_val = (e2 - 1.0) / (e2 + 1.0)
        acc = 0.5 * x * (1.0 + tanh_val)

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def adaptive_layernorm_kernel(
    X, Scale, Shift, Out,
    stride_xb, stride_xs, stride_xd,
    stride_sb, stride_sd,
    stride_tb, stride_td,
    stride_ob, stride_os, stride_od,
    N, eps,
    BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    off_x = pid_b * stride_xb + pid_s * stride_xs
    off_s = pid_b * stride_sb 
    off_t = pid_b * stride_tb
    off_out = pid_b * stride_ob + pid_s * stride_os

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    x = tl.load(X + off_x + cols * stride_xd, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    norm = x_centered * rstd

    scale = tl.load(Scale + off_s + cols * stride_sd, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(Shift + off_t + cols * stride_td, mask=mask, other=0.0).to(tl.float32)

    out = norm * (1.0 + scale) + shift
    
    tl.store(Out + off_out + cols * stride_od, out, mask=mask)

@triton.jit
def rms_norm_kernel(
    X, W, Out,
    stride_xb, stride_xs, stride_xd,
    stride_w,
    stride_ob, stride_os, stride_od,
    N, eps,
    BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    off_x = pid_b * stride_xb + pid_s * stride_xs
    off_out = pid_b * stride_ob + pid_s * stride_os

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    x = tl.load(X + off_x + cols * stride_xd, mask=mask, other=0.0).to(tl.float32)
    
    ms = tl.sum(x * x, axis=0) / N
    rms = tl.rsqrt(ms + eps)
    
    w = tl.load(W + cols * stride_w, mask=mask, other=0.0).to(tl.float32)
    out = x * rms * w
    
    tl.store(Out + off_out + cols * stride_od, out, mask=mask)

# New: fused RMSNorm for (Q, K) together to eliminate one kernel launch and reuse scheduling
@triton.jit
def rms_norm2_kernel(
    X1, W1, Out1,
    X2, W2, Out2,
    stride_xb, stride_xs, stride_xd,
    stride_w1, stride_w2,
    stride_o1b, stride_o1s, stride_o1d,
    stride_o2b, stride_o2s, stride_o2d,
    N, eps,
    BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    off_x = pid_b * stride_xb + pid_s * stride_xs

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    x1 = tl.load(X1 + off_x + cols * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2 + off_x + cols * stride_xd, mask=mask, other=0.0).to(tl.float32)

    ms1 = tl.sum(x1 * x1, axis=0) / N
    ms2 = tl.sum(x2 * x2, axis=0) / N
    rms1 = tl.rsqrt(ms1 + eps)
    rms2 = tl.rsqrt(ms2 + eps)

    w1 = tl.load(W1 + cols * stride_w1, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(W2 + cols * stride_w2, mask=mask, other=0.0).to(tl.float32)

    y1 = x1 * rms1 * w1
    y2 = x2 * rms2 * w2

    tl.store(Out1 + pid_b * stride_o1b + pid_s * stride_o1s + cols * stride_o1d, y1, mask=mask)
    tl.store(Out2 + pid_b * stride_o2b + pid_s * stride_o2s + cols * stride_o2d, y2, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_resgate_kernel(
    A, B, X, G, C,
    M, N, K, S_rows,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_xm, stride_xn,
    stride_gb, stride_gs, stride_gn,
    stride_cm, stride_cn,
    BIAS, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Load residual X
    x_ptrs = X + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    x_val = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)

    # Compute gate index using batch from rows
    b_rows = (offs_m // S_rows)[:, None]
    g_ptrs = G + (b_rows * stride_gb + 0 * stride_gs + offs_n[None, :] * stride_gn)
    g_val = tl.load(g_ptrs, mask=(offs_n[None, :] < N), other=0.0).to(tl.float32)

    out = x_val + acc * g_val
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_resadd_kernel(
    A, B, X, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_xm, stride_xn,
    stride_cm, stride_cn,
    BIAS, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Load residual and add
    x_ptrs = X + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    x_val = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    out = x_val + acc

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# ============================================================================
# Helpers
# ============================================================================

def triton_matmul(x, w, bias=None, activation=""):
    is_3d = x.ndim == 3
    if is_3d:
        B, S, K = x.shape
        M = B * S
        x_2d = x.view(M, K)
    else:
        M, K = x.shape
        x_2d = x
        
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    stride_am, stride_ak = x_2d.stride(0), x_2d.stride(1)
    stride_bk, stride_bn = w.stride(1), w.stride(0)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    has_bias = bias is not None
    is_gelu = activation == "gelu"
    
    matmul_kernel[grid](
        x_2d, w, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        bias if has_bias else x_2d,
        HAS_BIAS=has_bias,
        ACT_GELU_TANH=is_gelu
    )
    
    if is_3d:
        return out.view(B, S, N)
    return out

def triton_adaptive_norm(x, scale, shift, eps):
    B, S, D = x.shape
    out = torch.empty_like(x)
    
    def get_strides_bd(t):
        if t.ndim == 3: return t.stride(0), t.stride(2)
        return t.stride(0), t.stride(1)
    
    ss_b, ss_d = get_strides_bd(scale)
    st_b, st_d = get_strides_bd(shift)
    
    BLOCK_N = triton.next_power_of_2(D)
    grid = (B, S)
    
    adaptive_layernorm_kernel[grid](
        x, scale, shift, out,
        x.stride(0), x.stride(1), x.stride(2),
        ss_b, ss_d,
        st_b, st_d,
        out.stride(0), out.stride(1), out.stride(2),
        D, eps,
        BLOCK_N=BLOCK_N
    )
    return out

def triton_rms_norm(x, weight, eps):
    B, S, D = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(D)
    grid = (B, S)
    
    rms_norm_kernel[grid](
        x, weight, out,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        D, eps,
        BLOCK_N=BLOCK_N
    )
    return out

def triton_rms_norm2(x1, w1, x2, w2, eps):
    # x1, x2: (B, S, D) with same B,S,D
    B, S, D = x1.shape
    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)
    BLOCK_N = triton.next_power_of_2(D)
    grid = (B, S)
    rms_norm2_kernel[grid](
        x1, w1, out1,
        x2, w2, out2,
        x1.stride(0), x1.stride(1), x1.stride(2),
        w1.stride(0), w2.stride(0),
        out1.stride(0), out1.stride(1), out1.stride(2),
        out2.stride(0), out2.stride(1), out2.stride(2),
        D, eps,
        BLOCK_N=BLOCK_N
    )
    return out1, out2

def fused_matmul_residual_gate(A, W, bias, X, G, S_rows):
    # A: (B, S, K), W: (N, K), X: (B, S, N), G: (B, 1 or None, N) or (B, N)
    B, S, K = A.shape
    M = B * S
    N = W.shape[0]
    A2d = A.view(M, K)
    X2d = X.view(M, N)
    out = torch.empty_like(X2d)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    has_bias = bias is not None
    # Strides for gate (B, 1, N) or (B, N)
    stride_gb = G.stride(0)
    stride_gs = 0 if (G.ndim == 3 and G.shape[1] == 1) else (G.stride(1) if G.ndim == 3 else 0)
    stride_gn = G.stride(-1)

    matmul_bias_resgate_kernel[grid](
        A2d, W, X2d, G, out,
        M, N, K, S_rows,
        A2d.stride(0), A2d.stride(1),
        W.stride(1), W.stride(0),
        X2d.stride(0), X2d.stride(1),
        stride_gb, stride_gs, stride_gn,
        out.stride(0), out.stride(1),
        bias if has_bias else W,  # dummy if no bias
        HAS_BIAS=has_bias
    )
    return out.view(B, S, N)

def fused_matmul_residual(A, W, bias, X):
    # A: (B, S, K), W: (N, K), X: (B, S, N)
    B, S, K = A.shape
    M = B * S
    N = W.shape[0]
    A2d = A.view(M, K)
    X2d = X.view(M, N)
    out = torch.empty_like(X2d)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    has_bias = bias is not None

    matmul_bias_resadd_kernel[grid](
        A2d, W, X2d, out,
        M, N, K,
        A2d.stride(0), A2d.stride(1),
        W.stride(1), W.stride(0),
        X2d.stride(0), X2d.stride(1),
        out.stride(0), out.stride(1),
        bias if has_bias else W,
        HAS_BIAS=has_bias
    )
    return out.view(B, S, N)

# ============================================================================
# Optimized Model
# ============================================================================

class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int = 1536,
        ffn_dim: int = 8960,
        num_heads: int = 12,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        
        # Preserve parameter structure (state_dict compatibility)
        class WanAttention(nn.Module):
            def __init__(self, dim, heads, dim_head, eps, dropout, added_kv_proj_dim, cross_attention_dim_head):
                super().__init__()
                self.inner_dim = dim_head * heads
                self.heads = heads
                self.dim_head = dim_head
                self.added_kv_proj_dim = added_kv_proj_dim
                self.cross_attention_dim_head = cross_attention_dim_head
                self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

                self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
                self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
                self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
                self.to_out = nn.ModuleList([
                    nn.Linear(self.inner_dim, dim, bias=True),
                    nn.Dropout(dropout),
                ])
                self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
                self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

                self.add_k_proj = self.add_v_proj = None
                if added_kv_proj_dim is not None:
                    self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
                    self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
                    self.norm_added_k = nn.RMSNorm(dim_head * heads, eps=eps)

        class FeedForward(nn.Module):
            def __init__(self, dim, inner_dim, dropout, bias=True):
                super().__init__()
                self.net = nn.ModuleList([])
                class GELU(nn.Module):
                    def __init__(self, dim_in, dim_out, approximate, bias):
                        super().__init__()
                        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
                        self.approximate = approximate
                self.net.append(GELU(dim, inner_dim, "tanh", bias))
                self.net.append(nn.Dropout(dropout))
                self.net.append(nn.Linear(inner_dim, dim, bias=bias))

        class FP32LayerNorm(nn.LayerNorm): 
            pass # Placeholder for structure

        class WanTransformerBlock(nn.Module):
            def __init__(self, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim):
                super().__init__()
                self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
                self.attn1 = WanAttention(dim, num_heads, dim // num_heads, eps, 0.0, None, None)
                self.attn2 = WanAttention(dim, num_heads, dim // num_heads, eps, 0.0, added_kv_proj_dim, dim // num_heads)
                self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
                self.ffn = FeedForward(dim, ffn_dim, 0.0, bias=True)
                self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
                self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
                self.eps = eps

        self.block = WanTransformerBlock(dim, ffn_dim, num_heads, "rms_norm_across_heads", cross_attn_norm, eps, None)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    @torch.no_grad()
    def fuse(self):
        # Pre-concatenate weights for fused operations
        # Self-Attention QKV
        self.w_qkv_self = torch.cat([self.block.attn1.to_q.weight, self.block.attn1.to_k.weight, self.block.attn1.to_v.weight], dim=0)
        self.b_qkv_self = torch.cat([self.block.attn1.to_q.bias, self.block.attn1.to_k.bias, self.block.attn1.to_v.bias], dim=0)
        
        # Cross-Attention KV
        self.w_kv_cross = torch.cat([self.block.attn2.to_k.weight, self.block.attn2.to_v.weight], dim=0)
        self.b_kv_cross = torch.cat([self.block.attn2.to_k.bias, self.block.attn2.to_v.bias], dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        block = self.block
        B, S, D = hidden_states.shape
        H = self.num_heads
        Dh = self.head_dim

        # Modulation
        if temb.ndim == 4:
            mods = (block.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [x.squeeze(2) for x in mods]
        else:
            mods = (block.scale_shift_table + temb.float()).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mods

        # --------------------------------------------------------------------
        # 1. Self-Attention
        # --------------------------------------------------------------------
        # Fused Adaptive LayerNorm
        norm_hidden = triton_adaptive_norm(hidden_states, scale_msa, shift_msa, block.eps)
        
        # Fused QKV via single matmul
        qkv = triton_matmul(norm_hidden, self.w_qkv_self, self.b_qkv_self)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Fused RMS Norm for Q and K to reduce 1 launch
        q, k = triton_rms_norm2(q, block.attn1.norm_q.weight, k, block.attn1.norm_k.weight, block.attn1.norm_q.eps)
        
        # Reshape & Attention
        q = q.view(B, S, H, Dh).transpose(1, 2)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        
        # Fused Output Projection + Gated Residual
        hidden_states = fused_matmul_residual_gate(
            attn_out, block.attn1.to_out[0].weight, block.attn1.to_out[0].bias,
            hidden_states, gate_msa, S_rows=S
        )
        
        # --------------------------------------------------------------------
        # 2. Cross-Attention
        # --------------------------------------------------------------------
        # Norm
        if not isinstance(block.norm2, nn.Identity):
            norm_hidden = block.norm2(hidden_states.float()).type_as(hidden_states)
        else:
            norm_hidden = hidden_states
            
        # Q Projection
        q2 = triton_matmul(norm_hidden, block.attn2.to_q.weight, block.attn2.to_q.bias)
        
        # Fused KV Projection
        kv2 = triton_matmul(encoder_hidden_states, self.w_kv_cross, self.b_kv_cross)
        k2, v2 = kv2.chunk(2, dim=-1)
        
        # RMS Norm
        q2 = triton_rms_norm(q2, block.attn2.norm_q.weight, block.attn2.norm_q.eps)
        k2 = triton_rms_norm(k2, block.attn2.norm_k.weight, block.attn2.norm_k.eps)
        
        # Attention
        q2 = q2.view(B, S, H, Dh).transpose(1, 2)
        T_text = encoder_hidden_states.shape[1]
        k2 = k2.view(B, T_text, H, Dh).transpose(1, 2)
        v2 = v2.view(B, T_text, H, Dh).transpose(1, 2)
        
        attn_out2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=False)
        attn_out2 = attn_out2.transpose(1, 2).contiguous().view(B, S, D)
        
        # Fused Cross-Attn Output Proj + Residual (no gate)
        hidden_states = fused_matmul_residual(
            attn_out2, block.attn2.to_out[0].weight, block.attn2.to_out[0].bias, hidden_states
        )
        
        # --------------------------------------------------------------------
        # 3. Feed-Forward
        # --------------------------------------------------------------------
        # Adaptive Norm
        norm_hidden = triton_adaptive_norm(hidden_states, c_scale_msa, c_shift_msa, block.eps)
        
        # Fused Linear + GELU
        ff_in = triton_matmul(
            norm_hidden, 
            block.ffn.net[0].proj.weight, 
            block.ffn.net[0].proj.bias, 
            activation="gelu"
        )
        
        # Fused Second Linear + Gated Residual
        hidden_states = fused_matmul_residual_gate(
            ff_in, block.ffn.net[2].weight, block.ffn.net[2].bias,
            hidden_states, c_gate_msa, S_rows=S
        )
        
        return hidden_states


def get_inputs():
    # randomly generate input tensors based on the model architecture (Wan 1.3B)
    batch_size = 2
    seq_len = 256  # Number of latent tokens (e.g., from video patches)
    dim = 1536  # Hidden dimension for Wan 1.3B

    # hidden_states: [batch_size, seq_len, dim]
    hidden_states = torch.randn(batch_size, seq_len, dim).cuda()

    # encoder_hidden_states: [batch_size, text_seq_len, dim] (text embeddings)
    text_seq_len = 512
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, dim).cuda()

    # temb: [batch_size, 6, dim] (timestep embedding projected to 6 modulation vectors)
    temb = torch.randn(batch_size, 6, dim).cuda()

    return [hidden_states, encoder_hidden_states, temb]


def get_init_inputs():
    # Initialization parameters for Wan 1.3B: dim, ffn_dim, num_heads, cross_attn_norm, eps
    return [1536, 8960, 12, False, 1e-6]

if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).cuda()
    model.fuse()
    # Get inputs and run forward pass
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)

    print(f"{output.shape=}")
