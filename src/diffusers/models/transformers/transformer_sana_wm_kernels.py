# Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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

# ruff: noqa: E501

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F


# Optional ``einops`` import. Only a handful of kernel-prep helpers in this
# file use ``rearrange`` / ``repeat``; keep the dep optional so that
# ``import diffusers.models.transformers.transformer_sana_wm_kernels`` works
# on minimal-deps installs and we only error out if those helpers are called.
try:
    from einops import rearrange, repeat
except ImportError:

    def rearrange(*args, **kwargs):
        raise ImportError("`einops` is required to run SANA-WM kernels. Install with `pip install einops`.")

    def repeat(*args, **kwargs):
        raise ImportError("`einops` is required to run SANA-WM kernels. Install with `pip install einops`.")


# Optional Triton import. The kernels below are the fast path on CUDA + Triton
# >= 3.x, but they are not correctness-essential: SanaWMTransformer3DModel has
# pure-PyTorch attention variants for every ``*Triton`` class (the dispatcher
# in ``transformer_sana_wm.py`` auto-falls-back when Triton isn't usable). On
# a Triton-less system, ``@triton.jit`` becomes a no-op so the kernel function
# *definitions* still load (so the module can be imported anywhere), but
# calling any of the Triton-backed entry points raises a clear error.
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

    class _TritonShim:
        """No-op stand-in for ``triton`` / ``triton.language`` on systems without Triton.

        ``@triton.jit`` becomes a pass-through so the @-decorated kernel
        functions are still defined as plain Python (and never called on the
        torch fallback path). Any attribute access returns the same shim so
        ``tl.constexpr``, ``tl.load`` etc. evaluate to a harmless sentinel —
        which is fine as long as no kernel body actually executes.
        """

        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]
            return self

        def jit(self, fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

    triton = _TritonShim()
    tl = _TritonShim()


def is_triton_available() -> bool:
    """Whether ``triton`` was importable and the kernels in this module can be launched."""
    return _TRITON_AVAILABLE


def _require_triton(entry_point: str) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            f"{entry_point} requires the `triton` package to run. Install Triton "
            f"or switch to the pure-PyTorch attention variant (e.g. drop the "
            f"`Triton` suffix from `attn_type` / `camctrl_type` on "
            f"SanaWMTransformer3DModel — the dispatcher does this automatically "
            f"when Triton isn't usable)."
        )


# =====================================================================
#  GPU-adaptive kernel config
# =====================================================================


def _get_kernel_config() -> dict:
    """Return optimal kernel parameters for the current GPU.

    STATE_FP32: use fp32 state_prev when SRAM is large enough.
      - bf16 state_prev: ~96KB total SRAM (fits GB10's 101KB).
      - fp32 state_prev: ~128KB total SRAM (needs H100's 228KB+).
    """
    if not torch.cuda.is_available():
        return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 4, "STATE_FP32": False}
    smem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
    state_fp32 = smem >= 150 * 1024  # H100 (228KB) yes, GB10 (101KB) no
    return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 8, "STATE_FP32": state_fp32}


_KCFG = None


def _kcfg():
    global _KCFG
    if _KCFG is None:
        _KCFG = _get_kernel_config()
    return _KCFG


# precision=0 → IEEE fp32 dots + fp32 state  (DOT_PRECISION=2, STATE_FP32=1)
# precision=1 → TF32  dots   + fp32 state    (DOT_PRECISION=1, STATE_FP32=1)
# precision=2 → bf16  dots   + fp32 state    (DOT_PRECISION=0, STATE_FP32=1) [default]
# precision=3 → bf16  dots   + bf16 state    (DOT_PRECISION=0, STATE_FP32=0)
def _precision_params(precision: int) -> tuple:
    if precision == 0:
        return 2, True
    elif precision == 1:
        return 1, True
    elif precision == 3:
        return 0, False
    else:  # default
        return 0, True


_env_prec = os.environ.get("FUSED_GDN_PRECISION", None)
PRECISION_OVERRIDE: int | None = int(_env_prec) if _env_prec is not None else None


def _resolve_launch_config() -> tuple:
    """Returns (prec, dot_prec, state_fp32, num_warps).

    Uses ``PRECISION_OVERRIDE`` when set; otherwise falls back to ``_kcfg()``
    (which picks ``STATE_FP32`` based on per-GPU SRAM). ``num_warps`` is
    clamped to 4 when dots run on fp32 operands (more registers needed).
    """
    cfg = _kcfg()
    prec = PRECISION_OVERRIDE if PRECISION_OVERRIDE is not None else 2
    dot_prec, state_fp32 = _precision_params(prec)
    if PRECISION_OVERRIDE is None:
        state_fp32 = cfg["STATE_FP32"]
    nw = cfg["num_warps"]
    if dot_prec >= 1:
        nw = min(nw, 4)
    return prec, dot_prec, state_fp32, nw


def prepare_rope_tables(rotary_emb, N: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex rotary_emb `(1, 1, N, D//2)` → expanded (N, D) cos/sin tables.

    Encodes the interleaved-pair rotation
        y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    as  y[d] = x[d]*cos_exp[d] + x[d^1]*sin_exp[d]
    where sin_exp[2i] = -sin[i], sin_exp[2i+1] = +sin[i].

    Returns (cos_exp, sin_exp) both (N, D) float32, contiguous.
    """
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)  # (N, D//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D)
    return rope_cos.contiguous(), rope_sin.contiguous()


def _precompute_inv_rms(qkv: torch.Tensor, idx: int, C: int, eps: float = 1e-5) -> torch.Tensor:
    """Compute 1/RMS for one component of QKV over the full C = H*D channel dim.

    Args:
      qkv:   (B, N, 3, H, D)
      idx:   0 for Q, 1 for K, 2 for V
      C:     H*D (channel count)
      eps:   RMSNorm epsilon

    Returns:
      inv_rms: (B, N) float32
    """
    raw = qkv[:, :, idx].float()  # (B, N, H, D)
    sq_sum = (raw * raw).sum(dim=(-2, -1))  # (B, N)
    return torch.rsqrt(sq_sum / C + eps)


# =====================================================================
#  Fused single-pass Q+K inverse-RMS Triton kernel
# =====================================================================
# Single Triton launch that reads each `(b, n)` row of `qkv` once and emits
# both `q_inv_rms[b, n]` and `k_inv_rms[b, n]`. Replaces two separate PyTorch
# scans (cast→square→sum→rsqrt) over `qkv[:, :, 0]` and `qkv[:, :, 1]`.
#
# Layout assumed: `qkv` is (B, N, 3, H, D) contiguous, so the C = H*D channels
# for a given (b, n, qkv_idx) live in a contiguous memory span.


@triton.jit
def _fused_qk_inv_rms_kernel(
    qkv_ptr,  # *T_in     (B, N, 3, H, D), contiguous
    q_inv_rms_ptr,  # *float32  (B, N)
    k_inv_rms_ptr,  # *float32  (B, N)
    N: tl.constexpr,
    C: tl.constexpr,  # H * D
    eps,
    BLOCK_C: tl.constexpr,
):
    bn_id = tl.program_id(0)
    qkv_row_stride = 3 * C
    row_base = bn_id * qkv_row_stride
    q_base = row_base
    k_base = row_base + C

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    q_vals = tl.load(qkv_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(qkv_ptr + k_base + offs, mask=mask, other=0.0).to(tl.float32)

    q_sq = tl.sum(q_vals * q_vals, axis=0)
    k_sq = tl.sum(k_vals * k_vals, axis=0)

    inv_c = 1.0 / C
    q_inv = tl.rsqrt(q_sq * inv_c + eps)
    k_inv = tl.rsqrt(k_sq * inv_c + eps)

    tl.store(q_inv_rms_ptr + bn_id, q_inv)
    tl.store(k_inv_rms_ptr + bn_id, k_inv)


def fused_qk_inv_rms(
    qkv: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass Triton fused Q+K inverse-RMS.

    Replaces ``(_precompute_inv_rms(qkv, 0, C, eps), _precompute_inv_rms(qkv, 1, C, eps))``
    with one launch that reads each ``(b, n)`` row of ``qkv`` exactly once.

    Args:
      qkv: (B, N, 3, H, D) contiguous tensor, any fp dtype.
      eps: RMSNorm epsilon.

    Returns:
      (q_inv_rms, k_inv_rms), each (B, N) float32 contiguous.
    """
    _require_triton("fused_qk_inv_rms")
    assert qkv.is_contiguous(), "qkv must be contiguous (B, N, 3, H, D)"
    assert qkv.dim() == 5 and qkv.shape[2] == 3, f"expected (B, N, 3, H, D), got {tuple(qkv.shape)}"
    B, N, _, H, D = qkv.shape
    C = H * D
    q_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    k_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    BLOCK_C = triton.next_power_of_2(C)
    _fused_qk_inv_rms_kernel[(B * N,)](
        qkv,
        q_inv_rms,
        k_inv_rms,
        N=N,
        C=C,
        eps=eps,
        BLOCK_C=BLOCK_C,
    )
    return q_inv_rms, k_inv_rms


# =====================================================================
#  Bidirectional GDN entry point (delegates to chunkwise)
# =====================================================================


def fused_bigdn_func(
    qkv: torch.Tensor,  # (B, N, 3, H, D)
    q_inv_rms: torch.Tensor,  # (B, N) float32
    k_inv_rms: torch.Tensor,  # (B, N) float32
    q_norm_weight: torch.Tensor,  # (C,) float32
    k_norm_weight: torch.Tensor,  # (C,) float32
    rope_cos: torch.Tensor,  # (N, D) float32
    rope_sin: torch.Tensor,  # (N, D) float32
    beta: torch.Tensor,  # (B, H, F, S)
    decay: torch.Tensor,  # (B, H, F)
    F: int,
    S: int,
    k_scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bidirectional fused GDN. Returns ``(B, N, H, D)``.

    Thin entry point kept for call-site stability; delegates to
    :func:`fused_bigdn_bidi_chunkwise` from ``fused_gdn_chunkwise``.
    """
    _require_triton("fused_bigdn_func")
    return fused_bigdn_bidi_chunkwise(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
    )


# =============================================================================
# Scalar helpers
# =============================================================================


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix batch (closed-form).

    Mirrors the production ``_invert_SE3`` in ``sana_camctrl_blocks.py``;
    inlined to keep this module dependency-light.
    """
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _process_camera_conditions_raymats_only(
    camera_conditions: torch.Tensor,
    B: int,
    HW: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Lightweight variant of ``_process_camera_conditions_ucpe`` — raymats only.

    Computes *only* the per-ray ``world -> ray_local`` SE(3) transforms used
    by UCPE single-path.  Skips the ``compute_up_lat_map`` path (absmap) that
    the cam branch never consumes — that saves ~1 ms per block on H100.

    Args:
        camera_conditions: ``(B, F, 20)`` — ``[c2w_16 | fx | fy | cx | cy]``.
        B: Batch size (redundant with ``camera_conditions.shape[0]``; kept
            for parity with the production signature).
        HW: ``(T_latent, H_latent, W_latent)`` from the caller.
        patch_size: ``(pt, ph, pw)`` patch embedding stride.

    Returns:
        ``raymats`` of shape ``(B, F, H_latent, W_latent, 4, 4)``.
    """
    F_dim = camera_conditions.shape[1]
    c2w_flat = camera_conditions[..., :16]
    C_to_W = c2w_flat.view(B, F_dim, 4, 4)

    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]
    H_dim, W_dim = HW[1], HW[2]
    image_width = W_dim * patch_size[2]
    image_height = H_dim * patch_size[1]

    xi = torch.zeros(
        (B, F_dim),
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    )
    x_fov = compute_fov_from_fx_xi(
        fx,
        xi,
        image_width,
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    ).view(B, F_dim)
    y_fov = compute_fov_from_fx_xi(
        fy,
        xi,
        image_height,
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    ).view(B, F_dim)

    d_cam = ucm_unproject_grid_fov(
        x_fov,
        y_fov,
        xi,
        H_dim,
        W_dim,
        cx / patch_size[2],
        cy / patch_size[1],
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    )
    if d_cam.ndim == 4 and d_cam.shape[0] == B * F_dim:
        d_cam = d_cam.view(B, F_dim, H_dim, W_dim, 3)

    return world_to_ray_mats(d_cam, C_to_W)  # (B, F, H, W, 4, 4)


def _precompute_cam_inv_rms(raw: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute ``1/RMS`` per ``(b, n)`` over full-``C`` channels.

    Args:
        raw: ``(B, N, H, D)`` raw QKV projection output (typically fp32).
        eps: RMSNorm epsilon.

    Returns:
        ``inv_rms`` of shape ``(B, N)`` in fp32, contiguous.
    """
    B, N, H, D = raw.shape
    C = H * D
    sq_sum = (raw.float() * raw.float()).sum(dim=(-1, -2))  # (B, N)
    return torch.rsqrt(sq_sum / C + eps).contiguous()


def _prepare_ucpe_rope_tables(
    rotary_emb_cam: torch.Tensor,
    N: int,
    D_half: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complex RoPE ``(1, 1, N, D_half//2)`` to interleaved ``(N, D_half)`` cos/sin.

    Uses the interleaved-pair convention:
        y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    encoded as  ``y[d] = x[d]*cos_exp[d] + x[d^1]*sin_exp[d]`` with
        sin_exp[2i] = -sin[i], sin_exp[2i+1] = +sin[i].
    """
    del device  # all outputs inherit device from freqs
    freqs = rotary_emb_cam.squeeze(0).squeeze(0)  # (N, D_half//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1).contiguous()
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D_half).contiguous()
    return rope_cos, rope_sin


# =============================================================================
# Triton kernels — lifted verbatim from cam_gdn_playground.py::TritonCamBranch
# =============================================================================


_DEFAULT_BLOCK_S = 64


@triton.jit
def _cam_prep_kernel(
    q_raw_ptr,  # (B, N, H, D) contiguous, any fp dtype
    k_raw_ptr,  # (B, N, H, D) contiguous (post short-conv on K)
    v_raw_ptr,  # (B, N, H, D) contiguous
    q_inv_rms_ptr,  # (B, N) float32 — precomputed over full C channels
    k_inv_rms_ptr,  # (B, N) float32
    q_norm_w_ptr,  # (C,) = (H*D,) float32
    k_norm_w_ptr,  # (C,) float32
    proj_q_ptr,  # (B, N, 4, 4) — applied to Q first D/2 dims (P_T)
    proj_kv_ptr,  # (B, N, 4, 4) — applied to K,V first D/2 dims (P_inv)
    rope_cos_ptr,  # (N, D_rope) float32, D_rope = D//2
    rope_sin_ptr,  # (N, D_rope) float32
    # --- outputs in (B, H, D, N) layout, same strides pattern ---
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    k_pre_norm_sq_ptr,  # (B, H, N) float32 — ||k_pre_ucpe||^2
    k_post_norm_sq_ptr,  # (B, H, N) float32 — ||k_post_ucpe||^2
    # --- dims ---
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,  # head dim
    D_HALF: tl.constexpr,  # D // 2
    N_GROUPS: tl.constexpr,  # D_HALF // 4
    K_SCALE,
    # --- tile sizes ---
    BLOCK_D_ROPE: tl.constexpr,  # next pow2 of D_HALF (rope block)
    BLOCK_GROUPS: tl.constexpr,  # next pow2 of N_GROUPS
):
    """One program per (b, n, h) — processes a single (Q, K, V) head slice.

    Loads the first D_HALF dims as a (N_GROUPS, 4) tile (for the UCPE
    block-diagonal 4x4 projmat), and the second D_HALF dims as a
    (D_HALF,) vector (for RoPE). No redundant loads.
    """
    pid = tl.program_id(0)
    h_idx = pid % H
    bn_idx = pid // H
    b_idx = bn_idx // N
    n_idx = bn_idx % N

    # layout (B, N, H, D) contiguous
    row_base = b_idx * (N * H * D) + n_idx * (H * D) + h_idx * D
    nw_off = h_idx * D

    # ---- load inv-RMS (scalar, shared across heads for this token) ----
    q_inv_rms = tl.load(q_inv_rms_ptr + bn_idx).to(tl.float32)
    k_inv_rms = tl.load(k_inv_rms_ptr + bn_idx).to(tl.float32)

    # ---- load per-token P matrices (4,4) shared across heads ----
    proj_base = (b_idx * N + n_idx) * 16
    offs_i = tl.arange(0, 4)
    offs_j = tl.arange(0, 4)
    P_q = tl.load(proj_q_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]).to(tl.float32)
    P_kv = tl.load(proj_kv_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]).to(tl.float32)

    # ==================================================================
    # Pass 1 — UCPE block-diagonal projmat on first D_HALF dims
    # ==================================================================
    offs_g = tl.arange(0, BLOCK_GROUPS)
    mask_g = offs_g < N_GROUPS
    offs_gj = offs_g[:, None] * 4 + offs_j[None, :]  # (BLOCK_GROUPS, 4)
    mask_gj = mask_g[:, None]

    q_half = tl.load(q_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    k_half = tl.load(k_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    v_half = tl.load(v_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)

    q_nw_half = tl.load(q_norm_w_ptr + nw_off + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)
    k_nw_half = tl.load(k_norm_w_ptr + nw_off + offs_gj, mask=mask_gj, other=0.0).to(tl.float32)

    q_half = q_half * q_inv_rms * q_nw_half
    q_half = tl.where(q_half > 0, q_half, 0.0)

    k_half = k_half * k_inv_rms * k_nw_half
    k_half = tl.where(k_half > 0, k_half, 0.0) * K_SCALE

    # Pre-UCPE ||k||^2 contribution from first half
    k_half_masked = tl.where(mask_gj, k_half, 0.0)
    k_pre_half_sq = tl.sum(k_half_masked * k_half_masked)

    # Apply 4x4 projmat: out[g, i] = sum_j P[i, j] * in[g, j]
    # (BLOCK_GROUPS, 1, 4) * (1, 4, 4) -> (BLOCK_GROUPS, 4, 4), sum axis=-1
    q_half_out = tl.sum(q_half[:, None, :] * P_q[None, :, :], axis=-1)
    k_half_out = tl.sum(k_half[:, None, :] * P_kv[None, :, :], axis=-1)
    v_half_out = tl.sum(v_half[:, None, :] * P_kv[None, :, :], axis=-1)

    # Post-UCPE ||k||^2 contribution from first half
    k_half_out_masked = tl.where(mask_gj, k_half_out, 0.0)
    k_post_half_sq = tl.sum(k_half_out_masked * k_half_out_masked)

    # ==================================================================
    # Pass 2 — RoPE on second D_HALF dims
    # ==================================================================
    offs_r = tl.arange(0, BLOCK_D_ROPE)
    mask_r = offs_r < D_HALF
    offs_r_pair = offs_r ^ 1
    mask_r_pair = offs_r_pair < D_HALF

    rope_row = n_idx * D_HALF
    cos_v = tl.load(rope_cos_ptr + rope_row + offs_r, mask=mask_r, other=1.0).to(tl.float32)
    sin_v = tl.load(rope_sin_ptr + rope_row + offs_r, mask=mask_r, other=0.0).to(tl.float32)

    # Load second-half raw values and their pair partners
    rope_base = row_base + D_HALF
    q_r = tl.load(q_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    k_r = tl.load(k_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    v_r = tl.load(v_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    q_r_pair = tl.load(q_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)
    k_r_pair = tl.load(k_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)
    v_r_pair = tl.load(v_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)

    q_nw_r = tl.load(q_norm_w_ptr + nw_off + D_HALF + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    k_nw_r = tl.load(k_norm_w_ptr + nw_off + D_HALF + offs_r, mask=mask_r, other=0.0).to(tl.float32)
    q_nw_r_pair = tl.load(q_norm_w_ptr + nw_off + D_HALF + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)
    k_nw_r_pair = tl.load(k_norm_w_ptr + nw_off + D_HALF + offs_r_pair, mask=mask_r_pair, other=0.0).to(tl.float32)

    q_r_n = q_r * q_inv_rms * q_nw_r
    q_r_n = tl.where(q_r_n > 0, q_r_n, 0.0)
    q_r_pair_n = q_r_pair * q_inv_rms * q_nw_r_pair
    q_r_pair_n = tl.where(q_r_pair_n > 0, q_r_pair_n, 0.0)

    k_r_n = k_r * k_inv_rms * k_nw_r
    k_r_n = tl.where(k_r_n > 0, k_r_n, 0.0) * K_SCALE
    k_r_pair_n = k_r_pair * k_inv_rms * k_nw_r_pair
    k_r_pair_n = tl.where(k_r_pair_n > 0, k_r_pair_n, 0.0) * K_SCALE

    # Pre-UCPE ||k||^2 contribution from second half (using post-ReLU/scale k_r_n)
    k_r_n_masked = tl.where(mask_r, k_r_n, 0.0)
    k_pre_rope_sq = tl.sum(k_r_n_masked * k_r_n_masked)

    q_rope_out = q_r_n * cos_v + q_r_pair_n * sin_v
    k_rope_out = k_r_n * cos_v + k_r_pair_n * sin_v
    v_rope_out = v_r * cos_v + v_r_pair * sin_v

    # Post-UCPE ||k||^2 contribution from second half
    k_rope_masked = tl.where(mask_r, k_rope_out, 0.0)
    k_post_rope_sq = tl.sum(k_rope_masked * k_rope_masked)

    # Store scalar per-token norm squares
    norm_out_idx = (b_idx * H + h_idx) * N + n_idx
    tl.store(k_pre_norm_sq_ptr + norm_out_idx, k_pre_half_sq + k_pre_rope_sq)
    tl.store(k_post_norm_sq_ptr + norm_out_idx, k_post_half_sq + k_post_rope_sq)

    # ==================================================================
    # Store outputs in (B, H, D, N) layout: ptr[b, h, d, n] = base_bh + d*N + n
    # ==================================================================
    out_base = b_idx * (H * D * N) + h_idx * (D * N) + n_idx

    # First half: d = g*4 + i, write at out_base + d*N (strided by N).
    offs_d_half = offs_g[:, None] * 4 + offs_i[None, :]  # (BLOCK_GROUPS, 4)
    mask_d_half = mask_g[:, None]
    tl.store(q_out_ptr + out_base + offs_d_half * N, q_half_out, mask=mask_d_half)
    tl.store(k_out_ptr + out_base + offs_d_half * N, k_half_out, mask=mask_d_half)
    tl.store(v_out_ptr + out_base + offs_d_half * N, v_half_out, mask=mask_d_half)

    # Second half (RoPE region): d = D_HALF + r
    offs_d_r = D_HALF + offs_r  # (BLOCK_D_ROPE,)
    tl.store(q_out_ptr + out_base + offs_d_r * N, q_rope_out, mask=mask_r)
    tl.store(k_out_ptr + out_base + offs_d_r * N, k_rope_out, mask=mask_r)
    tl.store(v_out_ptr + out_base + offs_d_r * N, v_rope_out, mask=mask_r)


def cam_prep_func(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,  # (B, N, 4, 4)
    proj_kv: torch.Tensor,  # (B, N, 4, 4)
    rope_cos: torch.Tensor,  # (N, D//2)
    rope_sin: torch.Tensor,  # (N, D//2)
    k_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + ReLU + (K-scale on K) + UCPE 4x4 + RoPE for the cam branch.

    Args:
        q_raw, k_raw, v_raw: ``(B, N, H, D)`` contiguous (any fp dtype).
            ``K`` must already have the short convolution applied.
        q_norm_weight, k_norm_weight: ``(C,) = (H*D,)`` fp32.
        proj_q, proj_kv: ``(B, N, 4, 4)`` fp32 (``P_T`` and ``P_inv`` in UCPE).
        rope_cos, rope_sin: ``(N, D//2)`` fp32 interleaved-pair tables.
        k_scale: ``(D^-0.5) * (S^-0.5)``.
        norm_eps: RMSNorm epsilon.

    Returns:
        q_trans, k_trans, v_trans: ``(B, H, D, N)`` same dtype as ``q_raw``.
        inflation_sq: ``(B, H, N)`` fp32, ratio
            ``(||k_post_ucpe|| / ||k_pre_ucpe||)^2`` per token/head.
    """
    _require_triton("cam_prep_func")
    B, N, H, D = q_raw.shape
    assert k_raw.shape == q_raw.shape and v_raw.shape == q_raw.shape
    assert D % 2 == 0 and (D // 2) % 4 == 0, f"D={D} must be 2x and (D/2) % 4 == 0"
    D_half = D // 2
    N_groups = D_half // 4

    assert q_raw.is_contiguous() and k_raw.is_contiguous() and v_raw.is_contiguous()
    assert proj_q.shape == (B, N, 4, 4) and proj_q.is_contiguous()
    assert proj_kv.shape == (B, N, 4, 4) and proj_kv.is_contiguous()
    assert rope_cos.shape == (N, D_half) and rope_cos.is_contiguous()
    assert rope_sin.shape == (N, D_half) and rope_sin.is_contiguous()
    assert q_norm_weight.numel() == H * D and q_norm_weight.dtype == torch.float32
    assert k_norm_weight.numel() == H * D and k_norm_weight.dtype == torch.float32

    # Precompute inv-RMS over full C channels (shared across heads per token).
    q_inv_rms = _precompute_cam_inv_rms(q_raw, norm_eps)
    k_inv_rms = _precompute_cam_inv_rms(k_raw, norm_eps)

    out_dtype = q_raw.dtype
    q_out = torch.empty(B, H, D, N, dtype=out_dtype, device=q_raw.device)
    k_out = torch.empty(B, H, D, N, dtype=out_dtype, device=q_raw.device)
    v_out = torch.empty(B, H, D, N, dtype=out_dtype, device=q_raw.device)
    k_pre_sq = torch.empty(B, H, N, dtype=torch.float32, device=q_raw.device)
    k_post_sq = torch.empty(B, H, N, dtype=torch.float32, device=q_raw.device)

    BLOCK_D_ROPE = triton.next_power_of_2(D_half)
    BLOCK_GROUPS = triton.next_power_of_2(N_groups)

    grid = (B * N * H,)
    _cam_prep_kernel[grid](
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        proj_q,
        proj_kv,
        rope_cos,
        rope_sin,
        q_out,
        k_out,
        v_out,
        k_pre_sq,
        k_post_sq,
        H=H,
        N=N,
        D=D,
        D_HALF=D_half,
        N_GROUPS=N_groups,
        K_SCALE=k_scale,
        BLOCK_D_ROPE=BLOCK_D_ROPE,
        BLOCK_GROUPS=BLOCK_GROUPS,
        num_warps=1,
    )
    # inflation_sq = (clamp(sqrt(post), 1e-6) / clamp(sqrt(pre), 1e-6))^2
    #              = clamp(post, 1e-12) / clamp(pre, 1e-12)  (equivalent).
    inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)
    return q_out, k_out, v_out, inflation_sq


_CAM_IDENTITY_CACHE: dict[
    tuple[str, int | None, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}

# ════════════════════════════════════════════════════════════════
#  Per-architecture launch config (auto-selected via compute capability)
# ════════════════════════════════════════════════════════════════
#
# Empirically tuned at production config (B=1..8, T=11, S=920, H=20, D=112) on
# A100 / H100 / GB200. Two effects matter:
#
# 1. **Precision sets BLOCK_S**: fp32 operand fragments are 2× the size of
#    bf16. BLOCK_S=64 + fp32 → register spills (catastrophic, 40-100× slower).
#    BLOCK_S=32 + fp32 → no spills. So fp32 mode forces BLOCK_S=32 everywhere.
#
# 2. **Arch sets BLOCK_S for bf16**: A100 (192 KB SRAM, fewer registers per
#    block) prefers BLOCK_S=32 even at bf16. H100/GB200 (228 KB SRAM) tolerate
#    BLOCK_S=64 cleanly at bf16.
#
# Each entry: (phase_a_warps, phase_a_BLOCK_S,
#              phase_b_warps, phase_b_stages,
#              phase_c_warps, phase_c_BLOCK_S, phase_c_stages)

# ── Launch-config tuning table ─────────────────────────────────────
#
# We tune 8 knobs across 3 phases:
#   Phase A : (nw, BS)               streaming accumulator in registers
#   Phase B : (nw, use_acc, ns)      serial-F scan with persistent M in regs
#   Phase C : (nw, BS, ns)           streams Pass-2 output; loads fp32 M[128,128]
#
# Each arch × precision combination gets a named entry below. Values come from
# empirical sweeps (see commit log: T6 A100/H100 sweep 2026-04-19; Blackwell-DC
# 2026-04-20; Spark GB10 tuning notes in commits 5da52db6 / 3ad104d0) and from
# kernel-structure analysis (Phase B's persistent M[128,128] fp32 is 64 KB → nw
# controls register spread; Phase C's loaded M[128,128] is 64 KB → BS controls
# transient SMEM footprint).
#
# Adding a new arch: pick the closest existing bucket, then override individual
# fields in _CHUNKWISE_SHAPE_OVERRIDES once a targeted sweep lands.


@dataclass(frozen=True)
class _PhaseCfg:
    nw: int  # num_warps
    BS: int = 0  # BLOCK_S (Phase A/C only; 0 = N/A for Phase B)
    ns: int = 1  # num_stages
    use_acc: bool = False  # Phase B only: fold A_f via MMA accumulator


@dataclass(frozen=True)
class _ChunkwiseCfg:
    A: _PhaseCfg
    B: _PhaseCfg
    C: _PhaseCfg

    def as_tuple(self) -> tuple:
        """Flatten to the 8-tuple the legacy API returns."""
        return (
            self.A.nw,
            self.A.BS,
            self.B.nw,
            self.B.ns,
            self.B.use_acc,
            self.C.nw,
            self.C.BS,
            self.C.ns,
        )


# ──────────────────────────────────────────────────────────────────
# Primary tuning table: (arch_key, prec_key) → _ChunkwiseCfg.
# Arch keys:
#   "ampere"          sm_80     A100 (164 KB SRAM, no WGMMA)
#   "hopper"          sm_90     H100 (228 KB SRAM, WGMMA)
#   "blackwell_dc"    sm_100    B200 / GB200 (228 KB SRAM, WGMMA v2)
#   "blackwell_spark" sm_120+ with < 150 KB SRAM  5090 / GB10 (~102 KB SRAM)
# Prec keys:
#   "bf16"  dot_prec == 0  (bf16 TC, half-size operand fragments)
#   "fp32"  dot_prec >= 1  (TF32 TC or IEEE Markidis 3-pass; same launch shape)
# ──────────────────────────────────────────────────────────────────
_CHUNKWISE_TUNING: dict[tuple[str, str], _ChunkwiseCfg] = {
    # A100: smaller SRAM than Hopper, no WGMMA → bigger CTAs hide MMA latency.
    # Phase B fp32 needs nw=32 to spread persistent M across warps (no acc-fusion
    # available pre-Hopper, so ns=2 fills the MMA pipeline slot instead).
    ("ampere", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=4, BS=32, ns=1),  # nw=4 bf16 C: 27% faster than nw=8 per T6
    ),
    ("ampere", "fp32"): _ChunkwiseCfg(
        # 2026-04-30 PM retune: Phase A nw=8 → 16 BS=32 yields 8-13× speedup
        # across F ∈ {3, 5, 11, 14, 17, 20} (cos=1.0 verified). Old nw=8 was a
        # legacy default never re-swept; sweep showed nw=16 dominates every F.
        # Closes A100 sink/rolling chunkwise regression where Phase B was
        # already optimal (sub-percent tuning gap) — Phase A was the bottleneck.
        A=_PhaseCfg(nw=16, BS=32),
        B=_PhaseCfg(nw=32, use_acc=False, ns=2),  # ns=2 fills pipe (no acc-fusion)
        C=_PhaseCfg(nw=16, BS=32, ns=1),  # 2026-04-30 retune: nw=16 BS=32 is 2.8x faster (was nw=8 BS=16)
    ),
    # Hopper (H100): WGMMA + 228 KB SRAM → big tiles win at bf16.
    # Phase B fp32 uses acc-fusion (MMA accumulator folds A_f in one op, +12%).
    ("hopper", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),  # small CTAs pack better on WGMMA
        C=_PhaseCfg(nw=8, BS=32, ns=1),
    ),
    ("hopper", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),  # fp32 operand 2× bigger → half BS
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=1
        ),  # 2026-04-29 retune: acc_fusion=False is 3x faster post precision-gate fix
        C=_PhaseCfg(nw=16, BS=32, ns=1),  # 2026-04-30 retune: nw=16 BS=32 is 1.7x faster (was nw=8 BS=16)
    ),
    # Blackwell-DC (B200 / GB200): 228 KB SRAM + improved WGMMA codegen.
    # bf16 likes small CTAs (nw=4); fp32 stays at nw=8 (nw=4 + BS=64 fp32 = 92× regression).
    ("blackwell_dc", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=4, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=64, ns=1),  # 228 KB SRAM leaves room for BS=64 bf16
    ),
    ("blackwell_dc", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(
            nw=8, BS=128
        ),  # 2026-04-30 retune: nw=8 BS=128 ~5% faster at production F=3-6 (sweep across F=3,5,6,11)
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=3
        ),  # 2026-04-29 retune: 14x faster (was nw=8 acc=True 17ms; now nw=32 ns=3 acc=False 1.23ms)
        C=_PhaseCfg(
            nw=4, BS=64, ns=1
        ),  # 2026-04-30 retune: nw=4 BS=64 is 3-5x faster than old nw=8 BS=16 (sweep 2026-04-30)
    ),
    # Blackwell-Spark (5090 / GB10, ~102 KB SRAM): shares SRAM penalty of small
    # chips but not Blackwell-DC's WGMMA-v2 register-spread benefit. Empirically
    # behaves like Hopper at fp32 (Phase B wants nw=32 to spread persistent M
    # across warps, not nw=8 like DC). BS shrunk one step vs DC; Phase A bf16
    # wants nw=8 (nw=4 tested 22× slower per 2026-04-20 sweep).
    # Sweep 2026-04-24 (prod dim F=11 S=920): Phase B nw=32 gives 1.84×/2.65×
    # (GB10/5090) at fp32 over prior nw=8 setting.
    ("blackwell_spark", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),  # nw=8 (not 4) at bf16: ~5% across F=3,6,11
        # 2026-05-06 P1/P2 retune (5090, F=11 S=920): C.nw=4 BS=32 is ~3.5%
        # faster than nw=8 (Phase C is bandwidth-bound, fewer warps schedules
        # better on the small SRAM). BS=64 bf16 on Spark OOMs SRAM.
        C=_PhaseCfg(nw=4, BS=32, ns=1),
    ),
    ("blackwell_spark", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=16),  # fp32 operand 2× bigger → BS=16 (half of DC's 32)
        # 2026-05-06 retune: nw=16 OOMs the 102 KB SRAM cap at TF32 on 5090
        # (131 KB needed). nw=8 fits and is within noise of the prior nw=16
        # benchmark. The Phase B D-tile path (auto-enabled on spark, see
        # `_pick_phase_b_d_splits`) is ~2.6× faster than this baseline at TF32
        # and ~13% faster at IEEE — these baseline params only apply when
        # PHASE_B_D_SPLITS=1 is forced.
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=16, ns=1),  # binding constraint: M.fp32 64 KB + Q stage
    ),
}


# ──────────────────────────────────────────────────────────────────
# Shape-aware override table: empty by default. Keyed by
#     (arch_key, prec_key, shape_hint)
# where shape_hint is a free-form string (e.g. "small_BH", "large_F",
# "B>=8") chosen when populating. Lookup is exact-match; values are
# full `_ChunkwiseCfg` instances (no partial overrides — copy-paste
# from `_CHUNKWISE_TUNING` and edit the one phase you want to change).
#
# Leave empty unless a targeted sweep shows a particular shape regresses
# with the broad arch config. Adding here is strictly additive — base
# table remains the fallback.
# ──────────────────────────────────────────────────────────────────
_CHUNKWISE_SHAPE_OVERRIDES: dict[tuple[str, str, str], _ChunkwiseCfg] = {}


# Per-(cap, dot_prec) exact overrides (pins a specific GPU model if the arch
# bucket is wrong for it). Also empty by default.
_ARCH_OVERRIDES: dict = {}


def _arch_key(cap: tuple) -> str:
    """Map compute capability → named arch bucket in `_CHUNKWISE_TUNING`.

    Blackwell (cap[0] >= 10) is split into "blackwell_dc" and "blackwell_spark"
    by SRAM size (≥150 KB vs less). Without CUDA or for unknown archs we
    default to the conservative "ampere" bucket.
    """
    if cap[0] == 8:
        return "ampere"
    if cap[0] == 9:
        return "hopper"
    if cap[0] >= 10:
        has_big_sram = True
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
            has_big_sram = smem >= 150 * 1024
        return "blackwell_dc" if has_big_sram else "blackwell_spark"
    return "ampere"


def _prec_key(dot_prec: int) -> str:
    return "fp32" if dot_prec >= 1 else "bf16"


def _auto_config(dot_prec: int, cap: tuple, shape_hint: str | None = None) -> tuple:
    """Look up chunkwise kernel launch params from the tuning table.

    Resolution order:
      1. `_ARCH_OVERRIDES[(cap, dot_prec)]` — exact-capability pin, highest priority.
      2. `_CHUNKWISE_SHAPE_OVERRIDES[(arch, prec, shape_hint)]` — sweep-driven overrides.
      3. `_CHUNKWISE_TUNING[(arch, prec)]` — primary per-(arch, prec) table.
      4. Fallback to ("ampere", prec) if the arch is unrecognised.

    Returns the legacy 8-tuple `(a_nw, a_BS, b_nw, b_ns, b_use_acc, c_nw, c_BS, c_ns)`
    for backward compatibility with `_get_arch_config` callers.
    """
    arch = _arch_key(cap)
    prec = _prec_key(dot_prec)

    if shape_hint is not None:
        cfg = _CHUNKWISE_SHAPE_OVERRIDES.get((arch, prec, shape_hint))
        if cfg is not None:
            return cfg.as_tuple()

    cfg = _CHUNKWISE_TUNING.get((arch, prec)) or _CHUNKWISE_TUNING[("ampere", prec)]
    return cfg.as_tuple()


def _get_arch_config(
    dot_precision: int = 0,
    shape_hint: str | None = None,
    device: torch.device | int | None = None,
):
    """Returns (a_warps, a_BLOCK_S, b_warps, b_stages, b_use_acc_fusion,
                c_warps, c_BLOCK_S, c_stages).

    dot_precision: 0=bf16 TC, 1=TF32 TC, 2=IEEE fp32.
    shape_hint:    optional string key for `_CHUNKWISE_SHAPE_OVERRIDES`.
    device:        device whose capability drives the lookup. Defaults to the
                   current CUDA device — pass ``qkv.device`` (or any input
                   tensor's device) when launching kernels in heterogeneous
                   or multi-GPU single-process setups so the right tuning
                   bucket is chosen.
    """
    if not torch.cuda.is_available():
        cap = (9, 0)  # assume modern when querying from CPU
    else:
        if device is None:
            dev_idx = torch.cuda.current_device()
        elif isinstance(device, int):
            dev_idx = device
        else:
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(dev_idx)
    key = (cap, dot_precision)
    if key in _ARCH_OVERRIDES:
        return _ARCH_OVERRIDES[key]
    return _auto_config(dot_precision, cap, shape_hint)


# ════════════════════════════════════════════════════════════════
#  Phase A — split into KV and Z kernels
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_a_kv_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    I_minus_P_kv_ptr,  # output: (I - K_rot^T diag(β) K_rot)
    A_ptr,  # output: K_rot^T diag(β) V
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SKIP_RELU: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_kv_bhf = I_minus_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    A_bhf = A_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    k_nw_pair = tl.load(k_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)

    # KV stream accumulators (in-loop fp32 to avoid bf16 round-off compounding)
    P_kv_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    A_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        k_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d[None, :] * stride_d
        v_ptrs = qkv_bh + n_idx[:, None] * stride_n + 2 * stride_3 + offs_d[None, :] * stride_d
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        V_raw = tl.load(v_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        if SKIP_RELU:
            K = K_normed * k_scale
        else:
            K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

        K_pair_raw = tl.reshape(
            tl.flip(tl.reshape(K_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )
        K_pair_normed = K_pair_raw * k_inv_rms[:, None] * k_nw_pair[None, :]
        if SKIP_RELU:
            K_pair = K_pair_normed * k_scale
        else:
            K_pair = tl.where(K_pair_normed > 0, K_pair_normed, 0.0) * k_scale

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        K_rot = K * Cos + K_pair * Sin

        beta_Krot = beta_t[:, None] * K_rot
        beta_V = beta_t[:, None] * V_raw

        K_rot_T = tl.trans(K_rot)
        P_kv_acc += tl.dot(
            K_rot_T.to(dot_dtype), beta_Krot.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip
        )
        A_acc += tl.dot(K_rot_T.to(dot_dtype), beta_V.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)

    # Store bf16 outputs. Padded positions are 0 by construction (K_rot is 0 outside D).
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    I_minus_P_kv = tl.where(diag_in_range, 1.0 - P_kv_acc, -P_kv_acc)
    if DOT_PRECISION >= 1:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv)
        tl.store(A_bhf + offs_dd, A_acc)
    else:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv.to(tl.bfloat16))
        tl.store(A_bhf + offs_dd, A_acc.to(tl.bfloat16))


@triton.jit
def _phase_a_z_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    I_minus_P_z_ptr,  # output: (I - K^T diag(β) K)
    B_ptr,  # output: K^T β
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Z stream: uses K (no RoPE). Cheaper than KV — no V load, no RoPE compute,
    no K_pair derivation."""
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_z_bhf = I_minus_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    B_bhf = B_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    P_z_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    B_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        # Only K_raw needed (no V, no Cos/Sin)
        k_ptrs = qkv_bh + n_idx[:, None] * stride_n + 1 * stride_3 + offs_d[None, :] * stride_d
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

        beta_K = beta_t[:, None] * K

        K_T = tl.trans(K)
        P_z_acc += tl.dot(K_T.to(dot_dtype), beta_K.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
        B_acc += tl.sum(beta_K, axis=0)

    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    I_minus_P_z = tl.where(diag_in_range, 1.0 - P_z_acc, -P_z_acc)

    if DOT_PRECISION >= 1:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z)
    else:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z.to(tl.bfloat16))
    # B stays fp32 (vector, only 0.5 KB, negligible HBM cost)
    tl.store(B_bhf + offs_d, B_acc)


def phase_a(
    qkv: torch.Tensor,
    beta: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    F: int,
    S: int,
    k_scale: float = 1.0,
    norm_eps: float = 1e-5,
    num_warps: int | None = None,
    num_stages: int = 1,
    BLOCK_S: int | None = None,
    dot_precision: int = 0,
    skip_relu: bool = False,
    skip_z: bool = False,
):
    """Compute (I-P_kv), A, (I-P_z), B for all (B, H, F) via 2 kernels (KV + Z).

    `skip_relu=True` makes the K-stream prep a pure linear chain (no ReLU on
    K_normed * k_scale). Used by the camera-branch chunkwise wrapper, where K
    has already been ReLU'd by the cam_prep kernel and subsequently rotated
    by UCPE+RoPE — re-applying ReLU on the rotated values would clobber
    legitimate negatives.

    `skip_z=True` skips the Phase A Z kernel entirely and returns placeholder
    tensors for I_P_z and B_z. Used by NUM_ONLY callers (camera branch) to
    avoid wasted Z-stream prep when the denominator scan won't be used.
    """
    # Auto-pick (num_warps, BLOCK_S) per arch+precision unless overridden
    if num_warps is None or BLOCK_S is None:
        a_w, a_bs, *_ = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = a_w
        if BLOCK_S is None:
            BLOCK_S = a_bs
    B, N, three, H, D = qkv.shape
    assert three == 3 and N == F * S
    BLOCK_D = triton.next_power_of_2(D)
    BH = B * H

    # FAIR-COMPARE PATCH: keep fp32 inter-phase bridge at P0/P1 to match pytorch/fused
    bridge_dtype = torch.float32 if dot_precision >= 1 else torch.bfloat16
    I_P_kv = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)
    A = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)

    beta_c = beta.contiguous()
    grid = (BH * F,)

    _phase_a_kv_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        rope_cos,
        rope_sin,
        I_P_kv,
        A,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        SKIP_RELU=skip_relu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if skip_z:
        # NUM_ONLY callers (camera branch) do not consume the Z scan. Return
        # placeholders and let Phase B skip all Z loads/stores as well.
        I_P_z = torch.empty(1, device=qkv.device, dtype=bridge_dtype)
        B_z = torch.empty(1, device=qkv.device, dtype=torch.float32)
        return I_P_kv, A, I_P_z, B_z

    I_P_z = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)
    # B stays fp32 — small vector (0.5 KB/frame), no benefit to downcast
    B_z = torch.empty(BH, F, BLOCK_D, device=qkv.device, dtype=torch.float32)

    _phase_a_z_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        I_P_z,
        B_z,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return I_P_kv, A, I_P_z, B_z


# ════════════════════════════════════════════════════════════════
#  Phase B — serial scan, uses pre-stored (I - P) so MMA folds in M
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_b_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,  # (BH, BLOCK_D, BLOCK_D) — read when LOAD_INIT_STATE=1
    init_state_z_ptr,  # (BH, BLOCK_D)
    final_state_kv_ptr,  # (BH, BLOCK_D, BLOCK_D) — written when SAVE_FINAL_STATE=1
    final_state_z_ptr,  # (BH, BLOCK_D)
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,  # forward scan seeded with init state (vs zeros)
    SAVE_FINAL_STATE: tl.constexpr,  # write M_{F-1} of forward scan to final_state_*
    DIRECTION: tl.constexpr,  # 0=both, 1=fwd-only, 2=rev-only
    COMBINED_HISTORY: tl.constexpr,  # 1 → rev branch read-add-stores into M_fwd_ptr
    # (M_hist[f] = M_fwd[f] + M_rev[f]); skips the F-1 zero-write so the fwd
    # value at F-1 is preserved (rev contribution there is exactly zero anyway).
    # Only meaningful when DIRECTION=0. Saves one Phase C launch + one M-shaped
    # buffer downstream (Phase C runs once on M_hist instead of twice).
    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    bh = pid

    offs_d = tl.arange(0, BLOCK_D)
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]

    # ── Forward scan (skip when DIRECTION=2 i.e. rev-only) ──
    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd).to(tl.float32)
            if not SKIP_Z:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d).to(tl.float32)
        else:
            M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
            if not SKIP_Z:
                z = tl.zeros([BLOCK_D], dtype=tl.float32)
        for f in range(F):
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd)
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)

            # M = g · (I - P_kv) M + A_f
            if USE_ACC_FUSION:
                # Pre-scale (I-P) by g, accumulate A_f directly via the MMA accumulator.
                # Result: A_f + g·(I-P)·M  in one MMA — no separate M_temp tensor.
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            tl.store(M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d)
                # z = g · (I - P_z) z + B_f
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d, z)

        # Save terminal forward state for state-cached inference (autoregressive sampling).
        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d, z)

    # ── Reverse scan (skip when DIRECTION=1 i.e. fwd-only) ──
    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)
        # COMBINED_HISTORY mode: rev contributions get read-add-stored into the
        # fwd buffer (which thereby becomes M_hist = M_fwd + M_rev). The F-1
        # zero-write is skipped so M_hist[F-1] keeps the fwd value (rev value
        # there is zero by construction, so no add needed).
        if not COMBINED_HISTORY:
            tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + (F - 1) * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                tl.store(z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d, z)
        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd)
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            if not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:
                # Read-add-store into the fwd buffer. The fwd loop has already
                # written M_fwd[f_dst] to this slot; we add the rev contribution
                # in place. Stays in L1/L2 since fwd just touched it.
                M_addr = M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd
                tl.store(M_addr, tl.load(M_addr) + M)
                if not SKIP_Z:
                    z_addr = z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd, M)
                if not SKIP_Z:
                    tl.store(z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d, z)


def phase_b_triton(
    I_P_kv,
    A,
    I_P_z,
    B,
    decay,
    F,
    num_warps=None,
    num_stages=None,
    use_acc_fusion=None,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
    direction=0,
    combined_history=False,
    skip_z=False,
):
    """Phase B serial-F scan over (B*H,).

    Forward scan can be seeded with `init_state_kv`/`init_state_z` (autoregressive
    sampling chunk > 0) and can write the terminal `M_{F-1}`/`z_{F-1}` to caller-
    provided buffers when `return_final_state=True`.

    `direction`: 0=both (default), 1=forward-only, 2=reverse-only. Forward-only
    skips reverse scan + reverse output buffers; reverse-only skips forward scan
    + state load/save. Used by single-direction state-cached entry points.

    `combined_history` (only meaningful with direction=0): the rev branch
    read-add-stores into the fwd buffer so its contents become
    M_hist[f] = M_fwd[f] + M_rev[f] (and same for z). Lets the caller run
    Phase C exactly once on the combined history, since Phase C is linear in
    M and z (`Q @ (M_fwd + M_rev) = Q @ M_fwd + Q @ M_rev`). When set,
    M_rev/z_rev outputs are placeholder dummies; only M_fwd/z_fwd carry data.

    `skip_z`: skip the denominator/Z recurrence entirely. Used by camera
    numerator-only scans where Phase C runs with `num_only=True`.

    Returns (M_fwd, z_fwd, M_rev, z_rev) — and additionally (final_kv, final_z)
    when return_final_state=True. Skipped-direction outputs are returned as a
    1-element placeholder tensor (kernel never touches them when DIRECTION
    gates them off); callers should always discard the slot they didn't ask
    for. Reverse scan is always seeded with zeros (per upstream's bidi
    state-cache convention — only forward state is cached).
    """
    BH = I_P_kv.shape[0]
    _, _, BLOCK_D, _ = A.shape  # A is always full [BH, F, BLOCK_D, BLOCK_D]
    device, fdtype = I_P_kv.device, torch.float32

    if num_warps is None or num_stages is None or use_acc_fusion is None:
        _, _, b_w, b_s, b_acc, *_ = _get_arch_config(dot_precision, device=device)
        if num_warps is None:
            num_warps = b_w
        if num_stages is None:
            num_stages = b_s
        if use_acc_fusion is None:
            use_acc_fusion = b_acc

    if combined_history and direction != 0:
        raise ValueError("combined_history=True requires direction=0 (bidi)")

    # Phase B kernel is DIRECTION-gated (constexpr); skipped-direction writes
    # never happen, so we can hand it a 1-element placeholder for the inactive
    # buffers and free ~4× M_fwd-shaped allocations per single-direction call.
    decay_flat = decay.reshape(BH, F).contiguous().float()

    load_init = init_state_kv is not None
    dummy = torch.empty(1, device=device, dtype=fdtype)

    def full_M():
        return torch.empty(BH, F, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)

    def full_z():
        return torch.empty(BH, F, BLOCK_D, device=device, dtype=fdtype)

    M_fwd = dummy if direction == 2 else full_M()
    z_fwd = dummy if (direction == 2 or skip_z) else full_z()
    # Combined-history mode reuses M_fwd/z_fwd as M_hist/z_hist; rev outputs
    # become placeholders even though DIRECTION!=1.
    M_rev = dummy if (direction == 1 or combined_history) else full_M()
    z_rev = dummy if (direction == 1 or combined_history or skip_z) else full_z()
    if load_init:
        init_kv = init_state_kv.contiguous().view(BH, BLOCK_D, BLOCK_D)
        init_z = dummy if skip_z else init_state_z.contiguous().view(BH, BLOCK_D)
    else:
        init_kv = dummy
        init_z = dummy

    if return_final_state:
        final_kv = torch.empty(BH, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)
        final_z = dummy if skip_z else torch.empty(BH, BLOCK_D, device=device, dtype=fdtype)
    else:
        final_kv = dummy
        final_z = dummy

    d_splits, nw_override, ns_override, acc_override = _pick_phase_b_d_splits(BLOCK_D, dot_precision=dot_precision)
    if d_splits > 1:
        D_TILE = BLOCK_D // d_splits
        # Use D-tile-specific tuning if available, else fall back to baseline tuning
        nw_use = nw_override if nw_override is not None else num_warps
        ns_use = ns_override if ns_override is not None else num_stages
        acc_use = acc_override if acc_override is not None else use_acc_fusion
        _phase_b_dtile_kernel[(BH, d_splits)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            D_TILE=D_TILE,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=acc_use,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=nw_use,
            num_stages=ns_use,
        )
    else:
        _phase_b_kernel[(BH,)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=use_acc_fusion,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if return_final_state:
        return M_fwd, z_fwd, M_rev, z_rev, final_kv, final_z
    return M_fwd, z_fwd, M_rev, z_rev


# ════════════════════════════════════════════════════════════════
#  Phase B D-tile — j-axis split for grid parallelism (#118)
# ════════════════════════════════════════════════════════════════
# Same recurrence as _phase_b_kernel but each program owns a D_TILE-wide
# slice of M's output column dim. Grid: (BH, d_splits). M_new[*, j_tile]
# only depends on M_prev[*, j_tile] and full (I-P_kv) — independent across
# j-tiles. z is unsplittable; only `pid_d == 0` updates/writes z.
@triton.jit
def _phase_b_dtile_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,
    init_state_z_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,
    SAVE_FINAL_STATE: tl.constexpr,
    DIRECTION: tl.constexpr,
    COMBINED_HISTORY: tl.constexpr,
    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = pid_bh

    offs_d_full = tl.arange(0, BLOCK_D)
    offs_d_tile = pid_d * D_TILE + tl.arange(0, D_TILE)
    offs_dd_full = offs_d_full[:, None] * BLOCK_D + offs_d_full[None, :]
    offs_dd_tile = offs_d_full[:, None] * BLOCK_D + offs_d_tile[None, :]

    is_lead = pid_d == 0

    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile).to(tl.float32)
        else:
            M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)
            if is_lead and LOAD_INIT_STATE:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d_full).to(tl.float32)

        for f in range(F):
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_full)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_tile)
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            tl.store(M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_tile, M)

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd_full)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full, z)

        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile, M)
            if is_lead and not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d_full, z)

    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)

        if not COMBINED_HISTORY:
            tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + (F - 1) * BLOCK_D * BLOCK_D + offs_dd_tile, M)
            if is_lead and not SKIP_Z:
                tl.store(z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d_full, z)

        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(I_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd_full)
            A_f = tl.load(A_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd_tile)
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(I_P_kv_f.to(dot_dtype), M.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
                M = g_f * M_temp + A_f

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(I_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd_full)
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:
                M_addr = M_fwd_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd_tile
                tl.store(M_addr, tl.load(M_addr) + M)
                if is_lead and not SKIP_Z:
                    z_addr = z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(M_rev_ptr + bh * F * BLOCK_D * BLOCK_D + f_dst * BLOCK_D * BLOCK_D + offs_dd_tile, M)
                if is_lead and not SKIP_Z:
                    tl.store(z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full, z)


_PHASE_B_DTILE_ARCH_CACHE: dict = {}  # (dev, dot_prec) -> (d_splits, nw, ns, acc)


# Per-arch D-tile optimum from 2026-04-29 sweep (T=11 B=1 P0 IEEE):
#   WGMMA-server (A100 sm_80, H100 sm_90): (d=4, nw=32, ns=1, acc=True)
#   Blackwell-family (GB200 sm_100, 5090 sm_120, GB10 sm_121, Ada sm_89):
#                                          (d=8, nw=4,  ns=1, acc=False)
# Both clusters were tested across 96 configs (4 ds × 4 nw × 3 ns × 2 acc).
def _pick_phase_b_d_splits(BLOCK_D: int, dot_precision: int = 0):
    """Returns (d_splits, nw_override, ns_override, acc_override).

    `d_splits=1` → use baseline `_phase_b_kernel` with `_CHUNKWISE_TUNING` config.
    `d_splits>1` → use `_phase_b_dtile_kernel` with overrides for nw/ns/acc.
    Override via env: PHASE_B_D_SPLITS, PHASE_B_DTILE_NW, PHASE_B_DTILE_NS,
    PHASE_B_DTILE_ACC (1=True / 0=False).
    """
    import os

    env_d = os.environ.get("PHASE_B_D_SPLITS", None)
    if env_d is not None:
        d = int(env_d)
        if d < 1 or BLOCK_D % d != 0:
            return (1, None, None, None)
        nw = int(os.environ.get("PHASE_B_DTILE_NW", "0")) or None
        ns = int(os.environ.get("PHASE_B_DTILE_NS", "0")) or None
        acc_env = os.environ.get("PHASE_B_DTILE_ACC", None)
        acc = bool(int(acc_env)) if acc_env is not None else None
        return (d, nw, ns, acc)
    try:
        import torch

        if not torch.cuda.is_available():
            return (1, None, None, None)
        dev = torch.cuda.current_device()
        cache_key = (dev, dot_precision)
        if cache_key not in _PHASE_B_DTILE_ARCH_CACHE:
            cap = torch.cuda.get_device_capability(dev)
            major, minor = cap[0], cap[1]
            if dot_precision == 2:
                # IEEE fp32: D-tile dominates baseline on every arch (96-config sweep).
                if major == 8 and minor == 0:
                    cfg = (4, 32, 1, True)  # A100
                elif major == 9:
                    cfg = (4, 32, 1, True)  # H100 (Hopper)
                elif major == 8 and minor == 9:
                    cfg = (8, 4, 1, False)  # Ada (assume Blackwell-like)
                elif major >= 10:
                    cfg = (8, 4, 1, False)  # GB200/B200, 5090, GB10
                else:
                    cfg = (1, None, None, None)  # unknown — baseline
            else:
                # bf16/TF32: cap-specific dispatch. Multi-arch sweep 2026-05-06
                # (F=11 S=920) determined per-cap whether D-tile beats the
                # baseline _phase_b_kernel:
                #   sm_80  A100:   D-tile WIN 1.09× (P1) / 1.02× (P2) — (4,8,2,F).
                #   sm_90  H100:   D-tile WIN ~10% — P1 (4,8,2,F); P2 (8,8,2,F).
                #                  Use (4,8,2,F) for both (P2 within 0.4%).
                #   sm_100 GB200:  D-tile WIN ~12% — (4,8,2,F) both precisions.
                #   sm_120 5090:   D-tile WIN 2.6× (P1) / 1.13× (P2) — (8,8,1,F).
                #                  TF32 baseline OOMs at 102 KB SRAM cap.
                #   sm_121 GB10:   D-tile LOSS 4% — baseline wins. Despite same
                #                  reported SRAM/SM as sm_120, the baseline
                #                  kernel fits all configs up to nw=16 ns=2 on
                #                  sm_121 (Triton/codegen difference between
                #                  consumer-Blackwell variants), so baseline
                #                  saturates the chip without needing D-tile.
                if major == 8 and minor == 0:
                    cfg = (4, 8, 2, False)  # A100
                elif major == 9:
                    cfg = (4, 8, 2, False)  # H100
                elif major == 10:
                    cfg = (4, 8, 2, False)  # GB200 / B200
                elif major == 12 and minor == 0:
                    cfg = (8, 8, 1, False)  # 5090
                elif major == 12 and minor == 1:
                    cfg = (1, None, None, None)  # GB10 — baseline wins
                else:
                    cfg = (1, None, None, None)  # Ada, unknown
            _PHASE_B_DTILE_ARCH_CACHE[cache_key] = cfg
        return _PHASE_B_DTILE_ARCH_CACHE[cache_key]
    except Exception:
        return (1, None, None, None)


# ════════════════════════════════════════════════════════════════
#  Phase C — Pass 2 output (per (B, H, F)). Same as v1.
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_c_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    q_inv_rms_ptr,
    q_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    M_ptr,
    z_ptr,
    num_ptr,
    den_ptr,
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ACCUMULATE: tl.constexpr = False,
    SKIP_LAST_F: tl.constexpr = False,
    SKIP_RELU: tl.constexpr = False,
    NUM_ONLY: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    # Reverse-accumulate callers pass SKIP_LAST_F=True: M_rev[F-1] / z_rev[F-1]
    # are exactly zero (Phase B initializes the reverse scan with zeros and the
    # write loop only fills f<F-1), so the f=F-1 program would only re-write
    # the forward pass's output unchanged. Early-return saves one frame's worth
    # of Q+RoPE HBM reads + dot products per (B, H).
    if SKIP_LAST_F and pid_f == F - 1:
        return

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    num_bh = num_ptr + pid_b * (N * H * D) + pid_h * D
    den_bh = den_ptr + bh * N
    M_bhf = M_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    z_bhf = z_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    nw_offset = pid_h * D
    q_nw = tl.load(q_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    q_nw_pair = tl.load(q_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0).to(tl.float32)

    M_f = tl.load(M_bhf + offs_dd, mask=mask_dd, other=0.0)
    z_f = tl.load(z_bhf + offs_d, mask=mask_d, other=0.0)

    n_base = pid_f * S
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        q_ptrs = qkv_bh + n_idx[:, None] * stride_n + 0 * stride_3 + offs_d[None, :] * stride_d
        Q_raw = tl.load(q_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_pair_raw = tl.reshape(
            tl.flip(tl.reshape(Q_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )

        q_inv_rms = tl.load(q_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0).to(tl.float32)
        Q_normed = Q_raw * q_inv_rms[:, None] * q_nw[None, :]
        Q_pair_normed = Q_pair_raw * q_inv_rms[:, None] * q_nw_pair[None, :]
        if SKIP_RELU:
            Q = Q_normed
            Q_pair = Q_pair_normed
        else:
            Q = tl.where(Q_normed > 0, Q_normed, 0.0)
            Q_pair = tl.where(Q_pair_normed > 0, Q_pair_normed, 0.0)

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_rot = Q * Cos + Q_pair * Sin

        num = tl.dot(Q_rot.to(dot_dtype), M_f.to(dot_dtype), out_dtype=tl.float32, input_precision=dot_ip)
        if not NUM_ONLY:
            den = tl.sum(Q * z_f[None, :], axis=1)

        num_ptrs = num_bh + n_idx[:, None] * (H * D) + offs_d[None, :]
        if not NUM_ONLY:
            den_ptrs = den_bh + n_idx
        if ACCUMULATE:
            # Used by reverse-direction Phase C: add this pass onto forward's
            # already-written buffer instead of allocating a separate one.
            prev_num = tl.load(num_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            num = num + prev_num
            if not NUM_ONLY:
                prev_den = tl.load(den_ptrs, mask=mask_s, other=0.0).to(tl.float32)
                den = den + prev_den
        if DOT_PRECISION >= 1:
            tl.store(num_ptrs, num, mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den, mask=mask_s)
        else:
            tl.store(num_ptrs, num.to(tl.bfloat16), mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den.to(tl.bfloat16), mask=mask_s)


def phase_c(
    qkv,
    q_inv_rms,
    q_norm_w,
    rope_cos,
    rope_sin,
    M,
    z,
    F,
    S,
    num_warps=None,
    num_stages=None,
    BLOCK_S=None,
    dot_precision=0,
    num_out=None,
    den_out=None,
    accumulate=False,
    skip_last_frame=False,
    skip_relu: bool = False,
    num_only: bool = False,
):
    """Phase C Pass-2 output. Optionally accumulates into caller-provided
    ``num_out``/``den_out`` buffers (used to fuse reverse-direction output into
    forward-direction buffer without allocating a separate one — saves ~45 MB
    at B=1 bf16, ~180 MB at B=4).

    ``skip_last_frame=True`` early-returns the f=F-1 programs. Valid for the
    reverse-accumulate call only, where M[F-1]/z[F-1] are guaranteed zero.

    ``skip_relu=True`` matches Phase A KV's flag — used by the camera-branch
    chunkwise wrapper where Q has already been ReLU'd by cam_prep before
    being rotated by UCPE+RoPE; re-applying ReLU on the rotated Q would
    clobber legitimate negatives.

    ``num_only=True`` skips the denominator computation and store entirely
    (kernel writes only ``num_out``; ``den_out`` is allowed to be None /
    unallocated). Used by the camera-branch which has no Z scan.
    """
    if num_warps is None or num_stages is None or BLOCK_S is None:
        *_, c_w, c_bs, c_s = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = c_w
        if num_stages is None:
            num_stages = c_s
        if BLOCK_S is None:
            BLOCK_S = c_bs
    B, N, three, H, D = qkv.shape
    BLOCK_D = triton.next_power_of_2(D)
    if num_out is None:
        num_out = torch.empty(
            B, N, H, D, device=qkv.device, dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16)
        )
    if den_out is None and not num_only:
        den_out = torch.empty(
            B, H, N, device=qkv.device, dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16)
        )
    elif num_only and den_out is None:
        # Pass a 1-element placeholder; kernel guards den loads/stores under NUM_ONLY.
        den_out = torch.empty(1, device=qkv.device, dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16))

    _phase_c_kernel[(B * H * F,)](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M,
        z,
        num_out,
        den_out,
        H=H,
        F=F,
        S=S,
        D=D,
        NORM_EPS=1e-5,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        ACCUMULATE=1 if accumulate else 0,
        SKIP_LAST_F=skip_last_frame,
        SKIP_RELU=skip_relu,
        NUM_ONLY=num_only,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return num_out, den_out


def fused_bigdn_bidi_chunkwise(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_w,
    k_norm_w,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale=1.0,
    eps=1e-6,
    norm_eps=1e-5,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
):
    """Bidi chunkwise GDN forward, optionally with state-cache for autoregressive
    sampling (chunk 0 = full bidi with state save; chunks > 0 seed forward scan
    from saved state). Reverse always seeds from zero per upstream convention.

    Pipeline (2026-04-25 restructure): Phase A once → Phase B direction=0 with
    combined_history=True (fwd seeded with init_state and saves final state;
    rev zero-seeded; rev output summed into fwd buffer in-kernel via read-
    add-store so on exit M_hist[f] = M_fwd[f] + M_rev[f]) → Phase C ONCE on
    M_hist. Phase C linearity `Q @ (M_fwd + M_rev) = Q @ M_fwd + Q @ M_rev`
    makes the in-kernel sum exact.

    Replaces the prior 2× Phase B + 2× Phase C pattern. Saves one Phase C
    launch + one Q+RoPE HBM pass and one M-shape buffer per call.
    """
    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_w,
        k_norm_w,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
    )

    if return_final_state:
        M_hist, z_hist, _, _, final_kv, final_z = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            return_final_state=True,
            combined_history=True,
        )
    else:
        M_hist, z_hist, _, _ = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            combined_history=True,
        )
    num_out, den_out = phase_c(
        qkv,
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        accumulate=False,
    )
    del M_hist, z_hist, I_P_kv, A, I_P_z, B_z

    # ── Final divide ──
    total_den = den_out.float().permute(0, 2, 1).unsqueeze(-1)  # (B, N, H, 1)
    out = (num_out.float() / (total_den + eps)).to(qkv.dtype)
    del num_out, den_out, total_den
    if return_final_state:
        B = qkv.shape[0]
        H = qkv.shape[3]
        D = qkv.shape[4]
        BLOCK_D = final_kv.shape[1]
        state_kv = final_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
        state_z = final_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        return out, state_kv, state_z
    return out


def _default_dot_prec():
    """Pull dot_precision from `_resolve_launch_config` (honors PRECISION_OVERRIDE)."""

    _, dot_prec, _, _ = _resolve_launch_config()
    return dot_prec


def fused_gdn_func_chunkwise(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale,
    eps=1e-6,
    reverse=False,
    dot_precision=None,
):
    """Single-direction chunkwise GDN — drop-in for `fused_gdn.fused_gdn_func`.

    Computes only one scan direction (Phase B + Phase C × 1) and returns
    `(num, den)` shape-compatible with the upstream function. dot_precision
    defaults to whatever `_resolve_launch_config` returns (honors module-level
    `PRECISION_OVERRIDE`).
    """
    if dot_precision is None:
        dot_precision = _default_dot_prec()
    direction = 2 if reverse else 1
    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        dot_precision=dot_precision,
    )
    M_fwd, z_fwd, M_rev, z_rev = phase_b_triton(
        I_P_kv,
        A,
        I_P_z,
        B_z,
        decay,
        F=F,
        dot_precision=dot_precision,
        direction=direction,
    )
    M_use = M_rev if reverse else M_fwd
    z_use = z_rev if reverse else z_fwd
    num, den = phase_c(
        qkv, q_inv_rms, q_norm_weight, rope_cos, rope_sin, M_use, z_use, F=F, S=S, dot_precision=dot_precision
    )
    return num, den


def fused_gdn_stateful_chunkwise(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale,
    eps=1e-6,
    reverse=False,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
    dot_precision=None,
):
    """Single-direction chunkwise GDN with optional state cache — drop-in for
    `fused_gdn.fused_gdn_stateful`. Forward direction supports state load/save
    (used for autoregressive sampling); reverse direction always runs fresh
    (per upstream's bidi state-cache convention).
    """
    if dot_precision is None:
        dot_precision = _default_dot_prec()
    direction = 2 if reverse else 1
    if reverse and (init_state_kv is not None or return_final_state):
        raise ValueError(
            "fused_gdn_stateful_chunkwise: state cache is forward-only (matching "
            "upstream's bidi convention); pass reverse=False or omit state args."
        )
    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        dot_precision=dot_precision,
    )
    # Pad caller-supplied state from (B,H,D,D)/(B,H,D,1) to (BH, BLOCK_D, BLOCK_D)/(BH, BLOCK_D).
    # Needed because the state returned by this function is unpadded (B,H,D,D),
    # but phase_b_triton's kernel expects the padded layout.
    init_kv_padded, init_z_padded = init_state_kv, init_state_z
    if init_state_kv is not None:
        B_, H_, D_in, D_out = init_state_kv.shape
        BLOCK_D_ = I_P_kv.shape[-1]
        if D_in != BLOCK_D_ or D_out != BLOCK_D_:
            pad_in = BLOCK_D_ - D_in
            pad_out = BLOCK_D_ - D_out
            init_kv_padded = torch.nn.functional.pad(
                init_state_kv.transpose(-1, -2).reshape(B_ * H_, D_out, D_in), (0, pad_in, 0, pad_out)
            ).contiguous()
        else:
            init_kv_padded = init_state_kv.transpose(-1, -2).reshape(B_ * H_, BLOCK_D_, BLOCK_D_).contiguous()
        # z: (B, H, D) or (B, H, D, 1) → (BH, BLOCK_D)
        z_ = init_state_z.squeeze(-1) if init_state_z.dim() == 4 else init_state_z
        Bz_, Hz_, Dz_ = z_.shape
        if Dz_ != BLOCK_D_:
            init_z_padded = torch.nn.functional.pad(z_.reshape(Bz_ * Hz_, Dz_), (0, BLOCK_D_ - Dz_)).contiguous()
        else:
            init_z_padded = z_.reshape(Bz_ * Hz_, Dz_).contiguous()
    if return_final_state:
        M_fwd, z_fwd, M_rev, z_rev, final_kv, final_z = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=direction,
            init_state_kv=init_kv_padded,
            init_state_z=init_z_padded,
            return_final_state=True,
        )
    else:
        M_fwd, z_fwd, M_rev, z_rev = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=direction,
            init_state_kv=init_kv_padded,
            init_state_z=init_z_padded,
        )
    M_use = M_rev if reverse else M_fwd
    z_use = z_rev if reverse else z_fwd
    num, den = phase_c(
        qkv, q_inv_rms, q_norm_weight, rope_cos, rope_sin, M_use, z_use, F=F, S=S, dot_precision=dot_precision
    )
    if return_final_state:
        B = qkv.shape[0]
        H = qkv.shape[3]
        D = qkv.shape[4]
        BLOCK_D = final_kv.shape[1]
        state_kv = final_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
        state_z = final_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        return num, den, state_kv, state_z
    return num, den


def fused_bidi_stateful_chunkwise_shared_phase_a(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale,
    eps=1e-6,
    init_state_kv=None,
    init_state_z=None,
    dot_precision=None,
):
    """Bidi state-cached chunkwise GDN with shared Phase A and combined-history
    Phase B. Default chunkwise path for ``_fused_statecached_forward``.

    Pipeline (per layer per step):
      1. Phase A once over qkv  — K/V/RoPE pre-norm; was previously duplicated
         across two streams.
      2. Phase B with direction=0 + combined_history=True — single program does
         fwd then rev; fwd writes M_hist; rev read-add-stores into the same
         buffer so on exit M_hist[f] = M_fwd[f] + M_rev[f] (same for z).
         Forward branch loads init_state and saves final state.
      3. Phase C ONCE on M_hist/z_hist  — Phase C is linear in M/z so
         `Q @ (M_fwd + M_rev) = Q @ M_fwd + Q @ M_rev`.

    Returns ``(num_combined, den_combined, state_kv, state_z)`` — caller hands
    the num/den pair to ``fused_bidi_merge(num, None, den, None, eps, gate)``
    in PRE_SUMMED mode.

    HBM-traffic delta vs the prior 2× Phase C version (per call, B=1 prod):
      saved : 1× Phase C Q+RoPE pass (~90 MB)
      saved : one (B,N,H,D) num and (B,H,N) den allocation
      cost  : Phase B rev does read-add of M_hist (~14 MB extra per layer)
      net   : ~76 MB saved + 1 fewer kernel launch

    Measured speed on GB10 (sm_121) at H=20, S=920, D=112, vs the prior
    shared-Phase-A-with-2×-Phase-C path, across production F values:
      P0 IEEE fp32     : 1.26-1.42× (F=3,6,11; B=1,2)
      P2 bf16+fp32-st  : 1.57-1.80×
      P3 bf16+bf16-st  : 1.63-1.96×
    Correctness cos ≥ 0.999997 across all cells, state_kv exact.
    """
    if dot_precision is None:
        dot_precision = _default_dot_prec()

    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        dot_precision=dot_precision,
    )

    init_kv_padded, init_z_padded = init_state_kv, init_state_z
    if init_state_kv is not None:
        B_, H_, D_in, D_out = init_state_kv.shape
        BLOCK_D_ = I_P_kv.shape[-1]
        if D_in != BLOCK_D_ or D_out != BLOCK_D_:
            pad_in = BLOCK_D_ - D_in
            pad_out = BLOCK_D_ - D_out
            init_kv_padded = torch.nn.functional.pad(
                init_state_kv.transpose(-1, -2).reshape(B_ * H_, D_out, D_in), (0, pad_in, 0, pad_out)
            ).contiguous()
        else:
            init_kv_padded = init_state_kv.transpose(-1, -2).reshape(B_ * H_, BLOCK_D_, BLOCK_D_).contiguous()
        z_ = init_state_z.squeeze(-1) if init_state_z.dim() == 4 else init_state_z
        Bz_, Hz_, Dz_ = z_.shape
        if Dz_ != BLOCK_D_:
            init_z_padded = torch.nn.functional.pad(z_.reshape(Bz_ * Hz_, Dz_), (0, BLOCK_D_ - Dz_)).contiguous()
        else:
            init_z_padded = z_.reshape(Bz_ * Hz_, Dz_).contiguous()

    # combined_history=True routes the rev contribution into the fwd buffer →
    # M_hist[f] = M_fwd[f] + M_rev[f]. M_rev/z_rev outputs are placeholders.
    M_hist, z_hist, _, _, final_kv, final_z = phase_b_triton(
        I_P_kv,
        A,
        I_P_z,
        B_z,
        decay,
        F=F,
        dot_precision=dot_precision,
        direction=0,
        init_state_kv=init_kv_padded,
        init_state_z=init_z_padded,
        return_final_state=True,
        combined_history=True,
    )

    num, den = phase_c(
        qkv, q_inv_rms, q_norm_weight, rope_cos, rope_sin, M_hist, z_hist, F=F, S=S, dot_precision=dot_precision
    )

    B = qkv.shape[0]
    H = qkv.shape[3]
    D = qkv.shape[4]
    BLOCK_D = final_kv.shape[1]
    state_kv = final_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
    state_z = final_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
    return num, den, state_kv, state_z


def fused_bigdn_stateful_chunkwise(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale,
    eps=1e-6,
    return_final_state=False,
    dot_precision=None,
):
    """Drop-in replacement for `fused_gdn.fused_bigdn_stateful` using the
    chunkwise pipeline. Same signature, same return shape:
      output (B, N, H, D), and if return_final_state: + (state_kv, state_z).
    dot_precision defaults to whatever `_resolve_launch_config` returns.
    """
    if dot_precision is None:
        dot_precision = _default_dot_prec()
    if return_final_state:
        out, state_kv, state_z = fused_bigdn_bidi_chunkwise(
            qkv,
            q_inv_rms,
            k_inv_rms,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            beta,
            decay,
            F=F,
            S=S,
            k_scale=k_scale,
            eps=eps,
            dot_precision=dot_precision,
            return_final_state=True,
        )
        return out, state_kv, state_z
    out = fused_bigdn_bidi_chunkwise(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
        dot_precision=dot_precision,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Camera-branch wrapper — numerator-only single-path delta-rule scan via
#  chunkwise. Drop-in for `diffusion.model.ops.fused_cam_gdn.cam_scan_func`.
#
#  Cam math expanded:
#      state = state * g                               # apply decay
#      state += K^T @ ((V - K @ state) * β)            # delta-rule
#  Equivalently:
#      state_new = g (I - K^T β K) state_old + K^T β V
#                = g (I - P_kv) state_old + A
#  This is bit-identical to chunkwise's Phase B M update, so the scan kernel
#  is reusable. The only differences from main GDN:
#    1. Q/K/V come pre-prepped (cam_prep_kernel did RMSNorm+ReLU+UCPE+RoPE).
#       We disable chunkwise's prep with identity tables (k_inv_rms=1, k_nw=1,
#       k_scale=1, rope_cos=1, rope_sin=0) AND skip_relu=True (because cam
#       applied ReLU BEFORE UCPE; the post-UCPE values can have legitimate
#       negatives that re-applying ReLU would clobber).
#    2. No Z denominator scan; output is num-only (out = Q @ M, no /Z).
#       skip_z=True elides Phase A Z; num_only=True elides Phase C den compute.
# ─────────────────────────────────────────────────────────────────────────────
def _cam_identity_tables(
    *,
    B: int,
    N: int,
    H: int,
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached identity RMS/RoPE tables used by ``cam_scan_chunkwise``."""
    device_index = device.index if device.type == "cuda" else None
    key = (device.type, device_index, B, N, H * D, D)
    cached = _CAM_IDENTITY_CACHE.get(key)
    if cached is not None:
        return cached

    ones_inv_rms = torch.ones(B, N, device=device, dtype=torch.float32)
    ones_nw = torch.ones(H * D, device=device, dtype=torch.float32)
    ones_cos = torch.ones(N, D, device=device, dtype=torch.float32)
    zeros_sin = torch.zeros(N, D, device=device, dtype=torch.float32)
    cached = (ones_inv_rms, ones_nw, ones_cos, zeros_sin)
    _CAM_IDENTITY_CACHE[key] = cached
    return cached


def cam_scan_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    reverse: bool = False,
    init_state: torch.Tensor | None = None,
    save_final_state: bool = False,
    dot_precision: int | None = None,
):
    """Drop-in chunkwise replacement for `cam_scan_func`.

    Args mirror `cam_scan_func` exactly:
      q, k, v: ``(B, H, D, N)`` fp32 contiguous (cam-prep'd: RMSNorm+ReLU+UCPE+RoPE)
      beta:    ``(B, H, F, S)`` fp32 contiguous
      decay:   ``(B, H, F)`` fp32 contiguous
      reverse: bwd flip-and-shift semantics (autograd path); not yet supported.
      init_state:       optional ``(B*H, BLOCK_D, BLOCK_D)`` fp32 — cross-chunk AR state.
      save_final_state: when True, also returns ``(out, final_state)``.

    Returns ``out`` of shape ``(B, H, D, N)`` fp32, or
    ``(out, final_state: (B*H, BLOCK_D, BLOCK_D))`` if save_final_state=True.
    """
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta.is_contiguous() and decay.is_contiguous()
    assert q.dtype == torch.float32, f"cam_scan_chunkwise requires fp32 q/k/v (got {q.dtype})"

    if reverse and (init_state is not None or save_final_state):
        raise NotImplementedError(
            "cam_scan_chunkwise: state passing (init_state / save_final_state) is "
            "only supported for the forward direction (reverse=False). The cam "
            "branch's anti-causal pass resets per chunk; there is no global "
            "cross-prefix state to cache for the reverse direction."
        )

    B, H, D, N = q.shape
    F = beta.shape[2]
    assert N % F == 0
    S = N // F
    assert beta.shape == (B, H, F, S)
    assert decay.shape == (B, H, F)

    BLOCK_D = triton.next_power_of_2(D)

    if dot_precision is None:
        dot_precision = _default_dot_prec()

    # Repack (B, H, D, N) → (B, N, 3, H, D) for chunkwise's qkv layout.
    # Avoid ``stack(...).permute(...).contiguous()`` because that materializes
    # two large tensors. Direct packing allocates the destination once.
    qkv = torch.empty(B, N, 3, H, D, device=q.device, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))

    # Identity prep tables — make chunkwise's RMSNorm + RoPE no-ops.
    ones_inv_rms, ones_nw, ones_cos, zeros_sin = _cam_identity_tables(B=B, N=N, H=H, D=D, device=q.device)

    # Phase A (skip_relu=True for cam-prep'd K; skip_z=True since cam has no Z scan).
    # k_scale=1.0 because cam_prep already applied K-scale.
    I_P_kv, A_, I_P_z, B_z = phase_a(
        qkv,
        beta,
        ones_inv_rms,
        ones_inv_rms,
        ones_nw,
        ones_nw,
        ones_cos,
        zeros_sin,
        F=F,
        S=S,
        k_scale=1.0,
        norm_eps=1e-5,
        dot_precision=dot_precision,
        skip_relu=True,
        skip_z=True,
    )

    # Phase B (forward direction only; cam supports init_state on fwd, save_final
    # on fwd; no rev). Pads (B*H, D, D) ↔ (B*H, BLOCK_D, BLOCK_D) inline.
    init_kv_padded = None
    init_z_padded = None
    if init_state is not None:
        if init_state.shape != (B * H, BLOCK_D, BLOCK_D):
            raise ValueError(
                f"cam_scan_chunkwise: init_state shape {tuple(init_state.shape)} "
                f"!= expected (B*H, BLOCK_D, BLOCK_D) = {(B * H, BLOCK_D, BLOCK_D)}"
            )
        if init_state.dtype != torch.float32:
            raise ValueError(f"cam_scan_chunkwise: init_state must be fp32 (got {init_state.dtype}).")
        if not init_state.is_contiguous():
            raise ValueError("cam_scan_chunkwise: init_state must be contiguous.")
        # Cam stores state as M[K_feat, V_feat]. Chunkwise's Phase B kernel reads
        # state with offs_dd = i*BLOCK_D + j where i is the fwd loop's M row.
        # Storage layout matches cam's (row-major (D_K, D_V)), so a direct cast
        # to fp32 contiguous is enough — no transpose needed.
        init_kv_padded = init_state.to(torch.float32).contiguous()
        # No Z state in cam — pass zeros to satisfy phase_b_triton.
        init_z_padded = torch.zeros(B * H, BLOCK_D, device=q.device, dtype=torch.float32)

    direction = 2 if reverse else 1
    if save_final_state:
        M_fwd, z_fwd_out, M_rev, z_rev_out, final_kv, _final_z = phase_b_triton(
            I_P_kv,
            A_,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=direction,
            init_state_kv=init_kv_padded,
            init_state_z=init_z_padded,
            return_final_state=True,
            skip_z=True,
        )
    else:
        M_fwd, z_fwd_out, M_rev, z_rev_out = phase_b_triton(
            I_P_kv,
            A_,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=direction,
            init_state_kv=init_kv_padded,
            init_state_z=init_z_padded,
            skip_z=True,
        )

    # For reverse (flip-and-shift bwd), Phase B's reverse mode produces M_rev
    # such that M_rev[F-1] = 0 and M_rev[t] = state computed from K/V at frames
    # {F-1, F-2, ..., t+1} — exactly cam's REVERSE=1 semantics.
    M_use = M_rev if reverse else M_fwd
    z_use = z_rev_out if reverse else z_fwd_out

    # Phase C — num-only (NUM_ONLY=True skips den compute + store).
    # z is unused with NUM_ONLY but still required by the kernel signature.
    num_out, _ = phase_c(
        qkv,
        ones_inv_rms,
        ones_nw,
        ones_cos,
        zeros_sin,
        M_use,
        z_use,
        F=F,
        S=S,
        dot_precision=dot_precision,
        skip_relu=True,
        num_only=True,
    )

    # Convert chunkwise output (B, N, H, D) → cam's (B, H, D, N) layout, fp32.
    out = num_out.permute(0, 2, 3, 1).contiguous().to(torch.float32)

    if save_final_state:
        return out, final_kv  # final_kv already (B*H, BLOCK_D, BLOCK_D) fp32
    return out


def cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    dot_precision: int | None = None,
) -> torch.Tensor:
    """Bidirectional camera scan using shared chunkwise phases.

    This is equivalent to ``cam_scan_chunkwise(..., reverse=False) +
    cam_scan_chunkwise(..., reverse=True)`` for full bidirectional attention,
    but it packs QKV once, runs Phase A once, combines forward/reverse histories
    inside Phase B, and runs Phase C once on the summed state.
    """
    _require_triton("cam_scan_bidi_chunkwise")
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta.is_contiguous() and decay.is_contiguous()
    assert q.dtype == torch.float32, f"cam_scan_bidi_chunkwise requires fp32 q/k/v (got {q.dtype})"

    B, H, D, N = q.shape
    F = beta.shape[2]
    assert N % F == 0
    S = N // F
    assert beta.shape == (B, H, F, S)
    assert decay.shape == (B, H, F)

    if dot_precision is None:
        dot_precision = _default_dot_prec()

    qkv = torch.empty(B, N, 3, H, D, device=q.device, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))

    ones_inv_rms, ones_nw, ones_cos, zeros_sin = _cam_identity_tables(B=B, N=N, H=H, D=D, device=q.device)
    I_P_kv, A_, I_P_z, B_z = phase_a(
        qkv,
        beta,
        ones_inv_rms,
        ones_inv_rms,
        ones_nw,
        ones_nw,
        ones_cos,
        zeros_sin,
        F=F,
        S=S,
        k_scale=1.0,
        norm_eps=1e-5,
        dot_precision=dot_precision,
        skip_relu=True,
        skip_z=True,
    )
    M_hist, z_hist, _, _ = phase_b_triton(
        I_P_kv,
        A_,
        I_P_z,
        B_z,
        decay,
        F=F,
        dot_precision=dot_precision,
        direction=0,
        combined_history=True,
        skip_z=True,
    )
    num_out, _ = phase_c(
        qkv,
        ones_inv_rms,
        ones_nw,
        ones_cos,
        zeros_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        skip_relu=True,
        num_only=True,
    )
    return num_out.permute(0, 2, 3, 1).contiguous().to(torch.float32)


def cam_scan_pair_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta_fwd: torch.Tensor,
    decay_fwd: torch.Tensor,
    beta_rev: torch.Tensor,
    decay_rev: torch.Tensor,
    *,
    dot_precision: int | None = None,
) -> torch.Tensor:
    """Sum a forward camera scan and a separately-gated reverse scan.

    Chunk-causal camera attention needs the reverse branch to use boundary-masked
    gates while the forward branch uses the original gates. This wrapper keeps
    that exact behavior but shares QKV packing, identity tables, and the final
    output layout conversion across the two scans.
    """
    assert q.shape == k.shape == v.shape, f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta_fwd.is_contiguous() and decay_fwd.is_contiguous()
    assert beta_rev.is_contiguous() and decay_rev.is_contiguous()
    assert q.dtype == torch.float32, f"cam_scan_pair_chunkwise requires fp32 q/k/v (got {q.dtype})"

    B, H, D, N = q.shape
    F = beta_fwd.shape[2]
    assert N % F == 0
    S = N // F
    assert beta_fwd.shape == beta_rev.shape == (B, H, F, S)
    assert decay_fwd.shape == decay_rev.shape == (B, H, F)

    if dot_precision is None:
        dot_precision = _default_dot_prec()

    qkv = torch.empty(B, N, 3, H, D, device=q.device, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))

    ones_inv_rms, ones_nw, ones_cos, zeros_sin = _cam_identity_tables(B=B, N=N, H=H, D=D, device=q.device)

    I_P_kv, A_, I_P_z, B_z = phase_a(
        qkv,
        beta_fwd,
        ones_inv_rms,
        ones_inv_rms,
        ones_nw,
        ones_nw,
        ones_cos,
        zeros_sin,
        F=F,
        S=S,
        k_scale=1.0,
        norm_eps=1e-5,
        dot_precision=dot_precision,
        skip_relu=True,
        skip_z=True,
    )
    M_fwd, z_fwd, _, _ = phase_b_triton(
        I_P_kv,
        A_,
        I_P_z,
        B_z,
        decay_fwd,
        F=F,
        dot_precision=dot_precision,
        direction=1,
        skip_z=True,
    )
    num_out, _ = phase_c(
        qkv,
        ones_inv_rms,
        ones_nw,
        ones_cos,
        zeros_sin,
        M_fwd,
        z_fwd,
        F=F,
        S=S,
        dot_precision=dot_precision,
        skip_relu=True,
        num_only=True,
    )
    del I_P_kv, A_, I_P_z, B_z, M_fwd, z_fwd

    I_P_kv, A_, I_P_z, B_z = phase_a(
        qkv,
        beta_rev,
        ones_inv_rms,
        ones_inv_rms,
        ones_nw,
        ones_nw,
        ones_cos,
        zeros_sin,
        F=F,
        S=S,
        k_scale=1.0,
        norm_eps=1e-5,
        dot_precision=dot_precision,
        skip_relu=True,
        skip_z=True,
    )
    _, _, M_rev, z_rev = phase_b_triton(
        I_P_kv,
        A_,
        I_P_z,
        B_z,
        decay_rev,
        F=F,
        dot_precision=dot_precision,
        direction=2,
        skip_z=True,
    )
    phase_c(
        qkv,
        ones_inv_rms,
        ones_nw,
        ones_cos,
        zeros_sin,
        M_rev,
        z_rev,
        F=F,
        S=S,
        dot_precision=dot_precision,
        num_out=num_out,
        accumulate=True,
        skip_relu=True,
        num_only=True,
    )
    return num_out.permute(0, 2, 3, 1).contiguous().to(torch.float32)


# ===== camera utility helpers (used by both kernels and the transformer) =====


def compute_fov_from_fx_xi(
    fx: Union[torch.Tensor, float],
    xi: Union[torch.Tensor, float],
    width: int,
    device="cpu",
    dtype=torch.float32,
):
    """Inverse of :func:`compute_fx_from_fov_xi`."""

    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    fx = to_tensor_1d(fx).reshape(-1)
    xi = to_tensor_1d(xi).reshape(-1)
    B = max(fx.shape[0], xi.shape[0])
    fx = fx.expand(B)
    xi = xi.expand(B)
    A = 2.0 * fx / width
    phi = torch.atan(1.0 / A)
    denom = torch.sqrt(A * A + 1.0)
    ratio = (xi / denom).clamp(-1.0, 1.0)
    theta = torch.asin(ratio) + phi
    x_fov = torch.rad2deg(2.0 * theta)
    return x_fov


def ucm_unproject_grid_fov(
    x_fov: Union[float, torch.Tensor],
    y_fov: Union[float, torch.Tensor],
    xi: Union[float, torch.Tensor],
    height: int,
    width: int,
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unproject grid with intrinsics expressed as FoV (degrees) + xi."""
    is_batched = any(torch.is_tensor(p) and p.numel() > 1 for p in [x_fov, y_fov, xi, cx, cy])
    fx = compute_fx_from_fov_xi(x_fov, xi, width, device, dtype)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, device, dtype)
    d_cam = ucm_unproject_grid(
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        xi=xi if torch.is_tensor(xi) else torch.tensor([xi], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        y_down=True,
    )
    if not is_batched:
        d_cam = d_cam[0]
    return d_cam


def world_to_ray_mats(
    d_cam: torch.Tensor,  # [H, W, 3], [B, H, W, 3], or [B, T, H, W, 3]
    c2w: torch.Tensor,  # [B, T, 4, 4]
) -> torch.Tensor:
    """Build per-pixel ``ray<-world`` transforms from camera unit rays + C2W poses."""
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    if d_cam.ndim == 4:
        B, H, W, _ = d_cam.shape
        T = c2w.shape[1]
        d_cam = repeat(d_cam, "b h w c -> b t h w c", t=T)
    elif d_cam.ndim == 5:
        B, T, H, W, _ = d_cam.shape
    else:
        raise ValueError(f"Unsupported d_cam shape: {d_cam.shape}")

    device = d_cam.device
    dtype = d_cam.dtype
    R_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", R_cam, d_cam)
    cam_y = R_cam[..., :, 1]
    cam_y = repeat(cam_y, "b t c -> b t h w c", h=H, w=W)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-6)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-6)
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    R_w2l = rearrange(R_l2w, "b t h w i j -> b t h w j i")
    t_world = repeat(t_cam, "b t c -> b t h w c", h=H, w=W)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", R_w2l, t_world)
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)
    return raymats


def create_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a pixel coordinate grid of shape ``(H, W, 3)`` or ``(B, H, W, 3)``."""
    if device.type == "cpu":
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: {dtype} is not supported by {device.type}\nIf device is `cpu`, use float32 or float64"
        )
    _xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    _ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    ys, xs = torch.meshgrid([_ys, _xs], indexing="ij")
    zs = torch.ones_like(xs, dtype=dtype, device=device)
    grid = torch.stack((xs, ys, zs), dim=2)
    if batch is not None:
        grid = repeat(grid, "... -> b ...", b=batch)
    return grid


def ucm_unproject_grid(
    height: int,
    width: int,
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
    xi: Union[float, torch.Tensor],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    y_down: bool = True,
) -> torch.Tensor:
    """Unproject pixel grid into a camera-frame direction vector using the UCM."""
    fx_, fy_, cx_, cy_, xi_ = fx, fy, cx, cy, xi

    def to_tensor_flatten(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).reshape(-1)
        return torch.tensor([x], dtype=dtype, device=device)

    fx, fy, cx, cy, xi = map(to_tensor_flatten, (fx, fy, cx, cy, xi))
    B = max(fx.shape[0], fy.shape[0], cx.shape[0], cy.shape[0], xi.shape[0])
    fx = fx.expand(B)
    fy = fy.expand(B)
    cx = cx.expand(B)
    cy = cy.expand(B)
    xi = xi.expand(B)

    grid = create_grid(height=height, width=width, batch=B, dtype=dtype, device=device)
    u = grid[..., 0]
    v = grid[..., 1]
    fx = fx[:, None, None]
    fy = fy[:, None, None]
    cx = cx[:, None, None]
    cy = cy[:, None, None]
    xi = xi[:, None, None]
    x = (u - cx) / fx
    y = (v - cy) / fy
    if not y_down:
        y = -y
    r2 = x * x + y * y
    alpha = xi + torch.sqrt(1 + (1 - xi * xi) * r2)
    gamma = alpha / (1 + r2)
    X = gamma * x
    Y = gamma * y
    Z = gamma - xi
    d_cam = torch.stack([X, Y, Z], dim=-1)
    is_scalar_input = all(not torch.is_tensor(p) for p in (fx_, fy_, cx_, cy_, xi_))
    if is_scalar_input:
        return d_cam[0]
    else:
        return d_cam


def compute_fx_from_fov_xi(
    x_fov: Union[torch.Tensor, float],
    xi: Union[torch.Tensor, float],
    width: int,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Recover focal length ``fx`` from horizontal FoV (degrees) + UCM xi."""

    def to_tensor_flatten(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).view(-1)
        return torch.tensor([x], dtype=dtype, device=device)

    x_fov = to_tensor_flatten(x_fov)
    xi = to_tensor_flatten(xi)
    B = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.expand(B)
    xi = xi.expand(B)
    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / denom
    return fx


def project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi):
    """Project 3D points in camera frame to UCM image plane."""
    r = torch.sqrt(X * X + Y * Y + Z * Z)

    def reshape_param(p, target):
        if torch.is_tensor(p):
            if p.numel() == 1:
                return p
            if p.ndim == 1 and target.ndim == 4:
                return p.view(target.shape[0], target.shape[1], 1, 1)
            while p.ndim < target.ndim:
                p = p.unsqueeze(-1)
        return p

    xi = reshape_param(xi, X)
    fx = reshape_param(fx, X)
    fy = reshape_param(fy, X)
    cx = reshape_param(cx, X)
    cy = reshape_param(cy, X)

    alpha = Z + xi * r
    du = fx * (X / alpha) + cx
    dv = fy * (Y / alpha) + cy
    return du, dv


def project_ucm_points_fov(X, Y, Z, x_fov, y_fov, xi, height, width, cx, cy):
    """Project 3D points in camera frame to UCM image plane using FoV-based intrinsics."""
    fx = compute_fx_from_fov_xi(x_fov, xi, width, X.device, X.dtype)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, X.device, X.dtype)
    return project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi)


def compute_up_lat_map(
    R: torch.Tensor,
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    delta: float = 0.1,
):
    """Compute UCPE absolute embedding maps ``(up_map, lat_map)``.

    ``up_map`` is a 2-channel projected up-direction; ``lat_map`` is a 1-channel
    latitude. Concatenated they form the 3-channel absmap consumed by the
    camera branch.
    """
    B, T, _, _ = R.shape
    dtype = R.dtype
    R = R.float()
    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        y_fov=y_fov,
        xi=xi,
        height=height,
        width=width,
        cx=cx,
        cy=cy,
        device=device,
        dtype=torch.float32,
    )

    if d_cam.ndim == 3:
        d_cam_exp = repeat(d_cam, "H W C -> B T H W C", B=B, T=T)
    elif d_cam.ndim == 4:
        if d_cam.shape[0] == B * T:
            d_cam_exp = d_cam.view(B, T, height, width, 3)
        else:
            d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)
    else:
        d_cam_exp = d_cam

    mask_exp = d_cam_exp.isnan().any(dim=-1, keepdim=True)
    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)
    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(-1)
    v = d_world
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    k = torch.cross(v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1)
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)
    delta_t = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps = torch.cos(delta_t)
    sin_eps = torch.sin(delta_t)
    v_rot = (
        v * cos_eps + torch.cross(k, v, dim=-1) * sin_eps + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)
    )
    dirs_cam = torch.einsum("btij,bthwj->bthwi", R.transpose(-1, -2), v_rot)
    Xs, Ys, Zs = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]
    du, dv = project_ucm_points_fov(
        Xs,
        Ys,
        Zs,
        x_fov=x_fov.float(),
        y_fov=y_fov.float(),
        xi=xi.float(),
        height=height,
        width=width,
        cx=cx.float(),
        cy=cy.float(),
    )
    grid = create_grid(
        height=height,
        width=width,
        batch=B,
        dtype=torch.float32,
        device=device,
    )
    grid_x = grid[..., 0].unsqueeze(1)
    grid_y = grid[..., 1].unsqueeze(1)
    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)
    up_map = up_map.to(dtype=dtype)
    lat_map = lat_map.to(dtype=dtype)
    up_map = up_map.masked_fill(mask_exp, 0.0)
    lat_map = lat_map.masked_fill(mask_exp, 0.0)
    return up_map, lat_map
