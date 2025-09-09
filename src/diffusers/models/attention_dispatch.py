# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import contextlib
import functools
import inspect
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch

from ..utils import (
    get_logger,
    is_flash_attn_3_available,
    is_flash_attn_available,
    is_flash_attn_version,
    is_kernels_available,
    is_sageattention_available,
    is_sageattention_version,
    is_torch_npu_available,
    is_torch_version,
    is_torch_xla_available,
    is_torch_xla_version,
    is_xformers_available,
    is_xformers_version,
)
from ..utils.constants import DIFFUSERS_ATTN_BACKEND, DIFFUSERS_ATTN_CHECKS, DIFFUSERS_ENABLE_HUB_KERNELS


_REQUIRED_FLASH_VERSION = "2.6.3"
_REQUIRED_SAGE_VERSION = "2.1.1"
_REQUIRED_FLEX_VERSION = "2.5.0"
_REQUIRED_XLA_VERSION = "2.2"
_REQUIRED_XFORMERS_VERSION = "0.0.29"

_CAN_USE_FLASH_ATTN = is_flash_attn_available() and is_flash_attn_version(">=", _REQUIRED_FLASH_VERSION)
_CAN_USE_FLASH_ATTN_3 = is_flash_attn_3_available()
_CAN_USE_SAGE_ATTN = is_sageattention_available() and is_sageattention_version(">=", _REQUIRED_SAGE_VERSION)
_CAN_USE_FLEX_ATTN = is_torch_version(">=", _REQUIRED_FLEX_VERSION)
_CAN_USE_NPU_ATTN = is_torch_npu_available()
_CAN_USE_XLA_ATTN = is_torch_xla_available() and is_torch_xla_version(">=", _REQUIRED_XLA_VERSION)
_CAN_USE_XFORMERS_ATTN = is_xformers_available() and is_xformers_version(">=", _REQUIRED_XFORMERS_VERSION)


if _CAN_USE_FLASH_ATTN:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
else:
    flash_attn_func = None
    flash_attn_varlen_func = None


if _CAN_USE_FLASH_ATTN_3:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
else:
    flash_attn_3_func = None
    flash_attn_3_varlen_func = None

if DIFFUSERS_ENABLE_HUB_KERNELS:
    if not is_kernels_available():
        raise ImportError(
            "To use FA3 kernel for your hardware from the Hub, the `kernels` library must be installed. Install with `pip install kernels`."
        )
    from ..utils.kernels_utils import _get_fa3_from_hub

    flash_attn_interface_hub = _get_fa3_from_hub()
    flash_attn_3_func_hub = flash_attn_interface_hub.flash_attn_func
else:
    flash_attn_3_func_hub = None

if _CAN_USE_SAGE_ATTN:
    from sageattention import (
        sageattn,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_varlen,
    )
else:
    sageattn = None
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp16_triton = None
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    sageattn_varlen = None


if _CAN_USE_FLEX_ATTN:
    # We cannot import the flex_attention function from the package directly because it is expected (from the
    # pytorch documentation) that the user may compile it. If we import directly, we will not have access to the
    # compiled function.
    import torch.nn.attention.flex_attention as flex_attention


if _CAN_USE_NPU_ATTN:
    from torch_npu import npu_fusion_attention
else:
    npu_fusion_attention = None


if _CAN_USE_XLA_ATTN:
    from torch_xla.experimental.custom_kernel import flash_attention as xla_flash_attention
else:
    xla_flash_attention = None


if _CAN_USE_XFORMERS_ATTN:
    import xformers.ops as xops
else:
    xops = None

# Version guard for PyTorch compatibility - custom_op was added in PyTorch 2.4
if torch.__version__ >= "2.4.0":
    _custom_op = torch.library.custom_op
    _register_fake = torch.library.register_fake
else:

    def custom_op_no_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    def register_fake_no_op(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    _custom_op = custom_op_no_op
    _register_fake = register_fake_no_op


logger = get_logger(__name__)  # pylint: disable=invalid-name

# TODO(aryan): Add support for the following:
# - Sage Attention++
# - block sparse, radial and other attention methods
# - CP with sage attention, flex, xformers, other missing backends
# - Add support for normal and CP training with backends that don't support it yet

_SAGE_ATTENTION_PV_ACCUM_DTYPE = Literal["fp32", "fp32+fp32"]
_SAGE_ATTENTION_QK_QUANT_GRAN = Literal["per_thread", "per_warp"]
_SAGE_ATTENTION_QUANTIZATION_BACKEND = Literal["cuda", "triton"]


class AttentionBackendName(str, Enum):
    # EAGER = "eager"

    # `flash-attn`
    FLASH = "flash"
    FLASH_VARLEN = "flash_varlen"
    _FLASH_3 = "_flash_3"
    _FLASH_VARLEN_3 = "_flash_varlen_3"
    _FLASH_3_HUB = "_flash_3_hub"
    # _FLASH_VARLEN_3_HUB = "_flash_varlen_3_hub"  # not supported yet.

    # PyTorch native
    FLEX = "flex"
    NATIVE = "native"
    _NATIVE_CUDNN = "_native_cudnn"
    _NATIVE_EFFICIENT = "_native_efficient"
    _NATIVE_FLASH = "_native_flash"
    _NATIVE_MATH = "_native_math"
    _NATIVE_NPU = "_native_npu"
    _NATIVE_XLA = "_native_xla"

    # `sageattention`
    SAGE = "sage"
    SAGE_VARLEN = "sage_varlen"
    _SAGE_QK_INT8_PV_FP8_CUDA = "_sage_qk_int8_pv_fp8_cuda"
    _SAGE_QK_INT8_PV_FP8_CUDA_SM90 = "_sage_qk_int8_pv_fp8_cuda_sm90"
    _SAGE_QK_INT8_PV_FP16_CUDA = "_sage_qk_int8_pv_fp16_cuda"
    _SAGE_QK_INT8_PV_FP16_TRITON = "_sage_qk_int8_pv_fp16_triton"
    # TODO: let's not add support for Sparge Attention now because it requires tuning per model
    # We can look into supporting something "autotune"-ing in the future
    # SPARGE = "sparge"

    # `xformers`
    XFORMERS = "xformers"


class _AttentionBackendRegistry:
    _backends = {}
    _constraints = {}
    _supported_arg_names = {}
    _active_backend = AttentionBackendName(DIFFUSERS_ATTN_BACKEND)
    _checks_enabled = DIFFUSERS_ATTN_CHECKS

    @classmethod
    def register(cls, backend: AttentionBackendName, constraints: Optional[List[Callable]] = None):
        logger.debug(f"Registering attention backend: {backend} with constraints: {constraints}")

        def decorator(func):
            cls._backends[backend] = func
            cls._constraints[backend] = constraints or []
            cls._supported_arg_names[backend] = set(inspect.signature(func).parameters.keys())
            return func

        return decorator

    @classmethod
    def get_active_backend(cls):
        return cls._active_backend, cls._backends[cls._active_backend]

    @classmethod
    def list_backends(cls):
        return list(cls._backends.keys())


@contextlib.contextmanager
def attention_backend(backend: Union[str, AttentionBackendName] = AttentionBackendName.NATIVE):
    """
    Context manager to set the active attention backend.
    """
    if backend not in _AttentionBackendRegistry._backends:
        raise ValueError(f"Backend {backend} is not registered.")

    backend = AttentionBackendName(backend)
    _check_attention_backend_requirements(backend)

    old_backend = _AttentionBackendRegistry._active_backend
    _AttentionBackendRegistry._active_backend = backend

    try:
        yield
    finally:
        _AttentionBackendRegistry._active_backend = old_backend


def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[AttentionBackendName] = None,
) -> torch.Tensor:
    attention_kwargs = attention_kwargs or {}

    if backend is None:
        # If no backend is specified, we either use the default backend (set via the DIFFUSERS_ATTN_BACKEND environment
        # variable), or we use a custom backend based on whether user is using the `attention_backend` context manager
        backend_name, backend_fn = _AttentionBackendRegistry.get_active_backend()
    else:
        backend_name = AttentionBackendName(backend)
        backend_fn = _AttentionBackendRegistry._backends.get(backend_name)

    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        **attention_kwargs,
    }
    if is_torch_version(">=", "2.5.0"):
        kwargs["enable_gqa"] = enable_gqa

    if _AttentionBackendRegistry._checks_enabled:
        removed_kwargs = set(kwargs) - set(_AttentionBackendRegistry._supported_arg_names[backend_name])
        if removed_kwargs:
            logger.warning(f"Removing unsupported arguments for attention backend {backend_name}: {removed_kwargs}.")
        for check in _AttentionBackendRegistry._constraints.get(backend_name):
            check(**kwargs)

    kwargs = {k: v for k, v in kwargs.items() if k in _AttentionBackendRegistry._supported_arg_names[backend_name]}
    return backend_fn(**kwargs)


# ===== Checks =====
# A list of very simple functions to catch common errors quickly when debugging.


def _check_attn_mask_or_causal(attn_mask: Optional[torch.Tensor], is_causal: bool, **kwargs) -> None:
    if attn_mask is not None and is_causal:
        raise ValueError("`is_causal` cannot be True when `attn_mask` is not None.")


def _check_device(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    if query.device != key.device or query.device != value.device:
        raise ValueError("Query, key, and value must be on the same device.")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Query, key, and value must have the same dtype.")


def _check_device_cuda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    _check_device(query, key, value)
    if query.device.type != "cuda":
        raise ValueError("Query, key, and value must be on a CUDA device.")


def _check_device_cuda_atleast_smXY(major: int, minor: int) -> Callable:
    def check_device_cuda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
        _check_device_cuda(query, key, value)
        if torch.cuda.get_device_capability(query.device) < (major, minor):
            raise ValueError(
                f"Query, key, and value must be on a CUDA device with compute capability >= {major}.{minor}."
            )

    return check_device_cuda


def _check_qkv_dtype_match(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    if query.dtype != key.dtype:
        raise ValueError("Query and key must have the same dtype.")
    if query.dtype != value.dtype:
        raise ValueError("Query and value must have the same dtype.")


def _check_qkv_dtype_bf16_or_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    _check_qkv_dtype_match(query, key, value)
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Query, key, and value must be either bfloat16 or float16.")


def _check_shape(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> None:
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key must have the same last dimension.")
    if query.shape[-2] != value.shape[-2]:
        raise ValueError("Query and value must have the same second to last dimension.")
    if attn_mask is not None and attn_mask.shape[-1] != key.shape[-2]:
        raise ValueError("Attention mask must match the key's second to last dimension.")


# ===== Helper functions =====


def _check_attention_backend_requirements(backend: AttentionBackendName) -> None:
    if backend in [AttentionBackendName.FLASH, AttentionBackendName.FLASH_VARLEN]:
        if not _CAN_USE_FLASH_ATTN:
            raise RuntimeError(
                f"Flash Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `flash-attn>={_REQUIRED_FLASH_VERSION}`."
            )

    elif backend in [AttentionBackendName._FLASH_3, AttentionBackendName._FLASH_VARLEN_3]:
        if not _CAN_USE_FLASH_ATTN_3:
            raise RuntimeError(
                f"Flash Attention 3 backend '{backend.value}' is not usable because of missing package or the version is too old. Please build FA3 beta release from source."
            )

    # TODO: add support Hub variant of FA3 varlen later
    elif backend in [AttentionBackendName._FLASH_3_HUB]:
        if not DIFFUSERS_ENABLE_HUB_KERNELS:
            raise RuntimeError(
                f"Flash Attention 3 Hub backend '{backend.value}' is not usable because the `DIFFUSERS_ENABLE_HUB_KERNELS` env var isn't set. Please set it like `export DIFFUSERS_ENABLE_HUB_KERNELS=yes`."
            )
        if not is_kernels_available():
            raise RuntimeError(
                f"Flash Attention 3 Hub backend '{backend.value}' is not usable because the `kernels` package isn't available. Please install it with `pip install kernels`."
            )

    elif backend in [
        AttentionBackendName.SAGE,
        AttentionBackendName.SAGE_VARLEN,
        AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA,
        AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA_SM90,
        AttentionBackendName._SAGE_QK_INT8_PV_FP16_CUDA,
        AttentionBackendName._SAGE_QK_INT8_PV_FP16_TRITON,
    ]:
        if not _CAN_USE_SAGE_ATTN:
            raise RuntimeError(
                f"Sage Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `sageattention>={_REQUIRED_SAGE_VERSION}`."
            )

    elif backend == AttentionBackendName.FLEX:
        if not _CAN_USE_FLEX_ATTN:
            raise RuntimeError(
                f"Flex Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `torch>=2.5.0`."
            )

    elif backend == AttentionBackendName._NATIVE_NPU:
        if not _CAN_USE_NPU_ATTN:
            raise RuntimeError(
                f"NPU Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `torch_npu`."
            )

    elif backend == AttentionBackendName._NATIVE_XLA:
        if not _CAN_USE_XLA_ATTN:
            raise RuntimeError(
                f"XLA Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `torch_xla>={_REQUIRED_XLA_VERSION}`."
            )

    elif backend == AttentionBackendName.XFORMERS:
        if not _CAN_USE_XFORMERS_ATTN:
            raise RuntimeError(
                f"Xformers Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `xformers>={_REQUIRED_XFORMERS_VERSION}`."
            )


@functools.lru_cache(maxsize=128)
def _prepare_for_flash_attn_or_sage_varlen_without_mask(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    device: Optional[torch.device] = None,
):
    seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
    seqlens_k = torch.full((batch_size,), seq_len_kv, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def _prepare_for_flash_attn_or_sage_varlen_with_mask(
    batch_size: int,
    seq_len_q: int,
    attn_mask: torch.Tensor,
    device: Optional[torch.device] = None,
):
    seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
    seqlens_k = attn_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def _prepare_for_flash_attn_or_sage_varlen(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    attn_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> None:
    if attn_mask is None:
        return _prepare_for_flash_attn_or_sage_varlen_without_mask(batch_size, seq_len_q, seq_len_kv, device)
    return _prepare_for_flash_attn_or_sage_varlen_with_mask(batch_size, seq_len_q, attn_mask, device)


def _normalize_attn_mask(attn_mask: torch.Tensor, batch_size: int, seq_len_k: int) -> torch.Tensor:
    """
    Normalize an attention mask to shape [batch_size, seq_len_k] (bool) suitable for inferring seqlens_[q|k] in
    FlashAttention/Sage varlen.

    Supports 1D to 4D shapes and common broadcasting patterns.
    """
    if attn_mask.dtype != torch.bool:
        raise ValueError(f"Attention mask must be of type bool, got {attn_mask.dtype}.")

    if attn_mask.ndim == 1:
        # [seq_len_k] -> broadcast across batch
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 2:
        # [batch_size, seq_len_k]. Maybe broadcast across batch
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 2D attention mask."
            )
        attn_mask = attn_mask.expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 3:
        # [batch_size, seq_len_q, seq_len_k] -> reduce over query dimension
        # We do this reduction because we know that arbitrary QK masks is not supported in Flash/Sage varlen.
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 3D attention mask."
            )
        attn_mask = attn_mask.any(dim=1)
        attn_mask = attn_mask.expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 4:
        # [batch_size, num_heads, seq_len_q, seq_len_k] or broadcastable versions
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 4D attention mask."
            )
        attn_mask = attn_mask.expand(batch_size, -1, -1, seq_len_k)  # [B, H, Q, K]
        attn_mask = attn_mask.any(dim=(1, 2))  # [B, K]

    else:
        raise ValueError(f"Unsupported attention mask shape: {attn_mask.shape}")

    if attn_mask.shape != (batch_size, seq_len_k):
        raise ValueError(
            f"Normalized attention mask shape mismatch: got {attn_mask.shape}, expected ({batch_size}, {seq_len_k})"
        )

    return attn_mask


def _flex_attention_causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
    return q_idx >= kv_idx


# ===== torch op registrations =====
# Registrations are required for fullgraph tracing compatibility
# TODO: this is only required because the beta release FA3 does not have it. There is a PR adding
# this but it was never merged: https://github.com/Dao-AILab/flash-attention/pull/1590


@_custom_op("flash_attn_3::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _wrapped_flash_attn_3_original(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = flash_attn_3_func(query, key, value)
    lse = lse.permute(0, 2, 1)
    return out, lse


@_register_fake("flash_attn_3::_flash_attn_forward")
def _(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_heads, head_dim = query.shape
    lse_shape = (batch_size, seq_len, num_heads)
    return torch.empty_like(query), query.new_empty(lse_shape)


# ===== Attention backends =====


@_AttentionBackendRegistry.register(
    AttentionBackendName.FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    out = flash_attn_func(
        q=query,
        k=key,
        v=value,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName.FLASH_VARLEN,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _flash_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    if any(x is None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
        (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
            _prepare_for_flash_attn_or_sage_varlen(
                batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
            )
        )
    else:
        seqlens_k = torch.full((batch_size,), max_seqlen_k, dtype=torch.int32, device=query.device)
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=query.device)
        cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32, device=query.device)

    key_valid, value_valid = [], []
    for b in range(batch_size):
        valid_len = seqlens_k[b]
        key_valid.append(key[b, :valid_len])
        value_valid.append(value[b, :valid_len])

    query_packed = query.flatten(0, 1)
    key_packed = torch.cat(key_valid, dim=0)
    value_packed = torch.cat(value_valid, dim=0)

    out = flash_attn_varlen_func(
        q=query_packed,
        k=key_packed,
        v=value_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    out = out.unflatten(0, (batch_size, -1))

    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._FLASH_3,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _flash_attention_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    out, lse, *_ = flash_attn_3_func(
        q=query,
        k=key,
        v=value,
        softmax_scale=scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        softcap=softcap,
        num_splits=1,
        pack_gqa=None,
        deterministic=deterministic,
        sm_margin=0,
    )
    return (out, lse) if return_attn_probs else out


@_AttentionBackendRegistry.register(
    AttentionBackendName._FLASH_3_HUB,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _flash_attention_3_hub(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    out = flash_attn_3_func_hub(
        q=query,
        k=key,
        v=value,
        softmax_scale=scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        softcap=softcap,
        num_splits=1,
        pack_gqa=None,
        deterministic=deterministic,
        sm_margin=0,
        return_attn_probs=return_attn_probs,
    )
    # When `return_attn_probs` is True, the above returns a tuple of
    # actual outputs and lse.
    return (out[0], out[1]) if return_attn_probs else out


@_AttentionBackendRegistry.register(
    AttentionBackendName._FLASH_VARLEN_3,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _flash_varlen_attention_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    if any(x is None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
        (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
            _prepare_for_flash_attn_or_sage_varlen(
                batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
            )
        )
    else:
        seqlens_k = torch.full((batch_size,), max_seqlen_k, dtype=torch.int32, device=query.device)
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=query.device)
        cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32, device=query.device)

    key_valid, value_valid = [], []
    for b in range(batch_size):
        valid_len = seqlens_k[b]
        key_valid.append(key[b, :valid_len])
        value_valid.append(value[b, :valid_len])

    query_packed = query.flatten(0, 1)
    key_packed = torch.cat(key_valid, dim=0)
    value_packed = torch.cat(value_valid, dim=0)

    out, lse, *_ = flash_attn_3_varlen_func(
        q=query_packed,
        k=key_packed,
        v=value_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=None,
        seqused_k=None,
        softmax_scale=scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        softcap=softcap,
        num_splits=1,
        pack_gqa=None,
        deterministic=deterministic,
        sm_margin=0,
    )
    out = out.unflatten(0, (batch_size, -1))

    return (out, lse) if return_attn_probs else out


@_AttentionBackendRegistry.register(
    AttentionBackendName.FLEX,
    constraints=[_check_attn_mask_or_causal, _check_device, _check_shape],
)
def _native_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[Union[torch.Tensor, "flex_attention.BlockMask"]] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    # TODO: should we LRU cache the block mask creation?
    score_mod = None
    block_mask = None
    batch_size, seq_len_q, num_heads, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is None or isinstance(attn_mask, flex_attention.BlockMask):
        block_mask = attn_mask
    elif is_causal:
        block_mask = flex_attention.create_block_mask(
            _flex_attention_causal_mask_mod, batch_size, num_heads, seq_len_q, seq_len_kv, query.device
        )
    elif torch.is_tensor(attn_mask):
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.view(attn_mask.size(0), 1, attn_mask.size(1), 1)

        attn_mask = attn_mask.expand(batch_size, num_heads, seq_len_q, seq_len_kv)

        if attn_mask.dtype == torch.bool:
            # TODO: this probably does not work but verify!
            def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                return attn_mask[batch_idx, head_idx, q_idx, kv_idx]

            block_mask = flex_attention.create_block_mask(
                mask_mod, batch_size, None, seq_len_q, seq_len_kv, query.device
            )
        else:

            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                return score + attn_mask[batch_idx, head_idx, q_idx, kv_idx]
    else:
        raise ValueError("Attention mask must be either None, a BlockMask, or a 2D/4D tensor.")

    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = flex_attention.flex_attention(
        query=query,
        key=key,
        value=value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
    )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName.NATIVE,
    constraints=[_check_device, _check_shape],
)
def _native_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_CUDNN,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _native_cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_EFFICIENT,
    constraints=[_check_device, _check_shape],
)
def _native_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _native_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,  # not supported
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_MATH,
    constraints=[_check_device, _check_shape],
)
def _native_math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_NPU,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _native_npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    out = npu_fusion_attention(
        query,
        key,
        value,
        query.size(1),  # num_heads
        input_layout="BNSD",
        pse=None,
        scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
        pre_tockens=65536,
        next_tockens=65536,
        keep_prob=1.0 - dropout_p,
        sync=False,
        inner_precise=0,
    )[0]
    out = out.transpose(1, 2).contiguous()
    return out


# Reference: https://github.com/pytorch/xla/blob/06c5533de6588f6b90aa1655d9850bcf733b90b4/torch_xla/experimental/custom_kernel.py#L853
@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_XLA,
    constraints=[_check_device, _check_shape],
)
def _native_xla_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    query = query / math.sqrt(query.shape[-1])
    out = xla_flash_attention(
        q=query,
        k=key,
        v=value,
        causal=is_causal,
    )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName.SAGE,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )


@_AttentionBackendRegistry.register(
    AttentionBackendName.SAGE_VARLEN,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _sage_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    smooth_k: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    if any(x is None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
        (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
            _prepare_for_flash_attn_or_sage_varlen(
                batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
            )
        )
    else:
        seqlens_k = torch.full((batch_size,), max_seqlen_k, dtype=torch.int32, device=query.device)
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=query.device)
        cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32, device=query.device)

    key_valid, value_valid = [], []
    for b in range(batch_size):
        valid_len = seqlens_k[b]
        key_valid.append(key[b, :valid_len])
        value_valid.append(value[b, :valid_len])

    query_packed = query.flatten(0, 1)
    key_packed = torch.cat(key_valid, dim=0)
    value_packed = torch.cat(value_valid, dim=0)

    out = sageattn_varlen(
        q=query_packed,
        k=key_packed,
        v=value_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_causal=is_causal,
        sm_scale=scale,
        smooth_k=smooth_k,
    )
    out = out.unflatten(0, (batch_size, -1))

    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA,
    constraints=[_check_device_cuda_atleast_smXY(9, 0), _check_shape],
)
def _sage_qk_int8_pv_fp8_cuda_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    qk_quant_gran: _SAGE_ATTENTION_QK_QUANT_GRAN = "per_thread",
    pv_accum_dtype: _SAGE_ATTENTION_PV_ACCUM_DTYPE = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp8_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )


@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA_SM90,
    constraints=[_check_device_cuda_atleast_smXY(9, 0), _check_shape],
)
def _sage_qk_int8_pv_fp8_cuda_sm90_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    qk_quant_gran: _SAGE_ATTENTION_QK_QUANT_GRAN = "per_thread",
    pv_accum_dtype: _SAGE_ATTENTION_PV_ACCUM_DTYPE = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp8_cuda_sm90(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )


@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP16_CUDA,
    constraints=[_check_device_cuda_atleast_smXY(8, 0), _check_shape],
)
def _sage_qk_int8_pv_fp16_cuda_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    qk_quant_gran: _SAGE_ATTENTION_QK_QUANT_GRAN = "per_thread",
    pv_accum_dtype: _SAGE_ATTENTION_PV_ACCUM_DTYPE = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp16_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )


@_AttentionBackendRegistry.register(
    AttentionBackendName._SAGE_QK_INT8_PV_FP16_TRITON,
    constraints=[_check_device_cuda_atleast_smXY(8, 0), _check_shape],
)
def _sage_qk_int8_pv_fp16_triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    quantization_backend: _SAGE_ATTENTION_QUANTIZATION_BACKEND = "triton",
    smooth_k: bool = True,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp16_triton(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        quantization_backend=quantization_backend,
        is_causal=is_causal,
        sm_scale=scale,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )


@_AttentionBackendRegistry.register(
    AttentionBackendName.XFORMERS,
    constraints=[_check_attn_mask_or_causal, _check_device, _check_shape],
)
def _xformers_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    batch_size, seq_len_q, num_heads_q, _ = query.shape
    _, seq_len_kv, num_heads_kv, _ = key.shape

    if is_causal:
        attn_mask = xops.LowerTriangularMask()
    elif attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.view(attn_mask.size(0), 1, attn_mask.size(1), 1)
        elif attn_mask.ndim != 4:
            raise ValueError("Only 2D and 4D attention masks are supported for xformers attention.")
        attn_mask = attn_mask.expand(batch_size, num_heads_q, seq_len_q, seq_len_kv).type_as(query)

    if enable_gqa:
        if num_heads_q % num_heads_kv != 0:
            raise ValueError("Number of heads in query must be divisible by number of heads in key/value.")
        num_heads_per_group = num_heads_q // num_heads_kv
        query = query.unflatten(2, (num_heads_kv, -1))
        key = key.unflatten(2, (num_heads_kv, -1)).expand(-1, -1, -1, num_heads_per_group, -1)
        value = value.unflatten(2, (num_heads_kv, -1)).expand(-1, -1, -1, num_heads_per_group, -1)

    out = xops.memory_efficient_attention(query, key, value, attn_mask, dropout_p, scale)

    if enable_gqa:
        out = out.flatten(2, 3)

    return out
