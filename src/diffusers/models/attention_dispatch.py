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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch


if torch.distributed.is_available():
    import torch.distributed._functional_collectives as funcol

from ..utils import (
    get_logger,
    is_aiter_available,
    is_aiter_version,
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


if TYPE_CHECKING:
    from ._modeling_parallel import ParallelConfig

_REQUIRED_FLASH_VERSION = "2.6.3"
_REQUIRED_AITER_VERSION = "0.1.5"
_REQUIRED_SAGE_VERSION = "2.1.1"
_REQUIRED_FLEX_VERSION = "2.5.0"
_REQUIRED_XLA_VERSION = "2.2"
_REQUIRED_XFORMERS_VERSION = "0.0.29"

_CAN_USE_FLASH_ATTN = is_flash_attn_available() and is_flash_attn_version(">=", _REQUIRED_FLASH_VERSION)
_CAN_USE_FLASH_ATTN_3 = is_flash_attn_3_available()
_CAN_USE_AITER_ATTN = is_aiter_available() and is_aiter_version(">=", _REQUIRED_AITER_VERSION)
_CAN_USE_SAGE_ATTN = is_sageattention_available() and is_sageattention_version(">=", _REQUIRED_SAGE_VERSION)
_CAN_USE_FLEX_ATTN = is_torch_version(">=", _REQUIRED_FLEX_VERSION)
_CAN_USE_NPU_ATTN = is_torch_npu_available()
_CAN_USE_XLA_ATTN = is_torch_xla_available() and is_torch_xla_version(">=", _REQUIRED_XLA_VERSION)
_CAN_USE_XFORMERS_ATTN = is_xformers_available() and is_xformers_version(">=", _REQUIRED_XFORMERS_VERSION)


if _CAN_USE_FLASH_ATTN:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward, _wrapped_flash_attn_forward
else:
    flash_attn_func = None
    flash_attn_varlen_func = None
    _wrapped_flash_attn_backward = None
    _wrapped_flash_attn_forward = None


if _CAN_USE_FLASH_ATTN_3:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
else:
    flash_attn_3_func = None
    flash_attn_3_varlen_func = None


if _CAN_USE_AITER_ATTN:
    from aiter import flash_attn_func as aiter_flash_attn_func
else:
    aiter_flash_attn_func = None

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

    # `aiter`
    AITER = "aiter"

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
    _supports_context_parallel = {}
    _active_backend = AttentionBackendName(DIFFUSERS_ATTN_BACKEND)
    _checks_enabled = DIFFUSERS_ATTN_CHECKS

    @classmethod
    def register(
        cls,
        backend: AttentionBackendName,
        constraints: Optional[List[Callable]] = None,
        supports_context_parallel: bool = False,
    ):
        logger.debug(f"Registering attention backend: {backend} with constraints: {constraints}")

        def decorator(func):
            cls._backends[backend] = func
            cls._constraints[backend] = constraints or []
            cls._supported_arg_names[backend] = set(inspect.signature(func).parameters.keys())
            cls._supports_context_parallel[backend] = supports_context_parallel
            return func

        return decorator

    @classmethod
    def get_active_backend(cls):
        return cls._active_backend, cls._backends[cls._active_backend]

    @classmethod
    def list_backends(cls):
        return list(cls._backends.keys())

    @classmethod
    def _is_context_parallel_enabled(
        cls, backend: AttentionBackendName, parallel_config: Optional["ParallelConfig"]
    ) -> bool:
        supports_context_parallel = backend in cls._supports_context_parallel
        is_degree_greater_than_1 = parallel_config is not None and (
            parallel_config.context_parallel_config.ring_degree > 1
            or parallel_config.context_parallel_config.ulysses_degree > 1
        )
        return supports_context_parallel and is_degree_greater_than_1


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
    parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    attention_kwargs = attention_kwargs or {}

    if backend is None:
        # If no backend is specified, we either use the default backend (set via the DIFFUSERS_ATTN_BACKEND environment
        # variable), or we use a custom backend based on whether user is using the `attention_backend` context manager
        backend_name, backend_fn = _AttentionBackendRegistry.get_active_backend()
    else:
        backend_name = AttentionBackendName(backend)
        backend_fn = _AttentionBackendRegistry._backends.get(backend_name)

    if parallel_config is not None and not _AttentionBackendRegistry._is_context_parallel_enabled(
        backend_name, parallel_config
    ):
        raise ValueError(
            f"Backend {backend_name} either does not support context parallelism or context parallelism "
            f"was enabled with a world size of 1."
        )

    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        **attention_kwargs,
        "_parallel_config": parallel_config,
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

    elif backend == AttentionBackendName.AITER:
        if not _CAN_USE_AITER_ATTN:
            raise RuntimeError(
                f"Aiter Attention backend '{backend.value}' is not usable because of missing package or the version is too old. Please install `aiter>={_REQUIRED_AITER_VERSION}`."
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
@_custom_op("_diffusers_flash_attn_3::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _wrapped_flash_attn_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Hardcoded for now because pytorch does not support tuple/int type hints
    window_size = (-1, -1)
    out, lse, *_ = flash_attn_3_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        sm_margin=sm_margin,
    )
    lse = lse.permute(0, 2, 1)
    return out, lse


@_register_fake("_diffusers_flash_attn_3::_flash_attn_forward")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    window_size = (-1, -1)  # noqa: F841
    # A lot of the parameters here are not yet used in any way within diffusers.
    # We can safely ignore for now and keep the fake op shape propagation simple.
    batch_size, seq_len, num_heads, head_dim = q.shape
    lse_shape = (batch_size, seq_len, num_heads)
    return torch.empty_like(q), q.new_empty(lse_shape)


# ===== Helper functions to use attention backends with templated CP autograd functions =====


def _native_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    # Native attention does not return_lse
    if return_lse:
        raise ValueError("Native attention does not support return_lse=True")

    # used for backward pass
    if _save_ctx:
        ctx.save_for_backward(query, key, value)
        ctx.attn_mask = attn_mask
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa

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


def _native_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    query, key, value = ctx.saved_tensors

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        query=query_t,
        key=key_t,
        value=value_t,
        attn_mask=ctx.attn_mask,
        dropout_p=ctx.dropout_p,
        is_causal=ctx.is_causal,
        scale=ctx.scale,
        enable_gqa=ctx.enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)

    grad_out_t = grad_out.permute(0, 2, 1, 3)
    grad_query_t, grad_key_t, grad_value_t = torch.autograd.grad(
        outputs=out, inputs=[query_t, key_t, value_t], grad_outputs=grad_out_t, retain_graph=False
    )

    grad_query = grad_query_t.permute(0, 2, 1, 3)
    grad_key = grad_key_t.permute(0, 2, 1, 3)
    grad_value = grad_value_t.permute(0, 2, 1, 3)

    return grad_query, grad_key, grad_value


# https://github.com/pytorch/pytorch/blob/8904ba638726f8c9a5aff5977c4aa76c9d2edfa6/aten/src/ATen/native/native_functions.yaml#L14958
# forward declaration:
#   aten::_scaled_dot_product_cudnn_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
def _cudnn_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for cuDNN attention.")

    tensors_to_save = ()

    # Contiguous is a must here! Calling cuDNN backend with aten ops produces incorrect results
    # if the input tensors are not contiguous.
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    tensors_to_save += (query, key, value)

    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_mask,
            compute_log_sumexp=return_lse,
            dropout_p=dropout_p,
            is_causal=is_causal,
            return_debug_mask=False,
            scale=scale,
        )
    )

    tensors_to_save += (out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)
    if _save_ctx:
        ctx.save_for_backward(*tensors_to_save)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.attn_mask = attn_mask
        ctx.max_q = max_q
        ctx.max_k = max_k

    out = out.transpose(1, 2).contiguous()
    if lse is not None:
        lse = lse.transpose(1, 2).contiguous()
    return (out, lse) if return_lse else out


# backward declaration:
#   aten::_scaled_dot_product_cudnn_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, Tensor attn_bias, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, *, float? scale=None) -> (Tensor, Tensor, Tensor)
def _cudnn_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

    grad_out = grad_out.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    # Cannot pass first 5 arguments as kwargs because: https://github.com/pytorch/pytorch/blob/d26ca5de058dbcf56ac52bb43e84dd98df2ace97/torch/_dynamo/variables/torch.py#L1341
    grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp=lse,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        attn_bias=ctx.attn_mask,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=ctx.max_q,
        max_k=ctx.max_k,
        dropout_p=ctx.dropout_p,
        is_causal=ctx.is_causal,
        scale=ctx.scale,
    )
    grad_query, grad_key, grad_value = (x.transpose(1, 2).contiguous() for x in (grad_query, grad_key, grad_value))

    return grad_query, grad_key, grad_value


# Adapted from: https://github.com/Dao-AILab/flash-attention/blob/fd2fc9d85c8e54e5c20436465bca709bc1a6c5a1/flash_attn/flash_attn_interface.py#L807
def _flash_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not yet supported for flash-attn 2.")
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for flash-attn 2.")

    # Hardcoded for now
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    grad_enabled = any(x.requires_grad for x in (query, key, value))

    if scale is None:
        scale = query.shape[-1] ** (-0.5)

    # flash-attn only returns LSE if dropout_p > 0. So, we need to workaround.
    if grad_enabled or (_parallel_config is not None and _parallel_config.context_parallel_config._world_size > 1):
        dropout_p = dropout_p if dropout_p > 0 else 1e-30

    with torch.set_grad_enabled(grad_enabled):
        out, lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            query,
            key,
            value,
            dropout_p,
            scale,
            is_causal,
            window_size[0],
            window_size[1],
            softcap,
            alibi_slopes,
            return_lse,
        )
        lse = lse.permute(0, 2, 1)

    if _save_ctx:
        ctx.save_for_backward(query, key, value, out, lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

    return (out, lse) if return_lse else out


def _flash_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
):
    query, key, value, out, lse, rng_state = ctx.saved_tensors
    grad_query, grad_key, grad_value = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)

    lse_d = _wrapped_flash_attn_backward(  # noqa: F841
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        grad_query,
        grad_key,
        grad_value,
        ctx.dropout_p,
        ctx.scale,
        ctx.is_causal,
        ctx.window_size[0],
        ctx.window_size[1],
        ctx.softcap,
        ctx.alibi_slopes,
        ctx.deterministic,
        rng_state,
    )

    # Head dimension may have been padded
    grad_query = grad_query[..., : grad_out.shape[-1]]
    grad_key = grad_key[..., : grad_out.shape[-1]]
    grad_value = grad_value[..., : grad_out.shape[-1]]

    return grad_query, grad_key, grad_value


def _sage_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not yet supported for Sage attention.")
    if dropout_p > 0.0:
        raise ValueError("`dropout_p` is not yet supported for Sage attention.")
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for Sage attention.")

    out = sageattn(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
        return_lse=return_lse,
    )
    lse = None
    if return_lse:
        out, lse, *_ = out
        lse = lse.permute(0, 2, 1)

    return (out, lse) if return_lse else out


def _sage_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
):
    raise NotImplementedError("Backward pass is not implemented for Sage attention.")


# ===== Context parallel =====


# Reference:
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L827
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L246
# For fullgraph=True tracing compatibility (since FakeTensor does not have a `wait` method):
def _wait_tensor(tensor):
    if isinstance(tensor, funcol.AsyncCollectiveTensor):
        tensor = tensor.wait()
    return tensor


def _all_to_all_single(x: torch.Tensor, group) -> torch.Tensor:
    shape = x.shape
    # HACK: We need to flatten because despite making tensors contiguous, torch single-file-ization
    # to benchmark triton codegen fails somewhere:
    # buf25 = torch.ops._c10d_functional.all_to_all_single.default(buf24, [1, 1], [1, 1], '3')
    # ValueError: Tensors must be contiguous
    x = x.flatten()
    x = funcol.all_to_all_single(x, None, None, group)
    x = x.reshape(shape)
    x = _wait_tensor(x)
    return x


class TemplatedRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        ring_mesh = _parallel_config.context_parallel_config._ring_mesh
        rank = _parallel_config.context_parallel_config._ring_local_rank
        world_size = _parallel_config.context_parallel_config.ring_degree
        next_rank = (rank + 1) % world_size
        prev_out = prev_lse = None

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx.q_shape = query.shape
        ctx.kv_shape = key.shape
        ctx._parallel_config = _parallel_config

        kv_buffer = torch.cat([key.flatten(), value.flatten()]).contiguous()
        kv_buffer = funcol.all_gather_tensor(kv_buffer, gather_dim=0, group=ring_mesh.get_group())
        kv_buffer = kv_buffer.chunk(world_size)

        for i in range(world_size):
            if i > 0:
                kv = kv_buffer[next_rank]
                key_numel = key.numel()
                key = kv[:key_numel].reshape_as(key)
                value = kv[key_numel:].reshape_as(value)
                next_rank = (next_rank + 1) % world_size

            out, lse = forward_op(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                True,
                _save_ctx=i == 0,
                _parallel_config=_parallel_config,
            )

            if _parallel_config.context_parallel_config.convert_to_fp32:
                out = out.to(torch.float32)
                lse = lse.to(torch.float32)

            lse = lse.unsqueeze(-1)
            if prev_out is not None:
                out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
                lse = prev_lse - torch.nn.functional.logsigmoid(prev_lse - lse)
            prev_out = out
            prev_lse = lse

        out = out.to(query.dtype)
        lse = lse.squeeze(-1)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        ring_mesh = ctx._parallel_config.context_parallel_config._ring_mesh
        rank = ctx._parallel_config.context_parallel_config._ring_local_rank
        world_size = ctx._parallel_config.context_parallel_config.ring_degree
        next_rank = (rank + 1) % world_size
        next_ranks = list(range(1, world_size)) + [0]

        accum_dtype = torch.float32 if ctx._parallel_config.context_parallel_config.convert_to_fp32 else grad_out.dtype
        grad_query = torch.zeros(ctx.q_shape, dtype=accum_dtype, device=grad_out.device)
        grad_key = torch.zeros(ctx.kv_shape, dtype=accum_dtype, device=grad_out.device)
        grad_value = torch.zeros(ctx.kv_shape, dtype=accum_dtype, device=grad_out.device)
        next_grad_kv = None

        query, key, value, *_ = ctx.saved_tensors
        kv_buffer = torch.cat([key.flatten(), value.flatten()]).contiguous()
        kv_buffer = funcol.all_gather_tensor(kv_buffer, gather_dim=0, group=ring_mesh.get_group())
        kv_buffer = kv_buffer.chunk(world_size)

        for i in range(world_size):
            if i > 0:
                kv = kv_buffer[next_rank]
                key_numel = key.numel()
                key = kv[:key_numel].reshape_as(key)
                value = kv[key_numel:].reshape_as(value)
                next_rank = (next_rank + 1) % world_size

            grad_query_op, grad_key_op, grad_value_op, *_ = ctx.backward_op(ctx, grad_out)

            if i > 0:
                grad_kv_buffer = _wait_tensor(next_grad_kv)
                grad_key_numel = grad_key.numel()
                grad_key = grad_kv_buffer[:grad_key_numel].reshape_as(grad_key)
                grad_value = grad_kv_buffer[grad_key_numel:].reshape_as(grad_value)

            grad_query += grad_query_op
            grad_key += grad_key_op
            grad_value += grad_value_op

            if i < world_size - 1:
                grad_kv_buffer = torch.cat([grad_key.flatten(), grad_value.flatten()]).contiguous()
                next_grad_kv = funcol.permute_tensor(grad_kv_buffer, next_ranks, group=ring_mesh.get_group())

        grad_query, grad_key, grad_value = (x.to(grad_out.dtype) for x in (grad_query, grad_key, grad_value))

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


class TemplatedUlyssesAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        world_size = _parallel_config.context_parallel_config.ulysses_degree
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        B, S_Q_LOCAL, H, D = query.shape
        _, S_KV_LOCAL, _, _ = key.shape
        H_LOCAL = H // world_size
        query = query.reshape(B, S_Q_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        key = key.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        value = value.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        query, key, value = (_all_to_all_single(x, group) for x in (query, key, value))
        query, key, value = (x.flatten(0, 1).permute(1, 0, 2, 3).contiguous() for x in (query, key, value))

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out = out.reshape(B, world_size, S_Q_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
        out = _all_to_all_single(out, group)
        out = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()

        if return_lse:
            lse = lse.reshape(B, world_size, S_Q_LOCAL, H_LOCAL).permute(1, 3, 0, 2).contiguous()
            lse = _all_to_all_single(lse, group)
            lse = lse.flatten(0, 1).permute(1, 2, 0).contiguous()
        else:
            lse = None

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        ulysses_mesh = ctx._parallel_config.context_parallel_config._ulysses_mesh
        world_size = ctx._parallel_config.context_parallel_config.ulysses_degree
        group = ulysses_mesh.get_group()

        B, S_LOCAL, H, D = grad_out.shape
        H_LOCAL = H // world_size

        grad_out = grad_out.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        grad_out = _all_to_all_single(grad_out, group)
        grad_out = grad_out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()

        grad_query_op, grad_key_op, grad_value_op, *_ = ctx.backward_op(ctx, grad_out)

        grad_query, grad_key, grad_value = (
            x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
            for x in (grad_query_op, grad_key_op, grad_value_op)
        )
        grad_query, grad_key, grad_value = (_all_to_all_single(x, group) for x in (grad_query, grad_key, grad_value))
        grad_query, grad_key, grad_value = (
            x.flatten(0, 1).permute(1, 2, 0, 3).contiguous() for x in (grad_query, grad_key, grad_value)
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


def _templated_context_parallel_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    *,
    forward_op,
    backward_op,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if attn_mask is not None:
        raise ValueError("Attention mask is not yet supported for templated attention.")
    if is_causal:
        raise ValueError("Causal attention is not yet supported for templated attention.")
    if enable_gqa:
        raise ValueError("GQA is not yet supported for templated attention.")

    # TODO: add support for unified attention with ring/ulysses degree both being > 1
    if _parallel_config.context_parallel_config.ring_degree > 1:
        return TemplatedRingAttention.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op,
            backward_op,
            _parallel_config,
        )
    elif _parallel_config.context_parallel_config.ulysses_degree > 1:
        return TemplatedUlyssesAttention.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op,
            backward_op,
            _parallel_config,
        )
    else:
        raise ValueError("Reaching this branch of code is unexpected. Please report a bug.")


# ===== Attention backends =====


@_AttentionBackendRegistry.register(
    AttentionBackendName.FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    lse = None
    if _parallel_config is None:
        out = flash_attn_func(
            q=query,
            k=key,
            v=value,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=return_lse,
        )
        if return_lse:
            out, lse, *_ = out
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            None,
            dropout_p,
            is_causal,
            scale,
            False,
            return_lse,
            forward_op=_flash_attention_forward_op,
            backward_op=_flash_attention_backward_op,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse = out

    return (out, lse) if return_lse else out


@_AttentionBackendRegistry.register(
    AttentionBackendName.FLASH_VARLEN,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _flash_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
        _prepare_for_flash_attn_or_sage_varlen(
            batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
        )
    )

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
        return_attn_probs=return_lse,
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    out, lse = _wrapped_flash_attn_3(
        q=query,
        k=key,
        v=value,
        softmax_scale=scale,
        causal=is_causal,
    )
    return (out, lse) if return_lse else out


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
    _parallel_config: Optional["ParallelConfig"] = None,
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
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
        _prepare_for_flash_attn_or_sage_varlen(
            batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
        )
    )

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
        softmax_scale=scale,
        causal=is_causal,
    )
    out = out.unflatten(0, (batch_size, -1))

    return (out, lse) if return_lse else out


@_AttentionBackendRegistry.register(
    AttentionBackendName.AITER,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _aiter_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if not return_lse and torch.is_grad_enabled():
        # aiter requires return_lse=True by assertion when gradients are enabled.
        out, lse, *_ = aiter_flash_attn_func(
            q=query,
            k=key,
            v=value,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
        )
    else:
        out = aiter_flash_attn_func(
            q=query,
            k=key,
            v=value,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse, *_ = out

    return (out, lse) if return_lse else out


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
    _parallel_config: Optional["ParallelConfig"] = None,
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
    )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName.NATIVE,
    constraints=[_check_device, _check_shape],
    supports_context_parallel=True,
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native attention backend does not support setting `return_lse=True`.")
    if _parallel_config is None:
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
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op=_native_attention_forward_op,
            backward_op=_native_attention_backward_op,
            _parallel_config=_parallel_config,
        )

    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_CUDNN,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    lse = None
    if _parallel_config is None and not return_lse:
        query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
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
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op=_cudnn_attention_forward_op,
            backward_op=_cudnn_attention_backward_op,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse = out

    return (out, lse) if return_lse else out


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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native efficient attention backend does not support setting `return_lse=True`.")
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native flash attention backend does not support setting `return_lse=True`.")
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native math attention backend does not support setting `return_lse=True`.")
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("NPU attention backend does not support setting `return_lse=True`.")
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("XLA attention backend does not support setting `return_lse=True`.")
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
    supports_context_parallel=True,
)
def _sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    lse = None
    if _parallel_config is None:
        out = sageattn(
            q=query,
            k=key,
            v=value,
            tensor_layout="NHD",
            is_causal=is_causal,
            sm_scale=scale,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse, *_ = out
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            None,
            0.0,
            is_causal,
            scale,
            False,
            return_lse,
            forward_op=_sage_attention_forward_op,
            backward_op=_sage_attention_backward_op,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse = out

    return (out, lse) if return_lse else out


@_AttentionBackendRegistry.register(
    AttentionBackendName.SAGE_VARLEN,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _sage_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Sage varlen backend does not support setting `return_lse=True`.")

    batch_size, seq_len_q, _, _ = query.shape
    _, seq_len_kv, _, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
        _prepare_for_flash_attn_or_sage_varlen(
            batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
        )
    )

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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp8_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp8_cuda_sm90(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp16_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp16_triton(
        q=query,
        k=key,
        v=value,
        tensor_layout="NHD",
        is_causal=is_causal,
        sm_scale=scale,
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
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("xformers attention backend does not support setting `return_lse=True`.")

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
