# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
PyTorch utilities: Utilities related to PyTorch
"""

from __future__ import annotations

import functools
import os
from typing import Callable, ParamSpec, TypeVar

from . import logging
from .import_utils import (
    is_torch_available,
    is_torch_neuronx_available,
    is_torch_version,
)


T = TypeVar("T")
P = ParamSpec("P")


if is_torch_available():
    import torch
    from torch.fft import fftn, fftshift, ifftn, ifftshift

    _FP64_UNSUPPORTED_DEVICES = frozenset({"mps", "npu", "neuron"})
    _INT64_UNSUPPORTED_DEVICES = frozenset({"mps", "npu", "neuron"})
    _DTYPE_DOWNCAST = {torch.float64: torch.float32, torch.int64: torch.int32}
    _DTYPE_UNSUPPORTED_DEVICES = {torch.float64: _FP64_UNSUPPORTED_DEVICES, torch.int64: _INT64_UNSUPPORTED_DEVICES}

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls


def maybe_adjust_dtype_for_device(dtype: "torch.dtype", device: "torch.device") -> "torch.dtype":
    unsupported = _DTYPE_UNSUPPORTED_DEVICES.get(dtype)
    return _DTYPE_DOWNCAST[dtype] if unsupported and device.type in unsupported else dtype


def randn_tensor(
    shape: tuple | list,
    generator: list["torch.Generator"] | "torch.Generator" | None = None,
    device: str | "torch.device" | None = None,
    dtype: "torch.dtype" | None = None,
    layout: "torch.layout" | None = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    # Neuron does not support creating random tensors directly on device; always use CPU
    if device.type == "neuron":
        rand_device = torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device.type not in ("mps", "neuron"):
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slightly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def is_compiled_module(module) -> bool:
    """Check whether the module was compiled with torch.compile()"""
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def unwrap_module(module):
    """Unwraps a module if it was compiled with torch.compile()"""
    return module._orig_mod if is_compiled_module(module) else module


def fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int) -> "torch.Tensor":
    """Fourier filter as introduced in FreeU (https://huggingface.co/papers/2309.11497).

    This version of the method comes from here:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """
    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)
    # fftn does not support bfloat16, and produces the experimental ComplexHalf
    # dtype (torch.complex32) when given float16, which is numerically unstable
    # and triggers a UserWarning. Upcast any non-float32 dtype to float32.
    elif x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fftn(x, dim=(-2, -1))
    x_freq = fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = ifftshift(x_freq, dim=(-2, -1))
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)


def apply_freeu(
    resolution_idx: int, hidden_states: "torch.Tensor", res_hidden_states: "torch.Tensor", **freeu_kwargs
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Applies the FreeU mechanism as introduced in https:
    //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

    Args:
        resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
        hidden_states (`torch.Tensor`): Inputs to the underlying block.
        res_hidden_states (`torch.Tensor`): Features from the skip block corresponding to the underlying block.
        s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
        s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
        b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
        b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
    """
    if resolution_idx == 0:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b1"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s1"])
    if resolution_idx == 1:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b2"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s2"])

    return hidden_states, res_hidden_states


def get_torch_cuda_device_capability():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        compute_capability = torch.cuda.get_device_capability(device)
        compute_capability = f"{compute_capability[0]}.{compute_capability[1]}"
        return float(compute_capability)
    else:
        return None


class TorchDeviceBackend:
    """
    A proxy for the `torch.<backend>` namespace (`torch.cuda`, `torch.xpu`, `torch.mps`, ...) of one device. Attributes
    the class does not define are the module's own (`synchronize`, `device_count`, `Stream`, `current_stream`, ...);
    the methods defined here override the operations whose availability differs between backends and need a fallback:
    cache clearing, seeding and memory queries. With no `device`, detects the host accelerator through
    `torch.accelerator`. Raises if torch has no module for the backend rather than silently falling back to
    `torch.cuda`.
    """

    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(self._detect_device_type() if device is None else device)
        self.module = torch.get_device_module(self.device.type)

    @staticmethod
    @functools.lru_cache
    def _detect_device_type() -> str:
        if torch.accelerator.is_available():
            return torch.accelerator.current_accelerator().type

        # Neuron is XLA-based and never registers as a torch accelerator.
        if is_torch_neuronx_available() and hasattr(torch, "neuron") and torch.neuron.is_available():
            return "neuron"
        return "cpu"

    def empty_cache(self) -> None:
        # Backends without a caching allocator (cpu, neuron) have nothing to clear.
        empty_cache = getattr(self.module, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()

    def manual_seed(self, seed: int) -> None:
        # `torch.manual_seed` seeds every device, so it is the correct fallback for backends without their own.
        manual_seed = getattr(self.module, "manual_seed", None)
        if manual_seed is None:
            torch.manual_seed(seed)
            return
        manual_seed(seed)

    def __getattr__(self, name: str):
        # Proxy: anything not overridden here is `torch.<backend>`'s own attribute.
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def _accelerator_serves(self, min_torch_version: str) -> bool:
        # `torch.accelerator` only serves the process accelerator, and its memory API arrived in 2.9 (statistics) and
        # 2.10 (`get_memory_info`).
        current = torch.accelerator.current_accelerator()
        return is_torch_version(">=", min_torch_version) and current is not None and current.type == self.device.type

    def mem_get_info(self) -> tuple[int, int]:
        """Free and total device memory in bytes."""
        mem_get_info = getattr(self.module, "mem_get_info", None)
        if mem_get_info is not None:
            return mem_get_info(self.device.index)
        if self._accelerator_serves("2.10"):
            return torch.accelerator.get_memory_info(self.device)
        raise NotImplementedError(
            f"`torch.{self.device.type}` does not implement `mem_get_info()`, and `torch.accelerator.get_memory_info()` "
            f"cannot serve `{self.device}` on torch {torch.__version__} (requires torch>=2.10 and the current accelerator)."
        )

    def max_memory_allocated(self) -> int:
        """Peak memory allocated on the device in bytes since the last reset; 0 where the backend keeps no statistics."""
        max_memory_allocated = getattr(self.module, "max_memory_allocated", None)
        if max_memory_allocated is not None:
            return max_memory_allocated(self.device.index)
        if self._accelerator_serves("2.9"):
            return torch.accelerator.max_memory_allocated(self.device)
        logger.warning(
            f"`torch.{self.device.type}` keeps no memory statistics on torch {torch.__version__}; "
            "`max_memory_allocated()` returns 0."
        )
        return 0

    def reset_peak_memory_stats(self) -> None:
        reset_peak_memory_stats = getattr(self.module, "reset_peak_memory_stats", None)
        if reset_peak_memory_stats is not None:
            reset_peak_memory_stats(self.device.index)
            return
        if self._accelerator_serves("2.9"):
            torch.accelerator.reset_peak_memory_stats(self.device)
            return
        logger.warning(
            f"`torch.{self.device.type}` keeps no memory statistics on torch {torch.__version__}; "
            "`reset_peak_memory_stats()` is a no-op."
        )


def get_device() -> str:
    return TorchDeviceBackend._detect_device_type()


def empty_device_cache(device_type: str | None = None):
    TorchDeviceBackend(device_type).empty_cache()


# Function-style spellings of `TorchDeviceBackend` for test code.
def backend_manual_seed(device: str, seed: int):
    TorchDeviceBackend(device).manual_seed(seed)


def backend_synchronize(device: str):
    TorchDeviceBackend(device).synchronize()


def backend_empty_cache(device: str):
    TorchDeviceBackend(device).empty_cache()


def backend_device_count(device: str):
    return TorchDeviceBackend(device).device_count()


def backend_reset_peak_memory_stats(device: str):
    TorchDeviceBackend(device).reset_peak_memory_stats()


def backend_reset_max_memory_allocated(device: str):
    # `reset_max_memory_allocated` is CUDA's deprecated alias of `reset_peak_memory_stats`.
    TorchDeviceBackend(device).reset_peak_memory_stats()


def backend_max_memory_allocated(device: str):
    return TorchDeviceBackend(device).max_memory_allocated()


def backend_supports_training(device: str):
    return str(device).split(":")[0] not in ("mps", "neuron")


def enable_full_determinism():
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def disable_full_determinism():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""
    torch.use_deterministic_algorithms(False)


@functools.wraps(functools.lru_cache)
def lru_cache_unless_export(maxsize=128, typed=False):
    def outer_wrapper(fn: Callable[P, T]):
        cached = functools.lru_cache(maxsize=maxsize, typed=typed)(fn)
        if is_torch_version("<", "2.7.0"):
            return cached

        @functools.wraps(fn)
        def inner_wrapper(*args: P.args, **kwargs: P.kwargs):
            compiler = getattr(torch, "compiler", None)
            is_exporting = bool(compiler and hasattr(compiler, "is_exporting") and compiler.is_exporting())
            is_compiling = bool(compiler and hasattr(compiler, "is_compiling") and compiler.is_compiling())

            # Fallback for older builds where compiler.is_compiling is unavailable.
            if not is_compiling:
                dynamo = getattr(torch, "_dynamo", None)
                if dynamo is not None and hasattr(dynamo, "is_compiling"):
                    is_compiling = dynamo.is_compiling()

            if is_exporting or is_compiling:
                return fn(*args, **kwargs)
            return cached(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


if is_torch_available():
    torch_device = get_device()
