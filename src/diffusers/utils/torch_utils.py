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
"""
PyTorch utilities: Utilities related to PyTorch
"""

import functools
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

from . import logging
from .import_utils import is_torch_available, is_torch_mlu_available, is_torch_npu_available, is_torch_version


if is_torch_available():
    import torch
    from torch.fft import fftn, fftshift, ifftn, ifftshift

    BACKEND_SUPPORTS_TRAINING = {"cuda": True, "xpu": True, "cpu": True, "mps": False, "default": True}
    BACKEND_EMPTY_CACHE = {
        "cuda": torch.cuda.empty_cache,
        "xpu": torch.xpu.empty_cache,
        "cpu": None,
        "mps": torch.mps.empty_cache,
        "default": None,
    }
    BACKEND_DEVICE_COUNT = {
        "cuda": torch.cuda.device_count,
        "xpu": torch.xpu.device_count,
        "cpu": lambda: 0,
        "mps": lambda: 0,
        "default": 0,
    }
    BACKEND_MANUAL_SEED = {
        "cuda": torch.cuda.manual_seed,
        "xpu": torch.xpu.manual_seed,
        "cpu": torch.manual_seed,
        "mps": torch.mps.manual_seed,
        "default": torch.manual_seed,
    }
    BACKEND_RESET_PEAK_MEMORY_STATS = {
        "cuda": torch.cuda.reset_peak_memory_stats,
        "xpu": getattr(torch.xpu, "reset_peak_memory_stats", None),
        "cpu": None,
        "mps": None,
        "default": None,
    }
    BACKEND_RESET_MAX_MEMORY_ALLOCATED = {
        "cuda": torch.cuda.reset_max_memory_allocated,
        "xpu": getattr(torch.xpu, "reset_peak_memory_stats", None),
        "cpu": None,
        "mps": None,
        "default": None,
    }
    BACKEND_MAX_MEMORY_ALLOCATED = {
        "cuda": torch.cuda.max_memory_allocated,
        "xpu": getattr(torch.xpu, "max_memory_allocated", None),
        "cpu": 0,
        "mps": 0,
        "default": 0,
    }
    BACKEND_SYNCHRONIZE = {
        "cuda": torch.cuda.synchronize,
        "xpu": getattr(torch.xpu, "synchronize", None),
        "cpu": None,
        "mps": None,
        "default": None,
    }
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls


# This dispatches a defined function according to the accelerator from the function definitions.
def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):
    if device not in dispatch_table:
        return dispatch_table["default"](*args, **kwargs)

    fn = dispatch_table[device]

    # Some device agnostic functions return values. Need to guard against 'None' instead at
    # user level
    if not callable(fn):
        return fn

    return fn(*args, **kwargs)


# These are callables which automatically dispatch the function specific to the accelerator
def backend_manual_seed(device: str, seed: int):
    return _device_agnostic_dispatch(device, BACKEND_MANUAL_SEED, seed)


def backend_synchronize(device: str):
    return _device_agnostic_dispatch(device, BACKEND_SYNCHRONIZE)


def backend_empty_cache(device: str):
    return _device_agnostic_dispatch(device, BACKEND_EMPTY_CACHE)


def backend_device_count(device: str):
    return _device_agnostic_dispatch(device, BACKEND_DEVICE_COUNT)


def backend_reset_peak_memory_stats(device: str):
    return _device_agnostic_dispatch(device, BACKEND_RESET_PEAK_MEMORY_STATS)


def backend_reset_max_memory_allocated(device: str):
    return _device_agnostic_dispatch(device, BACKEND_RESET_MAX_MEMORY_ALLOCATED)


def backend_max_memory_allocated(device: str):
    return _device_agnostic_dispatch(device, BACKEND_MAX_MEMORY_ALLOCATED)


# These are callables which return boolean behaviour flags and can be used to specify some
# device agnostic alternative where the feature is unsupported.
def backend_supports_training(device: str):
    if not is_torch_available():
        return False

    if device not in BACKEND_SUPPORTS_TRAINING:
        device = "default"

    return BACKEND_SUPPORTS_TRAINING[device]


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
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

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
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
    # fftn does not support bfloat16
    elif x.dtype == torch.bfloat16:
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
) -> Tuple["torch.Tensor", "torch.Tensor"]:
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


@functools.lru_cache
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif is_torch_npu_available():
        return "npu"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_mlu_available():
        return "mlu"
    else:
        return "cpu"


def empty_device_cache(device_type: Optional[str] = None):
    if device_type is None:
        device_type = get_device()
    if device_type in ["cpu"]:
        return
    device_mod = getattr(torch, device_type, torch.cuda)
    device_mod.empty_cache()


def device_synchronize(device_type: Optional[str] = None):
    if device_type is None:
        device_type = get_device()
    device_mod = getattr(torch, device_type, torch.cuda)
    device_mod.synchronize()


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


if is_torch_available():
    torch_device = get_device()
