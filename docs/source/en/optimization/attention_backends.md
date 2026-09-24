<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Attention backends

> [!NOTE]
> The attention dispatcher is an experimental feature. Please open an issue if you have any feedback or encounter any problems.

Diffusers routes attention through an *attention dispatcher* so you can switch optimized backends behind one API. The dispatcher manages registered implementations and exposes a unified call path for them.

Refer to the table below for an overview of the available attention families and to the [Available backends](#available-backends) section for a more complete list. The fastest backend depends on the model, GPU, and dtype.

| attention family | main feature |
|---|---|
| FlashAttention | minimizes memory reads/writes through tiling and recomputation |
| AI Tensor Engine for ROCm | FlashAttention implementation optimized for AMD ROCm accelerators |
| SageAttention | quantizes attention to int8 |
| FlexAttention | PyTorch FlexAttention |
| PyTorch native | built-in PyTorch implementation using [scaled_dot_product_attention](./fp16#scaled-dot-product-attention) |
| xFormers | memory-efficient attention with support for various attention kernels |

Install each backend’s own package before you enable it. The [Available backends](#available-backends) table lists package requirements Diffusers checks at enable time, plus hardware targets where they matter.

## Set a backend on the model

The [`~ModelMixin.set_attention_backend`] method walks the model’s attention layers and applies the chosen backend on each one. It also sets the dispatcher’s process-wide active backend to the same value.

[`~ModelMixin.reset_attention_backend`] clears the backend on attention layers only. It does not clear the process-wide active backend. For a temporary switch that restores the previous active backend on exit, use the [attention_backend](#try-a-backend-temporarily) context manager.

The example below enables `_flash_3_hub` (FlashAttention-3 from the Hub) with `device_map="cuda"` only. FlashAttention-3 targets Hopper GPUs (for example H100 or H800). Prefer FlashAttention-2 backends such as `flash` or `flash_hub` on Ampere or Ada.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cuda"
)
pipeline.transformer.set_attention_backend("_flash_3_hub")

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

The non-Hub FlashAttention-3 backends (`_flash_3`, `_flash_varlen_3`) require building FlashAttention-3 from source. Prefer `_flash_3_hub` (or `_flash_3_varlen_hub`) when you want the Hub path with Kernels.

## Try a backend temporarily

The [`attention_backend`] context manager sets the process-wide active backend for the duration of the block and restores the previous backend when the block exits. Use it to try a backend for one call without leaving a permanent backend applied from [`~ModelMixin.set_attention_backend`].

```py
import torch
from diffusers import QwenImagePipeline, attention_backend

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cuda"
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

with attention_backend("_flash_3_hub"):
    image = pipeline(prompt).images[0]
```

> [!TIP]
> Most attention backends work with `torch.compile`. Whether that speeds up your pipeline depends on the model and backend. See [Precision and compilation](./fp16).

## Trusting remote kernels

Hub backends need the [Kernels](https://github.com/huggingface/kernels) library first, and then Diffusers fetches the Hub kernel on first use.

Hub attention backends download compute kernels with the [Kernels](https://github.com/huggingface/kernels) library and run them locally. The Hub attention names (`_flash_3_hub`, `flash_hub`, `sage_hub`, and the other `*_hub` backends) resolve to the [kernels-community](https://huggingface.co/kernels-community) organization. That organization is a trusted publisher in Kernels, so Diffusers loads those attention kernels without setting `DIFFUSERS_TRUST_REMOTE_KERNELS`.

Other kernel-backed features such as [GGUF](../quantization/gguf) and [Nunchaku Lite](../quantization/nunchaku) can pull kernels from publishers outside kernels-community. Those paths stay blocked unless you opt in with `DIFFUSERS_TRUST_REMOTE_KERNELS`. When set, Diffusers forwards `trust_remote_code=True` to Kernels so untrusted publishers can load too.

```bash
export DIFFUSERS_TRUST_REMOTE_KERNELS=true
```

Only enable this after inspecting the kernel repository. Without it, loading a kernel from an untrusted publisher raises an error. Diffusers performs this check itself, so it also applies to `kernels<0.14.0`, which predates the `trust_remote_code` argument. Setting `DIFFUSERS_DISABLE_REMOTE_CODE=true` disables remote code globally and takes precedence over `DIFFUSERS_TRUST_REMOTE_KERNELS`.

## Checks

The attention dispatcher can run debugging checks before each dispatched attention call. Which checks run depends on the constraints registered for the active backend.

1. Device checks verify that query, key, and value tensors live on the same device.
2. Data type checks, where registered, confirm matching dtypes and often require `bfloat16` or `float16`.
3. Shape checks validate tensor dimensions and prevent mixing attention masks with causal flags.

Enable checks with the `DIFFUSERS_ATTN_CHECKS` environment variable. Checks add overhead, so they are disabled by default.

```bash
export DIFFUSERS_ATTN_CHECKS=yes
```

With checks on, Diffusers runs those constraints before every dispatched attention call. The low-level example below calls [`dispatch_attention_fn`] directly. Pipeline inference does not need that import. It only needs the backend set via [`~ModelMixin.set_attention_backend`] or [`attention_backend`].

```py
import torch
from diffusers.models.attention_dispatch import attention_backend, dispatch_attention_fn

query = torch.randn(1, 10, 8, 64, dtype=torch.bfloat16, device="cuda")
key = torch.randn(1, 10, 8, 64, dtype=torch.bfloat16, device="cuda")
value = torch.randn(1, 10, 8, 64, dtype=torch.bfloat16, device="cuda")

try:
    with attention_backend("flash"):
        output = dispatch_attention_fn(query, key, value)
        print("✓ Flash Attention works with checks enabled")
except Exception as e:
    print(f"✗ Flash Attention failed: {e}")
```

## Available backends

Refer to the table below for a complete list of available attention backends and their variants. Diffusers checks package availability and version pins when you enable a backend.

| Backend Name | Family | Description | Prerequisite |
|--------------|--------|-------------|--------------|
| `native` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | Default backend using PyTorch's scaled_dot_product_attention | None |
| `flex` | [FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention) | PyTorch FlexAttention | `torch>=2.5.0` |
| `_native_cudnn` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | CuDNN-optimized attention | CUDA + CuDNN |
| `_native_efficient` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | Memory-efficient attention | None beyond PyTorch |
| `_native_flash` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | PyTorch's FlashAttention | CUDA |
| `_native_math` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | Math-based attention (fallback) | None |
| `_native_npu` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | NPU-optimized attention | `torch_npu` |
| `_native_xla` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | XLA-optimized attention | `torch_xla>=2.2` |
| `flash` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-2 | `flash-attn>=2.6.3` |
| `flash_hub` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-2 from Hub kernels | `kernels>=0.12` |
| `flash_varlen` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Variable length FlashAttention | `flash-attn>=2.6.3` |
| `flash_varlen_hub` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Variable length FlashAttention from Hub kernels | `kernels>=0.12` |
| `aiter_fa2_hub` | [AI Tensor Engine for ROCm](https://github.com/ROCm/aiter) | FlashAttention-2 for AMD ROCm from Hub kernels (`bfloat16`) | `kernels>=0.12`, ROCm |
| `flash_4_hub` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-4 from Hub kernels | `kernels>=0.12.3` |
| `_flash_3` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-3 (local; targets Hopper) | Build FA3 from source |
| `_flash_varlen_3` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Variable length FlashAttention-3 (local; targets Hopper) | Build FA3 from source |
| `_flash_3_hub` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-3 from Hub kernels (targets Hopper) | `kernels>=0.12` |
| `_flash_3_varlen_hub` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Variable length FlashAttention-3 from Hub kernels (targets Hopper) | `kernels>=0.12` |
| `sage` | [SageAttention](https://github.com/thu-ml/SageAttention) | Quantized attention (INT8 QK) | `sageattention>=2.1.1` |
| `sage_hub` | [SageAttention](https://github.com/thu-ml/SageAttention) | Quantized attention (INT8 QK) from Hub kernels | `kernels>=0.12` |
| `sage_varlen` | [SageAttention](https://github.com/thu-ml/SageAttention) | Variable length SageAttention | `sageattention>=2.1.1` |
| `_sage_qk_int8_pv_fp8_cuda` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP8 PV (CUDA) | `sageattention>=2.1.1` |
| `_sage_qk_int8_pv_fp8_cuda_sm90` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP8 PV (SM90) | `sageattention>=2.1.1`; SM90 |
| `_sage_qk_int8_pv_fp16_cuda` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP16 PV (CUDA) | `sageattention>=2.1.1` |
| `_sage_qk_int8_pv_fp16_triton` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP16 PV (Triton) | `sageattention>=2.1.1` |
| `xformers` | [xFormers](https://github.com/facebookresearch/xformers) | Memory-efficient attention | `xformers>=0.0.29` |
