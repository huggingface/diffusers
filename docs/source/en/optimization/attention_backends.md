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

Diffusers provides several optimized attention algorithms that are more memory and computationally efficient through it's *attention dispatcher*. The dispatcher acts as a router for managing and switching between different attention implementations and provides a unified interface for interacting with them.

Refer to the table below for an overview of the available attention families and to the [Available backends](#available-backends) section for a more complete list.

| attention family | main feature |
|---|---|
| FlashAttention | minimizes memory reads/writes through tiling and recomputation |
| SageAttention | quantizes attention to int8 |
| PyTorch native | built-in PyTorch implementation using [scaled_dot_product_attention](./fp16#scaled-dot-product-attention) |
| xFormers | memory-efficient attention with support for various attention kernels |

This guide will show you how to set and use the different attention backends.

## set_attention_backend

The [`~ModelMixin.set_attention_backend`] method iterates through all the modules in the model and sets the appropriate attention backend to use. The attention backend setting persists until [`~ModelMixin.reset_attention_backend`] is called.

The example below demonstrates how to enable the `_flash_3_hub` implementation for FlashAttention-3 from the [kernel](https://github.com/huggingface/kernels) library, which allows you to instantly use optimized compute kernels from the Hub without requiring any setup.

> [!NOTE]
> FlashAttention-3 is not supported for non-Hopper architectures, in which case, use FlashAttention with `set_attention_backend("flash")`.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipeline.transformer.set_attention_backend("_flash_3_hub")

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

To restore the default attention backend, call [`~ModelMixin.reset_attention_backend`].

```py
pipeline.transformer.reset_attention_backend()
```

## attention_backend context manager

The [attention_backend](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L225) context manager temporarily sets an attention backend for a model within the context. Outside the context, the default attention (PyTorch's native scaled dot product attention) is used. This is useful if you want to use different backends for different parts of a pipeline or if you want to test the different backends.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

with attention_backend("_flash_3_hub"):
    image = pipeline(prompt).images[0]
```

> [!TIP]
> Most attention backends support `torch.compile` without graph breaks and can be used to further speed up inference.

## Checks

The attention dispatcher includes debugging checks that catch common errors before they cause problems.

1. Device checks verify that query, key, and value tensors live on the same device.
2. Data type checks confirm tensors have matching dtypes and use either bfloat16 or float16.
3. Shape checks validate tensor dimensions and prevent mixing attention masks with causal flags.

Enable these checks by setting the `DIFFUSERS_ATTN_CHECKS` environment variable. Checks add overhead to every attention operation, so they're disabled by default. 

```bash
export DIFFUSERS_ATTN_CHECKS=yes
```

The checks are run now before every attention operation.

```py
import torch

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

You can also configure the registry directly.

```py
from diffusers.models.attention_dispatch import _AttentionBackendRegistry

_AttentionBackendRegistry._checks_enabled = True
```

## Available backends

Refer to the table below for a complete list of available attention backends and their variants.

<details>
<summary>Expand</summary>

| Backend Name | Family | Description |
|--------------|--------|-------------|
| `native` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | Default backend using PyTorch's scaled_dot_product_attention |
| `flex` | [FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention) | PyTorch FlexAttention implementation |
| `_native_cudnn` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | CuDNN-optimized attention |
| `_native_efficient` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | Memory-efficient attention |
| `_native_flash` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | PyTorch's FlashAttention |
| `_native_math` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | Math-based attention (fallback) |
| `_native_npu` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | NPU-optimized attention |
| `_native_xla` | [PyTorch native](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | XLA-optimized attention |
| `flash` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-2 |
| `flash_varlen` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Variable length FlashAttention |
| `_flash_3` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-3 |
| `_flash_varlen_3` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Variable length FlashAttention-3 |
| `_flash_3_hub` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FlashAttention-3 from kernels |
| `sage` | [SageAttention](https://github.com/thu-ml/SageAttention) | Quantized attention (INT8 QK) |
| `sage_varlen` | [SageAttention](https://github.com/thu-ml/SageAttention) | Variable length SageAttention |
| `_sage_qk_int8_pv_fp8_cuda` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP8 PV (CUDA) |
| `_sage_qk_int8_pv_fp8_cuda_sm90` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP8 PV (SM90) |
| `_sage_qk_int8_pv_fp16_cuda` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP16 PV (CUDA) |
| `_sage_qk_int8_pv_fp16_triton` | [SageAttention](https://github.com/thu-ml/SageAttention) | INT8 QK + FP16 PV (Triton) |
| `xformers` | [xFormers](https://github.com/facebookresearch/xformers) | Memory-efficient attention |

</details>