<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Attention backends

Diffusers provides several optimized attention algorithms that are more memory and computationally efficient through it's *attention dispatcher*. The dispatcher acts as a router for managing and switching between different attention implementations and provides a unified interface for interacting with them.

Available attention implementations include the following.

| attention family | main feature |
|---|---|
| FlashAttention | minimizes memory reads/writes through tiling and recomputation |
| SageAttention | quantizes attention to int8 |
| PyTorch native | built-in PyTorch implementation using [scaled_dot_product_attention](./fp16#scaled-dot-product-attention) |
| xFormers | memory-efficient attention with support for various attention kernels |

This guide will show you how to use the dispatcher to set and use the different attention backends.

## FlashAttention

[FlashAttention](https://github.com/Dao-AILab/flash-attention) reduces memory traffic by making better use of on-chip shared memory (SRAM) instead of global GPU memory so the data doesn't have to travel far. The latest variant, FlashAttention-3, is further optimized for modern GPUs (Hopper/Blackwell) and also overlaps computations and handles FP8 attention better.

There are several available FlashAttention variants, including variable length and the original FlashAttention. For a full list of supported implementations, check the list [here](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L163).

The example below demonstrates how to enable the `_flash_3_hub` implementation. The [kernel](https://github.com/huggingface/kernels) library allows you to instantly use optimized compute kernels from the Hub without requiring any setup.

Pass the attention backend to the [`~ModelMixin.set_attention_backend`] method.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipeline.transformer.set_attention_backend("_flash_3_hub")
```

You could also use the [attention_backend](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L225) context manager to temporarily set an attention backend for a model within the context.

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

To restore the default attention backend, call [`~ModelMixin.reset_attention_backend`].

```py
pipeline.transformer.reset_attention_backend()
```

## SageAttention

[SageAttention](https://github.com/thu-ml/SageAttention) quantizes attention by computing queries (Q) and keys (K) in INT8. The probability (P) and value (V) are calculated in either FP8 or FP16 to minimize error. This significantly increases inference throughput and with little to no degradation.

There are several SageAttention variants for FP8 and FP16 as well as whether it is CUDA or Triton based. For a full list of supported implementations, check the list [here](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L182).

The example below uses the `_sage_qk_int8_pv_fp8_cuda` implementation.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipeline.transformer.set_attention_backend("_sage_qk_int8_pv_fp8_cuda")
```

You could also use the [attention_backend](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L225) context manager to temporarily set an attention backend for a model within the context.

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

with attention_backend("_sage_qk_int8_pv_fp8_cuda"):
    image = pipeline(prompt).images[0]
```

To restore the default attention backend, call [`~ModelMixin.reset_attention_backend`].

```py
pipeline.transformer.reset_attention_backend()
```

## PyTorch native

PyTorch includes a [native implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) of several optimized attention implementations including [FlexAttention](https://pytorch.org/blog/flexattention/), FlashAttention, memory-efficient attention, and a C++ version.

For a full list of supported implementations, check the list [here](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L171).

The example below uses the `_native_flash` implementation.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipeline.transformer.set_attention_backend("_native_flash")
```

You could also use the [attention_backend](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L225) context manager to temporarily set an attention backend for a model within the context.

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

with attention_backend("_native_flash"):
    image = pipeline(prompt).images[0]
```

To restore the default attention backend, call [`~ModelMixin.reset_attention_backend`].

```py
pipeline.transformer.reset_attention_backend()
```

## xFormers

[xFormers](https://github.com/facebookresearch/xformers) provides memory-efficient attention algorithms such as sparse attention and block-sparse attention. Pass `xformers` to enable it.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipeline.transformer.set_attention_backend("xformers")
```

You could also use the [attention_backend](https://github.com/huggingface/diffusers/blob/5e181eddfe7e44c1444a2511b0d8e21d177850a0/src/diffusers/models/attention_dispatch.py#L225) context manager to temporarily set an attention backend for a model within the context.

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

with attention_backend("xformers"):
    image = pipeline(prompt).images[0]
```

To restore the default attention backend, call [`~ModelMixin.reset_attention_backend`].

```py
pipeline.transformer.reset_attention_backend()
```
