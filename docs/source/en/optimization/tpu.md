<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# TorchTPU

[TorchTPU](https://github.com/google-pytorch/torch_tpu/) provides a PyTorch backend for Google's Tensor Processing Units (TPUs), enabling you to run diffusers pipelines on Google Cloud TPUs (v6e, v5p, …) with minimal code changes.

Four execution modes are available:

| Mode | Constant | How to activate | Notes |
|---|---|---|---|
| **Strict Eager** (default) | `EagerMode.DEFER_NEVER` | just `import torch_tpu` | Operations dispatched one at a time, asynchronous |
| **Debug Eager** | `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING` | `set_eager_mode(EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING)` or `TPU_LAUNCH_BLOCKING=1` | Synchronous execution; useful for pinpointing errors |
| **Fused Eager** | `EagerMode.DEFER_AND_FUSE` | `set_eager_mode(EagerMode.DEFER_AND_FUSE)` or `TPU_DEFER_AND_FUSE=1` | Groups multiple ops for XLA fusion; best throughput in eager mode |
| **Compile** | — | `pipe.enable_tpu_compile()` | AOT compilation with `TpuBackend` |

## Installation

Follow the [TorchTPU installation guide](https://github.com/google-pytorch/torch_tpu/). After installation,
`import torch_tpu` registers the `"tpu"` device automatically.

## Basic usage (strict eager mode)

```python
import torch
import torch_tpu  # noqa: F401 — registers torch.tpu

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)

# Move only the denoising components to TPU; text encoders stay on CPU.
pipe.transformer.to("tpu")
pipe.vae.to("tpu")

# _execution_device is now "tpu" automatically.
image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]

image.save("output.png")
```

## Compiled mode (recommended for production)

`torch.compile` with `TpuBackend` traces the transformer statically. The first call (warmup)
is slow because it triggers compilation; subsequent calls reuse the compiled graph.

> [!IMPORTANT]
> TorchTPU requires **static shapes** — `torch.compile` is called with `dynamic=False`
> internally. Every time `height`, `width`, or `num_inference_steps` changes, the graph is
> recompiled from scratch. Keep these values constant across all calls after warmup, or call
> `tpu_warmup` again before changing them.

```python
import torch
import torch_tpu  # noqa: F401

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe.transformer.to("tpu")
pipe.vae.to("tpu")

# Compile TPU components with TpuBackend.
# Also applies AttnProcessor to replace SDP-based attention (required for XLA).
pipe.enable_tpu_compile()

# Warmup — triggers static graph compilation.
pipe.tpu_warmup(
    prompt="warmup",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
)

# Timed inference reuses the compiled graph.
image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]

image.save("output.png")
```

## Eager mode

TorchTPU defaults to **Strict Eager** (`EagerMode.DEFER_NEVER`): operations are dispatched one
at a time asynchronously, matching standard PyTorch GPU behaviour. Two alternative eager modes
are available:

**Debug Eager** — synchronous execution; every op blocks until the TPU finishes. Useful for
pinpointing the exact line that raises an error. Equivalent to `CUDA_LAUNCH_BLOCKING=1` on GPU.

```python
from torch_tpu._internal import execution_mode as em

# Globally for the session
em.eager_mode = em.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING

# Or via environment variable (before importing torch_tpu):
# TPU_LAUNCH_BLOCKING=1
```

**Fused Eager** — defers ops and lets the XLA compiler fuse across operation boundaries,
reducing memory traffic and dispatch overhead without full AOT compilation.

```python
from torch_tpu._internal import execution_mode as em

# Globally for the session
em.eager_mode = em.EagerMode.DEFER_AND_FUSE

# Or via environment variable (before importing torch_tpu):
# TPU_DEFER_AND_FUSE=1
```

Use `set_eager_mode` as a context manager to switch modes for a single block:

```python
from torch_tpu._internal import execution_mode as em

with em.set_eager_mode(em.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING):
    # synchronous — pinpoints the exact failing line
    output = model(input_data)
```

> [!TIP]
> For the best production throughput, prefer `torch.compile` via `pipe.enable_tpu_compile()`,
> which uses an Ahead-of-Time (AOT) strategy more aggressive than Fused Eager.

## API reference

### `enable_tpu_compile`

[[autodoc]] diffusers.DiffusionPipeline.enable_tpu_compile

### `tpu_warmup`

[[autodoc]] diffusers.DiffusionPipeline.tpu_warmup
