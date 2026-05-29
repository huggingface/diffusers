<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# TorchTPU

[TorchTPU](https://github.com/pytorch/tpu) registers a `"tpu"` device type with PyTorch, enabling you to run
diffusers pipelines on Google Cloud TPUs (v4, v5p, v5e, …) with minimal code changes.

Three execution modes are available:

| Mode | How to activate | Speed | Notes |
|---|---|---|---|
| **Lazy** (default) | just `import torch_tpu` | baseline | XLA traces the graph lazily |
| **Eager** | `set_eager_mode(EagerMode.DEFER_NEVER)` | medium | dispatch ops eagerly |
| **Compile** | `pipe.enable_tpu_compile()` | fastest (~4–6×) | static compilation with `TpuBackend` |

## Installation

Follow the [TorchTPU installation guide](https://github.com/pytorch/tpu). After installation,
`import torch_tpu` registers the `"tpu"` device automatically.

## Text encoders always stay on CPU

XLA's static graph compiler does not support certain dynamic ops used in text encoders (notably
`index_select` on large embedding tables). Text encoders must therefore remain on CPU. Their
output embeddings are moved to the TPU after encoding.

Diffusers handles this transparently:
- `_execution_device` detects any component on TPU and returns that device.
- `encode_prompt` runs the text encoder on its own device (`cpu`) and moves the resulting
  embeddings to the execution device (TPU).
- `randn_tensor` generates initial noise on CPU and moves it to TPU, avoiding a TPU RNG
  unaligned DUS (dynamic-update-slice) bug.

## Basic usage (lazy mode)

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

`torch.compile` with `TpuBackend` traces the transformer statically and gives the largest
speedup. The first call (warmup) is slow because it triggers compilation; subsequent calls
reuse the compiled graph.

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

## SDXL

SDXL uses a UNet instead of a transformer. The same approach applies.

```python
import torch
import torch_tpu  # noqa: F401

from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)

pipe.unet.to("tpu")
pipe.vae.to("tpu")

pipe.enable_tpu_compile()
pipe.tpu_warmup(
    prompt="warmup",
    height=1024,
    width=1024,
    num_inference_steps=20,
    guidance_scale=7.5,
)

image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=20,
    guidance_scale=7.5,
).images[0]

image.save("output.png")
```

> [!NOTE]
> In SDXL **lazy/eager mode** (without `enable_tpu_compile`), `time_proj` inside the UNet
> runs on CPU automatically to avoid an XLA unaligned DUS crash. `enable_tpu_compile` uses
> `TpuBackend` which handles the layout internally, so no wrapper is needed in compiled mode.

## Eager mode

Eager mode dispatches ops immediately instead of accumulating a lazy graph. Enter it
**before loading or moving models** to TPU:

```python
import torch
import torch_tpu  # noqa: F401
from torch_tpu._internal.execution_mode import EagerMode, set_eager_mode

eager_ctx = set_eager_mode(EagerMode.DEFER_NEVER)
eager_ctx.__enter__()

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.transformer.to("tpu")
pipe.vae.to("tpu")

image = pipe(prompt="a cat", height=1024, width=1024, num_inference_steps=4, guidance_scale=0.0).images[0]
image.save("output.png")

eager_ctx.__exit__(None, None, None)
```

## Performance benchmarks (v5p, BF16)

| Model | Mode | Steps | Resolution | Time/iter |
|---|---|---|---|---|
| FLUX.2-klein-9B | Lazy | 4 | 1024×1024 | 7.82 s |
| FLUX.2-klein-9B | Compile | 4 | 1024×1024 | 1.94 s |
| ERNIE-Image-Turbo | Lazy | 8 | 1024×1024 | 5.97 s |
| ERNIE-Image-Turbo | Compile | 8 | 1024×1024 | 2.24 s |
| Wan2.2-TI2V (video) | Eager | 50 | 480×832 | 82.2 s |
| Wan2.2-TI2V (video) | Compile | 50 | 480×832 | 14.2 s |

## API reference

### `enable_tpu_compile`

[[autodoc]] diffusers.DiffusionPipeline.enable_tpu_compile

### `tpu_warmup`

[[autodoc]] diffusers.DiffusionPipeline.tpu_warmup
