# torch.compile

## Overview

`torch.compile` traces a model's forward pass and compiles it to optimized machine code (via Triton or other backends). For diffusers, it typically speeds up the denoising loop by 20-50% after a warmup period.

## Full model compilation

Compile individual components, not the whole pipeline:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16).to("cuda")

pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
# Optionally compile the VAE decoder too
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)
```

The first 1-3 inference calls are slow (compilation/warmup). Subsequent calls are fast. Always do a warmup run before benchmarking.

## Regional compilation (preferred)

Regional compilation compiles only the frequently repeated sub-modules (transformer blocks) instead of the whole model. It provides the same runtime speedup but with ~8-10x faster compile time and better compatibility with offloading.

Diffusers models declare their repeated blocks via the `_repeated_blocks` class attribute (a list of class name strings). Most modern transformers define this:

```python
# FluxTransformer defines:
_repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
```

Use `compile_repeated_blocks()` to compile them:

```python
pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.compile_repeated_blocks(fullgraph=True)
```

**Always guard before calling** — raises `ValueError` if `_repeated_blocks` is empty or the named classes aren't found. Use this pattern universally, whether or not you're using offloading:

```python
# Works with or without enable_model_cpu_offload() / enable_group_offload()
if getattr(pipe.transformer, "_repeated_blocks", None):
    pipe.transformer.compile_repeated_blocks(fullgraph=True)
else:
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
```

`torch.compile` is compatible with diffusers' offloading methods — the offloading hooks use `@torch.compiler.disable()` on device-transfer operations so they run natively outside the compiled graph. Regional compilation is preferred when combining with offloading because it avoids compiling the parts that interact with the hooks.

Models with `_repeated_blocks` defined include: Flux, Flux2, HunyuanVideo, LTX2Video, Wan, CogVideo, SD3, UNet2DConditionModel, and most other modern architectures.

## Compile modes

| Mode | Speed gain | Compile time | Notes |
|---|---|---|---|
| `"default"` | Moderate | Fast | Safe starting point |
| `"reduce-overhead"` | Good | Moderate | Reduces Python overhead via CUDA graphs |
| `"max-autotune"` | Best | Very slow | Tries many kernel configs; best for repeated inference |

## `fullgraph=True`

Requires the entire forward pass to be compilable as a single graph. Most diffusers transformers support this. If you get a `torch._dynamo` graph break error, remove `fullgraph=True` to allow partial compilation.

## Limitations

- **Dynamic shapes**: Changing resolution between calls triggers recompilation. Use `torch.compile(..., dynamic=True)` for variable resolutions, at some speed cost.
- **First call is slow**: Budget 1-3 minutes for initial compilation depending on model size.
- **Windows**: `reduce-overhead` and `max-autotune` modes may have issues. Use `"default"` if you hit errors.
