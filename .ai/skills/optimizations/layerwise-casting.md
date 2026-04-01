# Layerwise Casting

## Overview

Layerwise casting stores model weights in a smaller data format (e.g., `torch.float8_e4m3fn`) to use less memory, and upcasts them to a higher precision (e.g., `torch.bfloat16`) on-the-fly during computation. This cuts weight memory roughly in half (bf16 → fp8) with minimal quality impact because normalization and modulation layers are automatically skipped.

This is one of the most effective techniques for fitting a large model on a GPU that's just slightly too small — it doesn't require any special quantization libraries, just PyTorch.

## When to use

- The model **almost** fits in VRAM (e.g., 28GB model on a 32GB GPU)
- You want memory savings with **less speed penalty** than offloading
- You want to **combine with group offloading** for even more savings

## Basic usage

Call `enable_layerwise_casting` on any Diffusers model component:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16)

# Store weights in fp8, compute in bf16
pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)

pipe.to("cuda")
```

The `storage_dtype` controls how weights are stored in memory. The `compute_dtype` controls the precision used during the actual forward pass. Normalization and modulation layers are automatically kept at full precision.

### Supported storage dtypes

| Storage dtype | Memory per param | Quality impact |
|---|---|---|
| `torch.float8_e4m3fn` | 1 byte (vs 2 for bf16) | Minimal for most models |
| `torch.float8_e5m2` | 1 byte | Slightly more range, less precision than e4m3fn |

## Functional API

For more control, use `apply_layerwise_casting` directly. This lets you target specific submodules or customize which layers to skip:

```python
from diffusers.hooks import apply_layerwise_casting

apply_layerwise_casting(
    pipe.transformer,
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
    skip_modules_classes=["norm"],  # skip normalization layers
    non_blocking=True,
)
```

## Combining with other techniques

Layerwise casting is compatible with both group offloading and model CPU offloading. Always apply layerwise casting **before** enabling offloading. See [reduce-memory.md](reduce-memory.md) for code examples and the memory savings formulas for each combination.

## Known limitations

- May not work with all models if the forward implementation contains internal typecasting of weights (assumes forward pass is independent of weight precision)
- May fail with PEFT layers (LoRA). There are some checks but they're not guaranteed for all cases
- Not suitable for training — inference only
- The `compute_dtype` should match what the model expects (usually bf16 or fp16)
