# Attention Backends

## Overview

Diffusers supports multiple attention backends through `dispatch_attention_fn`. The backend affects both speed and memory usage. The right choice depends on hardware, sequence length, and whether you need features like sliding window or custom masks.

## Available backends

| Backend | Key requirement | Best for |
|---|---|---|
| `torch_sdpa` (default) | PyTorch >= 2.0 | General use; auto-selects FlashAttention or memory-efficient kernels |
| `flash_attention_2` | `flash-attn` package, Ampere+ GPU | Long sequences, training, best raw throughput |
| `xformers` | `xformers` package | Older GPUs, memory-efficient attention |
| `flex_attention` | PyTorch >= 2.5 | Custom attention masks, block-sparse patterns |
| `sage_attention` | `sageattention` package | INT8 quantized attention for inference speed |

## How to set the backend

```python
# Global default
from diffusers import set_attention_backend
set_attention_backend("flash_attention_2")

# Per-model
pipe.transformer.set_attn_processor(AttnProcessor2_0())  # torch_sdpa

# Via environment variable
# DIFFUSERS_ATTENTION_BACKEND=flash_attention_2
```

## Debugging attention issues

- **NaN outputs**: Check if your attention mask dtype matches the expected dtype. Some backends require `bool`, others require float masks with `-inf` for masked positions.
- **Speed regression**: Profile with `torch.profiler` to verify the expected kernel is actually being dispatched. SDPA can silently fall back to the math kernel.
- **Memory spike**: FlashAttention-2 is memory-efficient for long sequences but has overhead for very short ones. For short sequences, `torch_sdpa` with math fallback may use less memory.

## Implementation notes

- Models integrated into diffusers should use `dispatch_attention_fn` (not `F.scaled_dot_product_attention` directly) so that backend switching works automatically.
- See the attention pattern in the `model-integration` skill for how to implement this in new models.
