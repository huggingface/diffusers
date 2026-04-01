# Reduce Memory

## Overview

Large diffusion models can exceed GPU VRAM. Diffusers provides several techniques to reduce peak memory, each with different speed/memory tradeoffs.

## Techniques (ordered by ease of use)

### 1. Model CPU offloading

Moves entire models to CPU when not in use, loads them to GPU just before their forward pass.

```python
pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
# Do NOT call pipe.to("cuda") — the hook handles device placement
```

- **Memory savings**: Significant — only one model on GPU at a time
- **Speed cost**: Moderate — full model transfers between CPU and GPU
- **When to use**: First thing to try when hitting OOM
- **Limitation**: If the single largest component (e.g. transformer) exceeds VRAM, this won't help — you need group offloading or layerwise casting instead.

### 2. Group offloading

Offloads groups of internal layers to CPU, loading them to GPU only during their forward pass. More granular than model offloading, faster than sequential offloading.

**Two offload types:**
- `block_level` — offloads groups of N layers at a time. Lower memory, moderate speed.
- `leaf_level` — offloads individual leaf modules. Equivalent to sequential offloading but can be made faster with CUDA streams.

**IMPORTANT**: `enable_model_cpu_offload()` will raise an error if any component has group offloading enabled. If you need offloading for the whole pipeline, use pipeline-level `enable_group_offload()` instead — it handles all components in one call.

#### Pipeline-level group offloading

Applies group offloading to ALL components in the pipeline at once. Simplest approach.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16)

# Option A: leaf_level with CUDA streams (recommended — fast + low memory)
pipe.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
)

# Option B: block_level (more memory savings, slower)
pipe.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2,
)
```

#### Component-level group offloading

Apply group offloading selectively to specific components. Useful when only the transformer is too large for VRAM but other components fit fine.

For Diffusers model components (inheriting from `ModelMixin`), use `enable_group_offload`:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16)

# Group offload the transformer (the largest component)
pipe.transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
)

# Group offload the VAE too if needed
pipe.vae.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_type="leaf_level",
)
```

For non-Diffusers components (e.g. text encoders from transformers library), use the functional API:

```python
from diffusers.hooks import apply_group_offloading

apply_group_offloading(
    pipe.text_encoder,
    onload_device=torch.device("cuda"),
    offload_type="block_level",
    num_blocks_per_group=2,
)
```

#### CUDA streams for faster group offloading

When `use_stream=True`, the next layer is prefetched to GPU while the current layer runs. This overlaps data transfer with computation. Requires ~2x CPU memory of the model.

```python
pipe.transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,  # slightly more speed, slightly more memory
)
```

If using `block_level` with `use_stream=True`, set `num_blocks_per_group=1` (a warning is raised otherwise).

#### Full parameter reference

Parameters available across the three group offloading APIs:

| Parameter | Pipeline | Model | `apply_group_offloading` | Description |
|---|---|---|---|---|
| `onload_device` | yes | yes | yes | Device to load layers onto for computation (e.g. `torch.device("cuda")`) |
| `offload_device` | yes | yes | yes | Device to offload layers to when idle (default: `torch.device("cpu")`) |
| `offload_type` | yes | yes | yes | `"block_level"` (groups of N layers) or `"leaf_level"` (individual modules) |
| `num_blocks_per_group` | yes | yes | yes | Required for `block_level` — how many layers per group |
| `non_blocking` | yes | yes | yes | Non-blocking data transfer between devices |
| `use_stream` | yes | yes | yes | Overlap data transfer and computation via CUDA streams. Requires ~2x CPU RAM of the model |
| `record_stream` | yes | yes | yes | With `use_stream`, marks tensors for stream. Faster but slightly more memory |
| `low_cpu_mem_usage` | yes | yes | yes | Pins tensors on-the-fly instead of pre-pinning. Saves CPU RAM when using streams, but slower |
| `offload_to_disk_path` | yes | yes | yes | Path to offload weights to disk instead of CPU RAM. Useful when system RAM is also limited |
| `exclude_modules` | **yes** | no | no | Pipeline-only: list of component names to skip (they get placed on `onload_device` instead) |
| `block_modules` | no | **yes** | **yes** | Override which submodules are treated as blocks for `block_level` offloading |
| `exclude_kwargs` | no | **yes** | **yes** | Kwarg keys that should not be moved between devices (e.g. mutable cache state) |

### 3. Sequential CPU offloading

Moves individual layers to GPU one at a time during forward pass.

```python
pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
# Do NOT call pipe.to("cuda") first — saves minimal memory if you do
```

- **Memory savings**: Maximum — only one layer on GPU at a time
- **Speed cost**: Very high — many small transfers per forward pass
- **When to use**: Last resort when group offloading with streams isn't enough
- **Note**: Group offloading with `leaf_level` + `use_stream=True` is essentially the same idea but faster. Prefer that.

### 4. VAE slicing

Processes VAE encode/decode in slices along the batch dimension.

```python
pipe.vae.enable_slicing()
```

- **Memory savings**: Reduces VAE peak memory for batch sizes > 1
- **Speed cost**: Minimal
- **When to use**: When generating multiple images/videos in a batch
- **Note**: `AutoencoderKLWan` and `AsymmetricAutoencoderKL` don't support slicing.
- **API note**: The pipeline-level `pipe.enable_vae_slicing()` is deprecated since v0.40.0. Use `pipe.vae.enable_slicing()`.

### 5. VAE tiling

Processes VAE encode/decode in spatial tiles. This is a **VRAM optimization** — only use when the VAE decode/encode would OOM without it.

```python
pipe.vae.enable_tiling()
```

- **Memory savings**: Bounds VAE peak memory by tile size rather than full resolution
- **Speed cost**: Some overhead from tile overlap processing
- **When to use** (only when VAE decode would OOM):
  - **Image models**: Typically needed above ~1.5 MP on ≤16 GB GPUs, or ~4 MP on ≤32 GB GPUs
  - **Video models**: When `H × W × num_frames` is large relative to remaining VRAM after denoising
- **When NOT to use**: At standard resolutions where the VAE fits comfortably — tiling adds overhead for no benefit
- **Note**: `AutoencoderKLWan` and `AsymmetricAutoencoderKL` don't support tiling.
- **API note**: The pipeline-level `pipe.enable_vae_tiling()` is deprecated since v0.40.0. Use `pipe.vae.enable_tiling()`.
- **Tip for group offloading with streams**: If combining VAE tiling with group offloading (`use_stream=True`), do a dummy forward pass first to avoid device mismatch errors.

### 6. Attention slicing (legacy)

```python
pipe.enable_attention_slicing()
```

- Largely superseded by `torch_sdpa` and FlashAttention
- Still useful on very old GPUs without SDPA support

## Combining techniques

Compatible combinations:
- Group offloading (pipeline-level) + VAE tiling — good general setup
- Group offloading (pipeline-level, `exclude_modules=["small_component"]`) — keeps small models on GPU, offloads large ones
- Model CPU offloading + VAE tiling — simple and effective when the largest component fits in VRAM
- Layerwise casting + group offloading — maximum savings (see [layerwise-casting.md](layerwise-casting.md))
- Layerwise casting + model CPU offloading — also works
- Quantization + model CPU offloading — works well
- Per-component group offloading with different configs — e.g. `block_level` for transformer, `leaf_level` for VAE

**Incompatible combinations:**
- `enable_model_cpu_offload()` on a pipeline where ANY component has group offloading — raises ValueError
- `enable_sequential_cpu_offload()` on a pipeline where ANY component has group offloading — same error

## Debugging OOM

1. Check which stage OOMs: loading, encoding, denoising, or decoding
2. If OOM during `.to("cuda")` — the full pipeline doesn't fit. Use model CPU offloading or group offloading
3. If OOM during denoising with model CPU offloading — the transformer alone exceeds VRAM. Use layerwise casting (see [layerwise-casting.md](layerwise-casting.md)) or group offloading instead
4. If still OOM during VAE decode, add `pipe.vae.enable_tiling()`
5. Consider quantization (see [quantization.md](quantization.md)) as a complementary approach
