# Reduce Memory

## Documentation

Read the full guide for usage examples of every technique:

- **Local:** `docs/source/en/optimization/memory.md`
- **Online:** https://huggingface.co/docs/diffusers/main/en/optimization/memory

## Compatibility rules

**Incompatible combinations (will raise `ValueError`):**
- `enable_model_cpu_offload()` on a pipeline where **any** component has group offloading enabled
- `enable_sequential_cpu_offload()` on a pipeline where **any** component has group offloading enabled

**Fix:** If you need pipeline-wide offloading, use `pipe.enable_group_offload(...)` instead of `enable_model_cpu_offload()`. The pipeline-level call supports `exclude_modules=["small_component"]` to keep specific components on GPU.

**Compatible combinations:**
- Group offloading (pipeline-level) + VAE tiling
- Model CPU offloading + VAE tiling
- Layerwise casting + group offloading — apply casting **first**
- Layerwise casting + model CPU offloading — apply casting **first**
- Quantization + model CPU offloading
- Per-component group offloading with different configs (e.g. `block_level` for transformer, `leaf_level` for VAE)

## Deprecated APIs (since v0.40.0)

Use the component-level APIs — the pipeline-level wrappers are deprecated:

| Deprecated | Use instead |
|---|---|
| `pipe.enable_vae_slicing()` | `pipe.vae.enable_slicing()` |
| `pipe.enable_vae_tiling()` | `pipe.vae.enable_tiling()` |

## VAE tiling — only add when needed

VAE tiling is a VRAM optimization, **not** a default best practice. It adds processing overhead. Only enable it when the VAE decode/encode step would OOM without it:

- **Image models:** typically needed above ~1.5 MP on ≤16 GB GPUs, or ~4 MP on ≤32 GB GPUs
- **Video models:** when `H × W × num_frames` is large relative to remaining VRAM after denoising

VAEs that do NOT support tiling: `AutoencoderKLWan`, `AsymmetricAutoencoderKL`.

**Tip:** When combining VAE tiling with group offloading (`use_stream=True`), do one dummy forward pass first — otherwise you may hit device mismatch errors from the stream prefetch interacting with the tiling state.

## VAE slicing — only for batch size > 1

`pipe.vae.enable_slicing()` reduces VAE peak memory for batched generation only. No benefit for single-image generation.

VAEs that do NOT support slicing: `AutoencoderKLWan`, `AsymmetricAutoencoderKL`.

## Group offloading — full parameter reference

Parameters available across the three group offloading APIs (`pipe.enable_group_offload`, `component.enable_group_offload`, `apply_group_offloading`):

| Parameter | Pipeline | Model | `apply_group_offloading` | Description |
|---|---|---|---|---|
| `onload_device` | yes | yes | yes | Device to load layers onto for computation |
| `offload_device` | yes | yes | yes | Device to offload layers to when idle (default: CPU) |
| `offload_type` | yes | yes | yes | `"block_level"` or `"leaf_level"` |
| `num_blocks_per_group` | yes | yes | yes | Required for `block_level` — layers per group |
| `non_blocking` | yes | yes | yes | Non-blocking device transfer |
| `use_stream` | yes | yes | yes | Overlap transfer/compute via CUDA streams (~2-3× RAM of model) |
| `record_stream` | yes | yes | yes | With `use_stream`: faster, slightly more VRAM |
| `low_cpu_mem_usage` | yes | yes | yes | With `use_stream`: pin on-the-fly instead of pre-pinning (saves RAM, slower) |
| `offload_to_disk_path` | yes | yes | yes | Offload to disk instead of CPU RAM |
| `exclude_modules` | **pipeline only** | — | — | Component names to keep on `onload_device` |
| `block_modules` | — | **yes** | **yes** | Override which submodules count as blocks |
| `exclude_kwargs` | — | **yes** | **yes** | Kwarg keys not to move between devices |

`use_stream=True` forces `num_blocks_per_group=1` for `block_level` offloading (a warning is raised otherwise).

Non-diffusers components (e.g. transformers text encoders) must use the functional `apply_group_offloading` API rather than `.enable_group_offload()`.

## Sequential CPU offloading

`enable_sequential_cpu_offload()` is a legacy API. Group offloading with `leaf_level + use_stream=True` does the same thing but faster. Prefer that.

## Debugging OOM

1. Identify which stage OOMs: loading, text encoding, denoising, or VAE decode
2. OOM during `.to("cuda")` → full pipeline doesn't fit; use model CPU offloading or group offloading
3. OOM during denoising with model CPU offloading → the transformer alone exceeds VRAM; use layerwise casting or group offloading instead
4. OOM during VAE decode → add `pipe.vae.enable_tiling()`
5. Still OOM → see [quantization.md](quantization.md) and [memory-calculator.md](memory-calculator.md)
