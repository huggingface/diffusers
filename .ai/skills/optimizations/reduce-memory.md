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

## Group offloading — parameters not in the optimization guide

For the full parameter reference, see `docs/source/en/optimization/memory.md` / https://huggingface.co/docs/diffusers/main/en/optimization/memory#group-offloading.

Three parameters covered only in the API docstrings, not in the guide:

- **`exclude_modules`** (pipeline-level only) — list of component names to keep on `onload_device` instead of offloading
- **`block_modules`** (model/`apply_group_offloading` only) — override which submodules are treated as blocks for `block_level` offloading
- **`exclude_kwargs`** (model/`apply_group_offloading` only) — kwarg keys that should not be moved between devices (e.g. mutable cache state)

Non-diffusers components (e.g. transformers text encoders) must use `apply_group_offloading` rather than `.enable_group_offload()`.

## Sequential CPU offloading

`enable_sequential_cpu_offload()` is a legacy API. Group offloading with `leaf_level + use_stream=True` does the same thing but faster — `use_stream` overlaps data transfer with computation via layer prefetching, reducing overall latency. Prefer that over sequential offloading.

See `docs/source/en/optimization/memory.md` / https://huggingface.co/docs/diffusers/main/en/optimization/memory#cuda-stream for details on `use_stream`, `record_stream`, and the 2× RAM requirement.

## Debugging OOM

1. Identify which stage OOMs: loading, text encoding, denoising, or VAE decode
2. OOM during `.to("cuda")` → full pipeline doesn't fit; use model CPU offloading or group offloading
3. OOM during denoising with model CPU offloading → the transformer alone exceeds VRAM; use layerwise casting or group offloading instead
4. OOM during VAE decode → add `pipe.vae.enable_tiling()`
5. Group offloading causes CPU-side OOM or heavy swapping → use `offload_to_disk_path` to spill to disk instead (see `docs/source/en/optimization/memory.md` / https://huggingface.co/docs/diffusers/main/en/optimization/memory#offloading-to-disk)
6. Still OOM → see [quantization.md](quantization.md) and [memory-calculator.md](memory-calculator.md)
