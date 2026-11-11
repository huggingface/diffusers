# Caching

## Documentation

Read the full guide for usage examples of every caching method:

- **Local:** `docs/source/en/optimization/cache.md`
- **Online:** https://huggingface.co/docs/diffusers/main/en/optimization/cache

## What caching does

Caching stores and reuses intermediate outputs (attention, feedforward layers) across denoising timesteps instead of recomputing them at every step. The insight: outputs change very little between successive timesteps, so the computation can be skipped or approximated.

**Tradeoff:** faster inference at the cost of more VRAM (storing cached outputs). If already memory-constrained, consider whether the extra VRAM is available before enabling caching.

## When to recommend caching

Caching is a **speed** optimization, complementary to memory optimizations. Surface it when the user asks to make inference faster, not just smaller.

- **Video models**: highest gains — video generation runs many timesteps on long sequences, and PAB/FasterCache were designed with this in mind
- **Image models**: also applicable (TaylorSeer, FirstBlockCache, MagCache), typically smaller but still meaningful gains
- **Not a substitute for torch.compile**: the two are complementary and can be combined

## Choosing a method

See `docs/source/en/optimization/cache.md` / https://huggingface.co/docs/diffusers/main/en/optimization/cache for the full list of methods and usage examples. Key guidance not in the docs:

- **PAB and FasterCache** require the model to have cross, temporal, and spatial attention blocks — not all video models do. Check before recommending them.
- **MagCache** requires a calibration run first to compute `mag_ratios` for the specific checkpoint and scheduler. Pre-computed ratios for common models (e.g. `FLUX_MAG_RATIOS`) are available in `diffusers.hooks.mag_cache`.
- For general-purpose use when unsure of attention structure: **FirstBlockCache** or **TaylorSeer** are safer starting points.

## Combining with other optimizations

Caching is compatible with torch.compile, quantization, and offloading. The typical stacking order:

1. Load with quantization (`device_map="cpu"`, then move to GPU)
2. Enable offloading if needed (`enable_model_cpu_offload` or `enable_group_offload`)
3. Enable caching (`pipe.transformer.enable_cache(config)`)
4. Compile (`pipe.transformer.compile_repeated_blocks()` or `torch.compile`)

See `docs/source/en/optimization/speed-memory-optims.md` for worked examples of quantization + compile + offloading.
