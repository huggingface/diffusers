# Layerwise Casting

## Documentation

Read the full guide for API details, usage examples, and supported dtypes:

- **Local:** `docs/source/en/optimization/memory.md` (search for "layerwise casting")
- **Online:** https://huggingface.co/docs/diffusers/main/en/optimization/memory#layerwise-casting

## When to use

- The model **almost** fits in VRAM (e.g., 28 GB model on a 32 GB GPU)
- You want memory savings with **less speed penalty** than offloading
- You want to **combine with group offloading** for even more savings

## Gotchas

### `pipe.to()` dtype caveat

`pipe.to(device)` preserves fp8 weights as long as you do **not** pass an explicit dtype:

```python
# Safe — preserves fp8 storage
pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
pipe.to("cuda")

# WRONG — overwrites fp8 back to bf16
pipe.to("cuda", torch.bfloat16)
```

When in doubt, use `enable_model_cpu_offload()` after applying layerwise casting — it moves components one at a time without dtype overrides.

### Apply layerwise casting before offloading

Always call `enable_layerwise_casting` (or `apply_layerwise_casting`) **before** calling `enable_model_cpu_offload()` or `enable_group_offload()`. Reversing the order results in full-precision weights being offloaded and the casting hooks not firing correctly.

For PEFT/LoRA incompatibility and the list of automatically skipped layer types, see `docs/source/en/optimization/memory.md` (layerwise casting section) / https://huggingface.co/docs/diffusers/main/en/optimization/memory#layerwise-casting.

When estimating the post-casting size, use `S_component * 0.45` rather than `0.5` — skipped norm/embed layers mean the reduction is less than half. See [memory-calculator.md](memory-calculator.md).
