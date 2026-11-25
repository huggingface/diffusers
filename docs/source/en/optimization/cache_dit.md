# CacheDiT

[CacheDiT](https://github.com/vipshop/cache-dit) speeds up DiT pipelines by reusing transformer block outputs across denoising steps. It does not need training and supports most Diffusers DiT pipelines, including Flux, Qwen-Image, Wan, and HunyuanVideo. Diffusers also has [built-in caching](./cache) which doesn't require an extra dependency.

Install CacheDiT from PyPI.

```bash
pip install -U cache-dit
```

Call `cache_dit.supported_pipelines()` to list the pipeline families CacheDiT supports.

```py
import cache_dit

cache_dit.supported_pipelines()
```

## Enable caching

Call `cache_dit.enable_cache` on a pipeline to cache it with the default settings, then run the pipeline as usual.

```py
import torch
import cache_dit
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16
).to("cuda")
cache_dit.enable_cache(pipeline)

image = pipeline(
    "A cat holding a sign that says hello world", num_inference_steps=28
).images[0]
```

CacheDiT also works with `torch.compile`. Compile the transformer after you call `enable_cache`. See the [compile](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/COMPILE.md) docs for settings that avoid recompilation with dynamic input shapes.

```py
pipeline.transformer = torch.compile(pipeline.transformer)
```

Call `cache_dit.summary` after inference to log how many steps were cached and the residual differences between steps.

```py
stats = cache_dit.summary(pipeline)
```

Call `cache_dit.disable_cache` to restore the original pipeline.

```py
cache_dit.disable_cache(pipeline)
```

## Configure the cache

DBCache (Dual Block Cache) computes the first n blocks (Fn) at every step. When their output barely changes from the previous step, it reuses the cached output for the remaining blocks, and it can recompute the last n blocks (Bn) to correct it. The TaylorSeer calibrator predicts the cached output from earlier steps instead of reusing it as is.

`enable_cache` defaults to DBCache with the first 8 blocks always computed (F8B0) and 8 uncached warmup steps. To trade speed for quality, raise `Fn_compute_blocks` or lower `residual_diff_threshold` (default `0.08`). For the best quality at high cache rates, add the TaylorSeer calibrator.

```py
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig

cache_dit.enable_cache(
    pipeline,
    cache_config=DBCacheConfig(
        max_warmup_steps=8,
        Fn_compute_blocks=8,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.12,
    ),
    calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1),
)
```

For supported pipelines, CacheDiT already knows whether CFG runs as a separate forward pass. For other models, set `enable_separate_cfg=True` in `DBCacheConfig` if the model runs the conditional and unconditional passes separately, or `False` if it fuses them or doesn't use CFG.

See the [DBCache design](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/DBCACHE_DESIGN.md) docs for how the Fn and Bn blocks work, and the [cache benchmarks](https://github.com/vipshop/cache-dit/blob/main/bench/cache/README.md) for speed and quality numbers.

## Next steps

- For pipelines CacheDiT doesn't support yet, see the [BlockAdapter](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/CACHE_API.md#automatic-block-adapter) docs.
- CacheDiT also supports [context parallelism](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/CONTEXT_PARALLEL.md) and [quantization](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/QUANTIZATION.md).
