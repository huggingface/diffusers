<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching

Caching reuses intermediate layer outputs across denoising steps to speed up inference. It uses more memory and doesn't need training. Enable a method on the transformer with a config.

## Choose a cache method

Pick a method depending on how much config you will set, and the fit you need.

| Method | Use when | Tradeoff |
|--------|----------|----------|
| Text KV Cache | NucleusMoE image only, need exact text K/V reuse across steps | Lossless |
| SeaCache | Video transformers that already have a SeaCache path | Approximate, settings often do not transfer across models |
| FirstBlockCache | Want one main speed/quality knob on a registered transformer | Approximate |
| MagCache | Have magnitude ratios for your checkpoint and scheduler, or will calibrate first | Approximate, ratios are checkpoint and scheduler specific |
| TaylorSeer | Want to predict later activations from earlier steps | Approximate |
| PAB | Video, willing to tune attention reuse (block and timestep skip ranges per attention kind) | Approximate |
| FasterCache | Like PAB, plus optional CFG-branch skipping | Approximate, experimental |

## Pyramid Attention Broadcast

[Pyramid Attention Broadcast (PAB)](https://huggingface.co/papers/2408.12588) approximates attention across denoising steps by reusing attention outputs for some blocks and timesteps instead of recomputing every step. Config separates attention kinds (spatial, temporal, cross) when the model has them. Not every video model exposes all three, and set only the ranges that match the blocks you have.

Each kind uses a `*_attention_block_skip_range` (how often to recompute vs reuse within the window) and a `*_attention_timestep_skip_range` (which denoising timesteps may skip). You must pass `current_timestep_callback` so the hook can read the pipeline’s current timestep. Wider or more aggressive skips usually mean more speed and more quality risk.

Pass a [`PyramidAttentionBroadcastConfig`] to enable it.

```python
import torch
from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe.current_timestep,
)
pipe.transformer.enable_cache(config)
```

## FasterCache

[FasterCache](https://huggingface.co/papers/2410.19355) caches and reuses attention features similar to [PAB](#pyramid-attention-broadcast). It can also skip the unconditional branch under classifier-free guidance and estimate it from the conditional branch when successive latents are redundant enough.

Pass a [`FasterCacheConfig`] to enable it. Like PAB, set `*_attention_block_skip_range` and `*_attention_timestep_skip_range` for the attention kinds you have, plus the CFG-branch skip options when you want them.

```python
import torch
from diffusers import CogVideoXPipeline, FasterCacheConfig

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

config = FasterCacheConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(-1, 681),
    current_timestep_callback=lambda: pipe.current_timestep,
    attention_weight_callback=lambda _: 0.3,
    unconditional_batch_skip_range=5,
    unconditional_batch_timestep_skip_range=(-1, 641),
    tensor_format="BFCHW",
)
pipe.transformer.enable_cache(config)
```

## SeaCache

[SeaCache](https://huggingface.co/papers/2602.18993) compares Spectral Evolution Aware (SEA) indicators between successive denoising steps. When the accumulated change stays under a threshold, it skips the transformer block stack and predicts the output from cached residuals. The method is approximate and designed for video generation.

Built-in adapters for SeaCache include:

- Cosmos 3 is the primary optimized and benchmarked integration.
- Wan T2V uses the generic repeated-block path as a demo for how to provide the raw vision latents to SeaCache. The same cache parameters may not transfer to Wan or other Wan variants.

Enable SeaCache on the transformer. The Cosmos 3 denoising loop attaches scheduler step, sigma, and step count to each `cache_context`, so no extra parameters are needed.

```python
from diffusers import Cosmos3OmniPipeline, SeaCacheConfig

pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano")
pipe.transformer.enable_cache(SeaCacheConfig(threshold=0.2, max_consecutive_cached=2))
```

SeaCache may change outputs. Call `pipe.transformer.disable_cache()` when you need every step to run the full transformer. The same enable call works with [`Cosmos3OmniPipeline`], [`Cosmos3OmniModularPipeline`], and [`Cosmos3DistilledModularPipeline`].

To integrate another video transformer, use `CacheMixin`, register the block layout in `TransformerBlockRegistry`, enter a `cache_context` on every call with `step_index`, `sigma`, and `num_inference_steps`, and pass a `raw_vision_callback` when no built-in adapter exists. Tune parameters per model and scheduler.

## FirstBlockCache

[`FirstBlockCacheConfig`] checks how much the early layers of the denoiser change from one timestep to the next. If the change is small, the model skips the expensive later layers and reuses the previous output.

Enable it through `enable_cache` so `disable_cache` and `is_cache_enabled` stay in sync. The default `threshold` is `0.05`. A higher value such as `0.2` skips more often for extra speed, but generation quality may drop.

```python
import torch
from diffusers import DiffusionPipeline, FirstBlockCacheConfig

pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16
)
pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=0.2))
```

## TaylorSeer Cache

[TaylorSeer Cache](https://huggingface.co/papers/2503.06923) accelerates diffusion inference with Taylor series expansions across denoising steps. It predicts later-step activations from earlier ones and reuses those predictions for several steps so the transformer does less full work.

- `cache_interval`: Number of steps to reuse cached outputs before performing a full forward pass
- `disable_cache_before_step`: Initial steps that use full computations to gather data for approximations
- `max_order`: Higher Taylor orders can be more accurate but use more memory. Keep this at `1` unless you have a reason to change it.

```python
import torch
from diffusers import FluxPipeline, TaylorSeerCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    dtype=torch.bfloat16,
).to("cuda")  # or "mps", "xpu", "cpu"

config = TaylorSeerCacheConfig(
    cache_interval=5,
    max_order=1,
    disable_cache_before_step=10,
    taylor_factors_dtype=torch.bfloat16,
)
pipe.transformer.enable_cache(config)
```

## MagCache

[MagCache](https://github.com/Zehong-Ma/MagCache) skips transformer blocks from the residual update magnitude. Update magnitudes decay predictably over denoising, and MagCache tracks an error budget from precomputed magnitude ratios (`mag_ratios`) to decide when reuse is safe. Those ratios are checkpoint and scheduler-specific. Ratios from a high step count can be interpolated down to fewer steps.

MagCache follows two steps:

1. Calibration: Run inference once with `calibrate=True`. The hook measures residual magnitudes and prints the calculated ratios.
2. Inference: Disable the calibration cache, then pass those ratios to `MagCacheConfig` for acceleration.

Classifier-free guidance may affect calibration. Pipelines that use true CFG with sequential contexts, such as Flux when `true_cfg_scale > 1`, enter `cache_context("cond")` and `cache_context("uncond")` separately. Calibration may print one array per context, but you should use the conditional array in most cases. Pipelines that batch CFG by concatenating conditional and unconditional inputs (for example, CogVideoX) produce a single joint array you can use directly.

```python
import torch
from diffusers import FluxPipeline, MagCacheConfig
from diffusers.hooks.mag_cache import FLUX_MAG_RATIOS

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"

# 1. Calibration Step
# Run full inference to measure model behavior.
calib_config = MagCacheConfig(calibrate=True, num_inference_steps=4)
pipe.transformer.enable_cache(calib_config)

# Run a prompt to trigger calibration
pipe("A cat playing chess", num_inference_steps=4)
# Prints: [MagCache] Calibration Complete. Copy these values to MagCacheConfig(mag_ratios=...):

# 2. Inference Step
# Disable calibration hooks before enabling MagCache for inference.
pipe.transformer.disable_cache()

# Apply ratios from calibration, or use the Flux defaults:
# mag_ratios=FLUX_MAG_RATIOS
mag_config = MagCacheConfig(
    mag_ratios=[1.0, 1.37, 0.97, 0.87],
    num_inference_steps=4
)

pipe.transformer.enable_cache(mag_config)

image = pipe("A cat playing chess", num_inference_steps=4).images[0]
```

## Text KV Cache

[`TextKVCacheConfig`] enables exact (lossless) reuse of text key and value projections across denoising steps. It is for NucleusMoE image only (`NucleusMoEImageTransformerBlock`, the architecture [`apply_text_kv_cache`] hooks). Enable it with `enable_cache`.

```python
import torch
from diffusers import NucleusMoEImagePipeline, TextKVCacheConfig

pipe = NucleusMoEImagePipeline.from_pretrained(
    "NucleusAI/NucleusMoE-Image", dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

pipe.transformer.enable_cache(TextKVCacheConfig())

image = pipe("A cat holding a sign that says hello world", num_inference_steps=50).images[0]
```
