<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching

Caching accelerates inference by storing and reusing intermediate outputs of different layers, such as attention and feedforward layers, instead of performing the entire computation at each inference step. It significantly improves generation speed at the expense of more memory and doesn't require additional training.

This guide shows you how to use the caching methods supported in Diffusers.

## Pyramid Attention Broadcast

[Pyramid Attention Broadcast (PAB)](https://huggingface.co/papers/2408.12588) is based on the observation that attention outputs aren't that different between successive timesteps of the generation process. The attention differences are smallest in the cross attention layers and are generally cached over a longer timestep range. This is followed by temporal attention and spatial attention layers.

> [!TIP]
> Not all video models have three types of attention (cross, temporal, and spatial)!

PAB can be combined with other techniques like sequence parallelism and classifier-free guidance parallelism (data parallelism) for near real-time video generation.

Set up and pass a [`PyramidAttentionBroadcastConfig`] to a pipeline's transformer to enable it. The `spatial_attention_block_skip_range` controls how often to skip attention calculations in the spatial attention blocks and the `spatial_attention_timestep_skip_range` is the range of timesteps to skip. Take care to choose an appropriate range because a smaller interval can lead to slower inference speeds and a larger interval can result in lower generation quality.

```python
import torch
from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe.current_timestep,
)
pipeline.transformer.enable_cache(config)
```

## FasterCache

[FasterCache](https://huggingface.co/papers/2410.19355) caches and reuses attention features similar to [PAB](#pyramid-attention-broadcast) since output differences are small for each successive timestep.

This method may also choose to skip the unconditional branch prediction, when using classifier-free guidance for sampling (common in most base models), and estimate it from the conditional branch prediction if there is significant redundancy in the predicted latent outputs between successive timesteps.

Set up and pass a [`FasterCacheConfig`] to a pipeline's transformer to enable it.

```python
import torch
from diffusers import CogVideoXPipeline, FasterCacheConfig

pipe line= CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

config = FasterCacheConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(-1, 681),
    current_timestep_callback=lambda: pipe.current_timestep,
    attention_weight_callback=lambda _: 0.3,
    unconditional_batch_skip_range=5,
    unconditional_batch_timestep_skip_range=(-1, 781),
    tensor_format="BFCHW",
)
pipeline.transformer.enable_cache(config)
```

## FirstBlockCache

[FirstBlock Cache](https://huggingface.co/docs/diffusers/main/en/api/cache#diffusers.FirstBlockCacheConfig) checks how much the early layers of the denoiser changes from one timestep to the next. If the change is small, the model skips the expensive later layers and reuses the previous output.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16
)
apply_first_block_cache(pipeline.transformer, FirstBlockCacheConfig(threshold=0.2))
```
## TaylorSeer Cache

[TaylorSeer Cache](https://huggingface.co/papers/2403.06923) accelerates diffusion inference by using Taylor series expansions to approximate and cache intermediate activations across denoising steps. The method predicts future outputs based on past computations, reusing them at specified intervals to reduce redundant calculations.

This caching mechanism delivers strong results with minimal additional memory overhead. For detailed performance analysis, see [our findings here](https://github.com/huggingface/diffusers/pull/12648#issuecomment-3610615080).

To enable TaylorSeer Cache, create a [`TaylorSeerCacheConfig`] and pass it to your pipeline's transformer:

- `cache_interval`: Number of steps to reuse cached outputs before performing a full forward pass
- `disable_cache_before_step`: Initial steps that use full computations to gather data for approximations
- `max_order`: Approximation accuracy (in theory, higher values improve quality but increase memory usage but we recommend it should be set to `1`)

```python
import torch
from diffusers import FluxPipeline, TaylorSeerCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

config = TaylorSeerCacheConfig(
    cache_interval=5,
    max_order=1,
    disable_cache_before_step=10,
    taylor_factors_dtype=torch.bfloat16,
)
pipe.transformer.enable_cache(config)
```

## MagCache

[MagCache](https://github.com/Zehong-Ma/MagCache) accelerates inference by skipping transformer blocks based on the magnitude of the residual update. It observes that the magnitude of updates (Output - Input) decays predictably over the diffusion process. By accumulating an "error budget" based on pre-computed magnitude ratios, it dynamically decides when to skip computation and reuse the previous residual.

MagCache relies on **Magnitude Ratios** (`mag_ratios`), which describe this decay curve. These ratios are specific to the model checkpoint and scheduler.

### Usage

To use MagCache, you typically follow a two-step process: **Calibration** and **Inference**.

1.  **Calibration**: Run inference once with `calibrate=True`. The hook will measure the residual magnitudes and print the calculated ratios to the console.
2.  **Inference**: Pass these ratios to `MagCacheConfig` to enable acceleration.

```python
import torch
from diffusers import FluxPipeline, MagCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
).to("cuda")

# 1. Calibration Step
# Run full inference to measure model behavior.
calib_config = MagCacheConfig(calibrate=True, num_inference_steps=4)
pipe.transformer.enable_cache(calib_config)

# Run a prompt to trigger calibration
pipe("A cat playing chess", num_inference_steps=4)
# Logs will print something like: "MagCache Calibration Results: [1.0, 1.37, 0.97, 0.87]"

# 2. Inference Step
# Apply the specific ratios obtained from calibration for optimized speed.
# Note: For Flux models, you can also import defaults: 
# from diffusers.hooks.mag_cache import FLUX_MAG_RATIOS
mag_config = MagCacheConfig(
    mag_ratios=[1.0, 1.37, 0.97, 0.87],
    num_inference_steps=4
)

pipe.transformer.enable_cache(mag_config) 

image = pipe("A cat playing chess", num_inference_steps=4).images[0]
```

> [!NOTE]
> `mag_ratios` represent the model's intrinsic magnitude decay curve. Ratios calibrated for a high number of steps (e.g., 50) can be reused for lower step counts (e.g., 20). The implementation uses interpolation to map the curve to the current number of inference steps.

> [!TIP]
> For pipelines that run Classifier-Free Guidance sequentially (like Kandinsky 5.0), the calibration log might print two arrays: one for the Conditional pass and one for the Unconditional pass. In most cases, you should use the first array (Conditional).

> [!TIP]
> For pipelines that run Classifier-Free Guidance in a **batched** manner (like SDXL or Flux), the `hidden_states` processed by the model contain both conditional and unconditional branches concatenated together. The calibration process automatically accounts for this, producing a single array of ratios that represents the joint behavior. You can use this resulting array directly without modification.
