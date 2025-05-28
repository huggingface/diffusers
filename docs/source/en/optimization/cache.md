<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching

Caching accelerates inference by storing and reusing redundant attention outputs instead of performing extra computation. It significantly improves efficiency and doesn't require additional training.

This guide shows you how to use the caching methods supported in Diffusers.

## Pyramid Attention Broadcast

[Pyramid Attention Broadcast (PAB)](https://huggingface.co/papers/2408.12588) is based on the observation that many of the attention output differences are redundant. The attention differences are smallest in the cross attention block so the cached attention states are broadcasted and reused over a longer range. This is followed by temporal attention and finally spatial attention.

PAB can be combined with other techniques like sequence parallelism and classifier-free guidance parallelism for near real-time video generation.

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

[FasterCache](https://huggingface.co/papers/2410.19355) computes and caches attention features at every other timestep instead of directly reusing cached features because it can cause flickering or blurry details in the generated video. The features from the skipped step are calculated from the difference between the adjacent cached features.

FasterCache also uses a classifier-free guidance (CFG) cache which computes both the conditional and unconditional outputs once. For future timesteps, only the conditional output is calculated and the unconditional output is estimated from the cached biases.

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