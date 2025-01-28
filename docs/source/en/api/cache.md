<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching methods

## Pyramid Attention Broadcast

[Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588) from Xuanlei Zhao, Xiaolong Jin, Kai Wang, Yang You.

Pyramid Attention Broadcast (PAB) is a method that speeds up inference in diffusion models by systematically skipping attention computations between successive inference steps and reusing cached attention states. The attention states are not very different between successive inference steps. The most prominent difference is in the spatial attention blocks, not as much in the temporal attention blocks, and finally the least in the cross attention blocks. Therefore, many cross attention computation blocks can be skipped, followed by the temporal and spatial attention blocks. By combining other techniques like sequence parallelism and classifier-free guidance parallelism, PAB achieves near real-time video generation.

Enable PAB with [`~PyramidAttentionBroadcastConfig`] on any pipeline. For some benchmarks, refer to [this](https://github.com/huggingface/diffusers/pull/9562) pull request.

```python
import torch
from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Increasing the value of `spatial_attention_timestep_skip_range[0]` or decreasing the value of
# `spatial_attention_timestep_skip_range[1]` will decrease the interval in which pyramid attention
# broadcast is active, leader to slower inference speeds. However, large intervals can lead to
# poorer quality of generated videos.
config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe.current_timestep,
)
pipe.transformer.enable_cache(config)
```

## Faster Cache

[FasterCache](https://huggingface.co/papers/2410.19355) from Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, Kwan-Yee K. Wong.

FasterCache is a method that speeds up inference in diffusion transformers by:
- Reusing attention states between successive inference steps, due to high similarity between them
- Skipping unconditional branch prediction used in classifier-free guidance by revealing redundancies between unconditional and conditional branch outputs for the same timestep, and therefore approximating the unconditional branch output using the conditional branch output

```python
import torch
from diffusers import CogVideoXPipeline, FasterCacheConfig

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipe.to("cuda")

config = FasterCacheConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(-1, 681),
    current_timestep_callback=lambda: pipe.current_timestep,
    attention_weight_callback=lambda _: 0.3,
    unconditional_batch_skip_range=5,
    unconditional_batch_timestep_skip_range=(-1, 781),
    tensor_format="BFCHW",
)
pipe.transformer.enable_cache(config)
```

### CacheMixin

[[autodoc]] CacheMixin

### PyramidAttentionBroadcastConfig

[[autodoc]] PyramidAttentionBroadcastConfig

[[autodoc]] apply_pyramid_attention_broadcast

### FasterCacheConfig

[[autodoc]] FasterCacheConfig

[[autodoc]] apply_faster_cache
