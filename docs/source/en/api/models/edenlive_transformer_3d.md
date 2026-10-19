<!-- Copyright 2026 The VAST-AI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EdenLiveTransformer3DModel

[EdenLive](https://github.com/VAST-AI-Research/EdenLive-Official) uses a video diffusion transformer conditioned on
text, a reference image, and projected DINO proxy features. The same model class supports bidirectional and causal
attention. See the [pipeline guide](../pipelines/edenlive) for video generation.

```python
import torch
from diffusers import EdenLiveTransformer3DModel

transformer = EdenLiveTransformer3DModel.from_pretrained(
    "VAST-AI/EdenLive-5B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
)
```

Use `subfolder="transformer_causal"` for the causal weights. Keep model timesteps in FP32 when calling the transformer
directly, including when its weights are BF16.

## EdenLiveTransformer3DModel

[[autodoc]] EdenLiveTransformer3DModel
    - forward
    - create_causal_state

## EdenLiveProxyProjection

Projects DINO features into the transformer's conditioning space. Use the trained projection and DINO encoder from
the same checkpoint; the projection returns FP32 features.

[[autodoc]] EdenLiveProxyProjection
    - forward

## EdenLiveCausalState

The causal pipeline manages this state automatically. For custom loops, create a separate state for each CFG branch
with `transformer.create_causal_state()`. Denoise with `cache_update=False`, then commit each clean chunk at timestep
zero with `cache_update=True`. Call `reset()` before an independent video.

[[autodoc]] EdenLiveCausalState
    - reset
