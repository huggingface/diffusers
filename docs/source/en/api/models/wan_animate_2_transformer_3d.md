<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# WanAnimate2Transformer3DModel

A Diffusion Transformer model for 3D video-like data used in [Wan-Animate-2](https://github.com/Wan-Video/Wan2.2) by the Alibaba Wan Team. It animates a character image with the motion of a driving video through an in-context reference mechanism: each segment first runs a reference pass (`kv_cache_mode="extract"`) that caches every layer's reference K/V, then the denoising passes (`kv_cache_mode="cached"`) attend jointly over the generation tokens and the cached reference tokens through a flex `BlockMask`.

The model can be loaded with the following code snippet.

```python
from diffusers import WanAnimate2Transformer3DModel

transformer = WanAnimate2Transformer3DModel.from_pretrained("Wan-AI/Wan2.2-Animate-2-14B-Diffusers", subfolder="transformer", dtype=torch.bfloat16)
```

## WanAnimate2Transformer3DModel

[[autodoc]] WanAnimate2Transformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
