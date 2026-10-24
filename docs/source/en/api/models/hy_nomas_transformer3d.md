<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# HyNomasTransformer3DModel

A double-stream transformer for segment autoregressive video generation. Video tokens use three-dimensional rotary
positions, text tokens retain their segment-local positions, and camera attention conditions each latent frame on
four pixel-frame extrinsics. The caller supplies the key/value cache explicitly.

```python
import torch
from diffusers import HyNomasTransformer3DModel

model = HyNomasTransformer3DModel.from_pretrained(
    "tencent/Hy-Nomas-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
)
```

## HyNomasTransformer3DModel

[[autodoc]] HyNomasTransformer3DModel
    - forward
