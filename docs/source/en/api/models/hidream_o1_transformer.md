<!-- Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# HiDreamO1Transformer2DModel

A Qwen3-VL based raw pixel patch transformer for
[HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image).

HiDream-O1 does not use a VAE. The transformer predicts raw RGB pixel patches through the O1 denoising path added on
top of Qwen3-VL.

The model can be loaded with the following code snippet.

```python
import torch
from diffusers import HiDreamO1Transformer2DModel

transformer = HiDreamO1Transformer2DModel.from_pretrained(
    "HiDream-ai/HiDream-O1-Image",
    torch_dtype=torch.bfloat16,
)
```

## HiDreamO1Transformer2DModel

[[autodoc]] HiDreamO1Transformer2DModel
