<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# WanAnimateTransformer3DModel

A Diffusion Transformer model for 3D video-like data was introduced in [Wan Animate](https://github.com/Wan-Video/Wan2.2) by the Alibaba Wan Team.

The model can be loaded with the following code snippet.

```python
from diffusers import WanAnimateTransformer3DModel

transformer = WanAnimateTransformer3DModel.from_pretrained("Wan-AI/Wan2.2-Animate-14B-720P-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## WanAnimateTransformer3DModel

[[autodoc]] WanAnimateTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
