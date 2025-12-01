<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# HunyuanVideo15Transformer3DModel

A Diffusion Transformer model for 3D video-like data used in [HunyuanVideo1.5](https://github.com/Tencent/HunyuanVideo1-1.5).

The model can be loaded with the following code snippet.

```python
from diffusers import HunyuanVideo15Transformer3DModel

transformer = HunyuanVideo15Transformer3DModel.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v" subfolder="transformer", torch_dtype=torch.bfloat16)
```

## HunyuanVideo15Transformer3DModel

[[autodoc]] HunyuanVideo15Transformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
