<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# HunyuanVideoTransformer3DModel

A Diffusion Transformer model for 3D video-like data was introduced in [HunyuanVideo: A Systematic Framework For Large Video Generative Models](https://huggingface.co/papers/2412.03603) by Tencent.

The model can be loaded with the following code snippet.

```python
from diffusers import HunyuanVideoTransformer3DModel

transformer = HunyuanVideoTransformer3DModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## HunyuanVideoTransformer3DModel

[[autodoc]] HunyuanVideoTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
