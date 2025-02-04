<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# EasyAnimateTransformer3DModel

A Diffusion Transformer model for 3D data from [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) was introduced by Alibaba PAI.

The model can be loaded with the following code snippet.

```python
from diffusers import EasyAnimateTransformer3DModel

transformer = EasyAnimateTransformer3DModel.from_pretrained("alibaba-pai/EasyAnimateV5.1-12b-zh", subfolder="transformer", torch_dtype=torch.float16).to("cuda")
```

## EasyAnimateTransformer3DModel

[[autodoc]] EasyAnimateTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
