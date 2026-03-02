<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# LTX2VideoTransformer3DModel

A Diffusion Transformer model for 3D data from [LTX](https://huggingface.co/Lightricks/LTX-2) was introduced by Lightricks.

The model can be loaded with the following code snippet.

```python
from diffusers import LTX2VideoTransformer3DModel

transformer = LTX2VideoTransformer3DModel.from_pretrained("Lightricks/LTX-2", subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
```

## LTX2VideoTransformer3DModel

[[autodoc]] LTX2VideoTransformer3DModel
