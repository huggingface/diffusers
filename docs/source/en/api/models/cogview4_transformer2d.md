<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# CogView4Transformer2DModel

A Diffusion Transformer model for 2D data from [CogView4]()

The model can be loaded with the following code snippet.

```python
from diffusers import CogView4Transformer2DModel

transformer = CogView4Transformer2DModel.from_pretrained("THUDM/CogView4-6B", subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
```

## CogView4Transformer2DModel

[[autodoc]] CogView4Transformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
