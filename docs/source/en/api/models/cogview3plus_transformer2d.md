<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# CogView3PlusTransformer2DModel

A Diffusion Transformer model for 2D data from [CogView3Plus](https://github.com/THUDM/CogView3) was introduced in [CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion](https://huggingface.co/papers/2403.05121) by Tsinghua University & ZhipuAI.

The model can be loaded with the following code snippet.

```python
from diffusers import CogView3PlusTransformer2DModel

vae = CogView3PlusTransformer2DModel.from_pretrained("THUDM/CogView3Plus-3b", subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
```

## CogView3PlusTransformer2DModel

[[autodoc]] CogView3PlusTransformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
