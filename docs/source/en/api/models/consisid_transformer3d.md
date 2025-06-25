<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# ConsisIDTransformer3DModel

A Diffusion Transformer model for 3D data from [ConsisID](https://github.com/PKU-YuanGroup/ConsisID) was introduced in [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://huggingface.co/papers/2411.17440) by Peking University & University of Rochester & etc.

The model can be loaded with the following code snippet.

```python
from diffusers import ConsisIDTransformer3DModel

transformer = ConsisIDTransformer3DModel.from_pretrained("BestWishYsh/ConsisID-preview", subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
```

## ConsisIDTransformer3DModel

[[autodoc]] ConsisIDTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
