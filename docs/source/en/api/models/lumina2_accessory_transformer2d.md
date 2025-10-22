<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Lumina2AccessoryTransformer2DModel

A Diffusion Transformer model for 2D data from [Lumina-Accessory](https://github.com/Alpha-VLLM/Lumina-Accessory). by Alpha-VLLM.

The model can be loaded with the following code snippet.

```python
from diffusers import Lumina2AccessoryTransformer2DModel

ckpt_path = "https://huggingface.co/Alpha-VLLM/Lumina-Accessory/blob/main/consolidated.00-of-01.pth"
transformer = Lumina2AccessoryTransformer2DModel.from_single_file(ckpt_path, torch_dtype=torch.bfloat16)
```

## Lumina2AccessoryTransformer2DModel

[[autodoc]] Lumina2AccessoryTransformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
