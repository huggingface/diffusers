<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# HiDreamImageTransformer2DModel

A Transformer model for image-like data from [HiDream-I1](https://huggingface.co/HiDream-ai).

The model can be loaded with the following code snippet.

```python
from diffusers import HiDreamImageTransformer2DModel

transformer = HiDreamImageTransformer2DModel.from_pretrained("HiDream-ai/HiDream-I1-Full", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## Loading GGUF quantized checkpoints for HiDream-I1

GGUF checkpoints for the `HiDreamImageTransformer2DModel` can  be loaded using `~FromOriginalModelMixin.from_single_file`

```python
import torch
from diffusers import GGUFQuantizationConfig, HiDreamImageTransformer2DModel

ckpt_path = "https://huggingface.co/city96/HiDream-I1-Dev-gguf/blob/main/hidream-i1-dev-Q2_K.gguf"
transformer = HiDreamImageTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16
)
```

## HiDreamImageTransformer2DModel

[[autodoc]] HiDreamImageTransformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
