<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# JoyVideoEditTransformer3DModel

A dual-stream MM-DiT transformer that denoises video latents one causal chunk at a time, used in
[`JoyVideoEditPipeline`]. The model can be loaded with the following code snippet.

```python
import torch
from diffusers import JoyVideoEditTransformer3DModel

transformer = JoyVideoEditTransformer3DModel.from_pretrained("jdopensource/JoyAI-Video-Edit-Diffusers", subfolder="transformer", dtype=torch.bfloat16)
```

## JoyVideoEditTransformer3DModel

[[autodoc]] JoyVideoEditTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
