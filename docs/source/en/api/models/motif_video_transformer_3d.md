<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MotifVideoTransformer3DModel

A Diffusion Transformer model for 3D video-like data was introduced in Motif-Video by the Motif Technologies Team.

The model uses a three-stage architecture with 12 dual-stream + 16 single-stream + 8 DDT decoder layers and rotary positional embeddings (RoPE) for video generation.

The model can be loaded with the following code snippet.

```python
from diffusers import MotifVideoTransformer3DModel

transformer = MotifVideoTransformer3DModel.from_pretrained("Motif-Technologies/Motif-Video-2B", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## MotifVideoTransformer3DModel

[[autodoc]] MotifVideoTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
