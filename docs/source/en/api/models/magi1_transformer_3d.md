<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Magi1Transformer3DModel

A Diffusion Transformer model for 3D video-like data was introduced in [MAGI-1: Autoregressive Video Generation at Scale](https://arxiv.org/abs/2505.13211) by Sand.ai.

MAGI-1 is an autoregressive denoising video generation model that generates videos chunk-by-chunk instead of as a whole. Each chunk (24 frames) is denoised holistically, and the generation of the next chunk begins as soon as the current one reaches a certain level of denoising.

The model can be loaded with the following code snippet.

```python
from diffusers import Magi1Transformer3DModel

transformer = Magi1Transformer3DModel.from_pretrained("sand-ai/MAGI-1", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## Magi1Transformer3DModel

[[autodoc]] Magi1Transformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput