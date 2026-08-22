<!-- Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# SanaWMTransformer3DModel

A 3D Diffusion Transformer (1.6B parameters) for camera-controlled image-to-video generation, used as the stage-1
sampler of [`SanaWMPipeline`]. The transformer combines:

* a bidirectional GDN-Triton linear-attention main branch (depth 20, hidden 2240, 20 heads),
* a UCPE (Unified Camera Pose Embedding) camera-control branch that consumes a raymap + Plücker representation of
  the requested trajectory, and
* a Wan-style 3D rotary position embedding plus periodic softmax-attention blocks injected every `softmax_every_n`
  layers.

The state-dict layout matches the public SANA-WM release one-to-one — the diffusers wrapper places the inner DiT
under a `_inner.` prefix. See [`SanaWMTransformer3DModel.add_inner_prefix`] for the helper used by the conversion
script.

The model can be loaded with:

```python
import torch
from diffusers import SanaWMTransformer3DModel

transformer = SanaWMTransformer3DModel.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
```

## SanaWMTransformer3DModel

[[autodoc]] SanaWMTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
