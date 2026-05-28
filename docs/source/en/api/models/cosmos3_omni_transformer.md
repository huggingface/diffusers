<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Cosmos3OmniTransformer

A Mixture-of-Transformer (MoT) joint vision-language transformer introduced as part of NVIDIA's Cosmos3 world foundation model family. The model runs two parallel computation pathways over a packed joint sequence:

- a **causal "understanding" pathway** that self-attends over text tokens with causal masking, and
- a **bi-directional "generation" pathway** that cross-attends from generation tokens (vision + optional sound latents) over the full understanding-plus-generation key/value set.

The two pathways share the same hidden size and number of layers but maintain **separate Q/K/V/O projections, MLPs, and RMSNorm parameters**, which is what makes the architecture a Mixture-of-Transformer rather than a standard Mixture-of-Experts. Position information is supplied through a 3D multimodal RoPE (mRoPE) that interleaves temporal / height / width frequencies for video latents and reuses the temporal axis for text and audio.

The model can be loaded as follows.

```python
import torch
from diffusers import Cosmos3OmniTransformer

transformer = Cosmos3OmniTransformer.from_pretrained(
    "nvidia/Cosmos3-Nano", subfolder="transformer", torch_dtype=torch.bfloat16
)
```

## Cosmos3OmniTransformer

[[autodoc]] Cosmos3OmniTransformer
