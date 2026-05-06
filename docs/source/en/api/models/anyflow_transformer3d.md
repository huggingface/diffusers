<!-- Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AnyFlowTransformer3DModel

A 3D Transformer used by `AnyFlowPipeline` and `AnyFlowFARPipeline`. The architecture extends the
Wan2.1 3D DiT backbone with two optional modules controlled by config flags:

1. **FAR causal blocks** (`init_far_model=True`) — block-sparse causal attention via
   `torch.nn.attention.flex_attention` plus a compressed-frame patch embedding. Enables frame-level
   autoregressive generation as introduced in [FAR (Gu et al., 2025)](https://arxiv.org/abs/2503.19325).
2. **Dual-timestep flow-map embedding** (`init_flowmap_model=True`) — adds a second timestep embedder
   (`delta_embedder`) that conditions on the target timestep `r_timestep` in addition to the source
   timestep, enabling flow-map sampling $\mathbf{z}_t \to \mathbf{z}_r$ over arbitrary intervals (introduced
   in [AnyFlow](https://huggingface.co/papers/<arxiv-id>)).

Setting both flags to `False` reduces this model to the v0.35.1 Wan2.1 transformer.

```python
from diffusers import AnyFlowTransformer3DModel

# Bidirectional AnyFlow checkpoint (T2V):
transformer = AnyFlowTransformer3DModel.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer"
)

# Causal AnyFlow checkpoint (FAR):
transformer = AnyFlowTransformer3DModel.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", subfolder="transformer"
)
```

## AnyFlowTransformer3DModel

[[autodoc]] AnyFlowTransformer3DModel
