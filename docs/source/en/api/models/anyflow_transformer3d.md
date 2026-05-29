<!-- Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AnyFlowTransformer3DModel

The bidirectional 3D Transformer used by [`AnyFlowPipeline`](../pipelines/anyflow#anyflowpipeline). It is the
v0.35.1 Wan2.1 backbone with one structural change: the timestep embedder is replaced by
``AnyFlowDualTimestepTextImageEmbedding``, so every forward call conditions on both the source timestep
``t`` and the target timestep ``r``. This is the embedding required to learn the flow map
$\Phi_{r\leftarrow t}$ introduced in
[AnyFlow](https://huggingface.co/papers/2605.13724). See the [`AnyFlowPipeline`](../pipelines/anyflow) page
for paper, authors, and released checkpoints.

For chunk-wise autoregressive (FAR causal) generation, use
[`AnyFlowFARTransformer3DModel`](anyflow_far_transformer3d) instead.

```python
from diffusers import AnyFlowTransformer3DModel

# Bidirectional AnyFlow checkpoint (T2V):
transformer = AnyFlowTransformer3DModel.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer"
)
```

## AnyFlowTransformer3DModel

[[autodoc]] AnyFlowTransformer3DModel
