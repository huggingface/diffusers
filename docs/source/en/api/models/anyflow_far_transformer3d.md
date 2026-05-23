<!-- Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AnyFlowFARTransformer3DModel

The causal (FAR) 3D Transformer used by [`AnyFlowFARPipeline`](../pipelines/anyflow#anyflowfarpipeline) —
the FAR variant of [AnyFlow](https://huggingface.co/papers/2605.13724). See the
[`AnyFlowFARPipeline`](../pipelines/anyflow) page for paper, authors, and released checkpoints. It extends
the v0.35.1 Wan2.1 backbone with three additions:

1. **FAR causal block-mask** via `torch.nn.attention.flex_attention`, supporting frame-level autoregressive
   generation as introduced in [FAR (Gu et al., 2025)](https://arxiv.org/abs/2503.19325).
2. **Compressed-frame patch embedding** (`far_patch_embedding`) for context (already-generated) frames,
   warm-started from the full-resolution `patch_embedding` at construction time via trilinear interpolation.
3. **Dual-timestep flow-map embedding** (same as
   [`AnyFlowTransformer3DModel`](anyflow_transformer3d)) — every forward call conditions on both the source
   timestep ``t`` and the target timestep ``r``.

The default chunk schedule (`chunk_partition`) is stored in the model config; the released NVIDIA AnyFlow-FAR
checkpoints use `[1, 3, 3, 3, 3, 3, 3, 2]` for the canonical 81-frame setting. `forward` accepts a per-call
`chunk_partition` override, so the same checkpoint also handles other `num_frames` configurations without
retraining.

```python
from diffusers import AnyFlowFARTransformer3DModel

# Causal AnyFlow checkpoint (FAR):
transformer = AnyFlowFARTransformer3DModel.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", subfolder="transformer"
)
```

## AnyFlowFARTransformer3DModel

[[autodoc]] AnyFlowFARTransformer3DModel

## AnyFlowFARTransformerOutput

[[autodoc]] models.transformers.transformer_anyflow_far.AnyFlowFARTransformerOutput
