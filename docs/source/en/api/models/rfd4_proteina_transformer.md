<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# RFD4ProteinaTransformer

The denoiser of [RFD4-Proteína](../pipelines/rfd4_proteina). It predicts velocities for the two modalities the
sampler integrates — backbone CA positions and per-token local latents — from prepared molecular tensors. Inputs are
named tensors plus a conditioning mapping whose contents the checkpoint configuration selects; structure parsing is
outside this class.

```py
from diffusers import RFD4ProteinaTransformer

transformer = RFD4ProteinaTransformer.from_pretrained(
    "nvidia/RFD4-Proteina-Diffusers", subfolder="transformer"
)
```

## RFD4ProteinaTransformer

[[autodoc]] RFD4ProteinaTransformer
  - forward

## RFD4ProteinaTransformerOutput

[[autodoc]] models.transformers.transformer_rfd4_proteina.RFD4ProteinaTransformerOutput
