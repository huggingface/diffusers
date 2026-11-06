<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# QwenImage21Transformer2DModel

The single-stream transformer used by Qwen-Image 2.1. Text and image latents share one sequence, and a single shared
`modulation` projection feeds every block.

Two behaviours distinguish 2.1 from earlier QwenImage transformers:

- **Block-causal attention** — attention follows `(q_idx >= kv_idx) or same_image_block`, so the joint sequence is
  causal while each image block stays internally bidirectional. `QwenImage21AttnProcessor` implements it as one
  attention call per prefix segment and is the default. `QwenImage21FlexAttnProcessor` implements it as a single
  `flex_attention` call driven by a `BlockMask`, which is faster once the model is compiled. Both produce the same
  results.
- `causal_condition` — text and condition-image tokens are modulated from `t = 0` rather than the sampled timestep.
  Their activations are independent of the denoising step, so the keys and values of that prefix are cacheable
  across steps via the `kv_cache` argument.

Load it with:

```python
import torch
from diffusers import QwenImage21Transformer2DModel

transformer = QwenImage21Transformer2DModel.from_pretrained(
    "Qwen/Qwen-Image-2.1", subfolder="transformer", dtype=torch.bfloat16
)
```

## QwenImage21Transformer2DModel

[[autodoc]] QwenImage21Transformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
