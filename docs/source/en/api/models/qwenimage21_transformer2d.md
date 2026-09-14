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

Two config flags set 2.1 apart from earlier QwenImage transformers. Neither adds parameters:

- `causal_block` — attention follows `(q_idx >= kv_idx) or same_image_block`, so the joint sequence is causal while
  each image block (every condition image and the target image) stays internally bidirectional. This requires the
  `flex` attention backend, since the mask is a `torch.nn.attention.flex_attention.BlockMask`.
- `causal_condition` — text and condition-image tokens are modulated from `t = 0` rather than the sampled timestep.
  Their activations are therefore independent of the denoising step, which is what makes the keys and values of that
  prefix cacheable across steps via the `kv_cache` argument.

The model can be loaded with the following code snippet.

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
