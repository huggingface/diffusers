<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Qwen-Image 2.1

Qwen-Image 2.1 encodes the prompt and any condition images together with a Qwen3-VL model, then denoises the target
image with a single-stream block-causal transformer. See
[`QwenImage21Transformer2DModel`](../models/qwenimage21_transformer2d) for details on block-causal attention and
`causal_condition`.

The `flex` attention backend (`torch.nn.attention.flex_attention`) gives efficient single-pass block-causal attention.
Without it, the model uses an exact multi-pass SDPA prefill that processes each image block with bidirectional
attention and text with causal attention, matching the block-causal mask exactly. Both paths produce the same results.

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16).to("cuda")

# Text-to-image
image = pipe("A capybara wearing a wizard hat, oil painting", num_inference_steps=40).images[0]
image.save("t2i.png")

# Image-conditioned editing
edited = pipe("Move it to a snowy mountain top", image=image, num_inference_steps=40).images[0]
edited.save("edit.png")
```

## QwenImage21Pipeline

[[autodoc]] QwenImage21Pipeline
    - all
    - __call__

## QwenImagePipelineOutput

[[autodoc]] pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput
