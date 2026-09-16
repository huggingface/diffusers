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
[`QwenImage21Transformer2DModel`](../models/qwenimage21_transformer2d) for block-causal attention, the attention
processors, and `causal_condition`.

The defaults are the values Qwen recommends: 40 steps and no guidance. Pass a `negative_prompt` together with
`true_cfg_scale > 1` to turn classifier-free guidance on, which doubles the work per step.

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16).to("cuda")

# Text-to-image
image = pipe("A capybara wearing a wizard hat, oil painting").images[0]
image.save("t2i.png")

# Image-conditioned editing
edited = pipe("Move it to a snowy mountain top", image=image).images[0]
edited.save("edit.png")
```

## Faster attention with flex_attention

The default `QwenImage21AttnProcessor` runs the block-causal prefill as one attention call per prefix segment. It
needs no compilation and works on any PyTorch build. `QwenImage21FlexAttnProcessor` expresses the same mask as a
single `flex_attention` call, which is faster once the model is compiled.

> [!TIP]
> Compile the model when you switch to the flex processor. An uncompiled `flex_attention` materializes the full
> attention score matrix in fp32, which is much slower and runs out of memory at high resolution.

```python
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
pipe.transformer.compile()
```

## QwenImage21Pipeline

[[autodoc]] QwenImage21Pipeline
    - all
    - __call__

## QwenImagePipelineOutput

[[autodoc]] pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput
