<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Lens

<div class="flex flex-wrap space-x-1">
</div>

Lens is a 3.8B-parameter foundational text-to-image model designed for efficient training and fast high-resolution generation. It combines dense-caption pre-training, mixed-resolution learning, GPT-OSS multi-layer text features, and the FLUX.2 semantic VAE to reach competitive quality with substantially less training compute than larger T2I models. For more details, please refer to the [model card](https://huggingface.co/microsoft/Lens).

The abstract from the paper is:

*Lens is a 3.8B-parameter foundational text-to-image model designed for efficient training and fast high-resolution generation. It combines dense-caption pre-training, mixed-resolution learning, GPT-OSS multi-layer text features, and the FLUX.2 semantic VAE to reach competitive quality with substantially less training compute than larger T2I models.*

## Usage Example

```python
import torch
from diffusers import LensPipeline

pipe = LensPipeline.from_pretrained("microsoft/Lens", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1440,
    width=1440,
    num_inference_steps=20,
    guidance_scale=5.0,
).images[0]
image.save("lens.png")
```

## LensPipeline

[[autodoc]] LensPipeline

- all
- __call__

## LensPipelineOutput

[[autodoc]] pipelines.lens.pipeline_output.LensPipelineOutput
