<!--Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Ideogram 4

Ideogram 4 is a flow-matching text-to-image model that uses a multimodal text encoder and an asymmetric
classifier-free guidance scheme: a dedicated `unconditional_transformer` produces the negative branch with zeroed text
features, while the main `transformer` consumes the full packed text + image sequence.

The pipeline defaults are the recommended settings for best quality, so a plain `pipe(prompt)` call produces
best-quality results out of the box: 48 flow-matching steps on a logit-normal schedule (`mu=0.0`, `std=1.5`) with
classifier-free guidance held at 7.0 for the main steps and dropped to 3.0 for the final 3 "polish" steps.

Key inference-time knobs are exposed via the pipeline call:

- `num_inference_steps`, `mu`, and `std` control the resolution-aware logit-normal flow-matching schedule.
- `guidance_scale` (or a full per-step `guidance_schedule`) blends the conditional and unconditional velocities.

## Text-to-image

```python
import torch
from diffusers import Ideogram4Pipeline

pipe = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-v4", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A photo of a cat holding a sign that says hello world"
# The defaults are the recommended settings for best quality.
image = pipe(prompt, height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(0)).images[0]
image.save("ideogram4.png")
```

## Ideogram4Pipeline

[[autodoc]] Ideogram4Pipeline
	- all
	- __call__

## Ideogram4PipelineOutput

[[autodoc]] pipelines.ideogram4.pipeline_output.Ideogram4PipelineOutput
