<!--Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Krea 2

Krea 2 (K2) is a flow-matching text-to-image model built around a single-stream MMDiT with grouped-query attention. A
Qwen3-VL text encoder provides the conditioning: instead of the last hidden state, hidden states from twelve decoder
layers are tapped per token and fused inside the transformer by a small text-fusion stage. Images are decoded with the
Qwen-Image VAE.

Two checkpoints are released, sharing the same architecture but with different recommended sampler settings:

- **Base (midtrain)** — use the full sampler with classifier-free guidance: `num_inference_steps=28`,
  `guidance_scale=4.5`.
- **TDM (distilled)** — distilled for few-step sampling, run with `num_inference_steps=8` and guidance disabled
  (`guidance_scale=0.0`).

`guidance_scale` follows the Krea 2 convention: the velocity is computed as `cond + guidance_scale * (cond - uncond)`
and guidance is enabled whenever `guidance_scale > 0` (this equals the usual CFG formulation with scale
`1 + guidance_scale`).

## Text-to-image

```python
import torch
from diffusers import Krea2Pipeline

# Load from a local directory produced by the Krea 2 conversion (no hub repo yet).
pipe = Krea2Pipeline.from_pretrained("path/to/krea2-diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "a fox in the snow"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=4.5,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2.png")
```

## Krea2Pipeline

[[autodoc]] Krea2Pipeline
  - all
  - __call__

## Krea2PipelineOutput

[[autodoc]] pipelines.krea2.pipeline_output.Krea2PipelineOutput

## Krea2ModularPipeline

[[autodoc]] Krea2ModularPipeline

## Krea2AutoBlocks

[[autodoc]] Krea2AutoBlocks
