<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SeFi-Image

[SeFi-Image](https://github.com/jmliu206/SeFi-Image) is a text-to-image model family based on Semantic-First Diffusion.
It denoises a semantic latent stream slightly ahead of a texture latent stream, then decodes the final texture latents
to images.

The public checkpoints on the Hub are gated and distributed under a non-commercial license. Accept the checkpoint
license on the Hub before downloading them. Original SeFi-Image artifacts can be converted with
`scripts/convert_sefi_to_diffusers.py` before loading them with [`SeFiPipeline`].

```python
import torch
from diffusers import SeFiPipeline

pipe = SeFiPipeline.from_pretrained("./sefi-1b-base-diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe("A red apple on a wooden table.").images[0]
image.save("sefi.png")
```

Convert the original artifacts first:

```bash
python scripts/convert_sefi_to_diffusers.py \
  --checkpoint SeFi-Image/SeFi-Image-1B-Base \
  --output ./sefi-1b-base-diffusers \
  --token $HF_TOKEN
```

Turbo checkpoints are distilled and support 4, 8, or 10 inference steps with `guidance_scale=1.0`.

## SeFiPipeline

[[autodoc]] SeFiPipeline
  - all

## SeFiPipelineOutput

[[autodoc]] pipelines.sefi.pipeline_output.SeFiPipelineOutput
