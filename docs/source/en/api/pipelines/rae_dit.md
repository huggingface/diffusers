<!-- Copyright 2026 The NYU Vision-X and HuggingFace Teams. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# RAE DiT

[Diffusion Transformers with Representation Autoencoders](https://huggingface.co/papers/2510.11690) introduces a
two-stage recipe: first train a representation autoencoder (RAE), then train a diffusion transformer on the resulting
latent space.

[`RAEDiTPipeline`] implements the Stage-2 class-conditional generator in Diffusers. It combines:

- [`RAEDiT2DModel`] for latent denoising
- [`FlowMatchEulerDiscreteScheduler`] for the denoising trajectory
- [`AutoencoderRAE`] for decoding latent samples to RGB images

> [!TIP]
> [`RAEDiTPipeline`] expects a Stage-2 checkpoint converted to Diffusers format together with a compatible
> [`AutoencoderRAE`] checkpoint.

## Loading a converted pipeline

```python
import torch
from diffusers import RAEDiTPipeline

pipe = RAEDiTPipeline.from_pretrained(
    "path/to/converted-rae-dit-imagenet256",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(class_labels=[207], num_inference_steps=25).images[0]
image.save("golden_retriever.png")
```

If the converted pipeline includes an `id2label` mapping, you can also look up class ids by name:

```python
class_id = pipe.get_label_ids("golden retriever")[0]
image = pipe(class_labels=[class_id], num_inference_steps=25).images[0]
```

## RAEDiTPipeline

[[autodoc]] RAEDiTPipeline
  - all
  - __call__

## RAEDiTPipelineOutput

[[autodoc]] RAEDiTPipelineOutput
