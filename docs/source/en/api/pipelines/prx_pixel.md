<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# PRX Pixel

PRXPixel is a pixel-space text-to-image generation model by Photoroom. A ~7B [`PRXTransformer2DModel`]
denoises raw RGB images directly — no VAE is needed. The model is conditioned on a Qwen3-VL text encoder
and uses flow matching where the transformer predicts the clean image at each step (x-prediction). The
generation resolution is fed into the timestep modulation so the model is aware of the target size.

## Available models

| Model | Resolution | Description | Suggested parameters | Recommended dtype |
|:-----:|:---------:|:----------:|:----------:|:----------:|
| [`Photoroom/prxpixel-t2i`](https://huggingface.co/Photoroom/prxpixel-t2i) | 1024 | Pixel-space ~7B model with Qwen3-VL text encoder | 28 steps, cfg=5.0 | `torch.bfloat16` |

## Loading the pipeline

[`PRXPixelPipeline`] requires `transformers >= 4.57` (the version that introduced `Qwen3VLTextModel`). Load it with [`~DiffusionPipeline.from_pretrained`]:

```py
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
image.save("prxpixel_output.png")
```

## Memory Optimization

For memory-constrained environments:

```py
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## PRXPixelPipeline

[[autodoc]] PRXPixelPipeline
  - all
  - __call__

## PRXPipelineOutput

[[autodoc]] pipelines.prx.pipeline_output.PRXPipelineOutput
