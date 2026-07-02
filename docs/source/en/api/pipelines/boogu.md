<!--Copyright 2025 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
-->

# Boogu-Image

## Overview

Boogu-Image is an instruction-driven image generation and editing model. Rather than a
plain text prompt, it is conditioned on a natural-language *instruction* that is encoded
by a Qwen3-VL multimodal LLM, which can also attend to optional reference images. A
single/double-stream transformer denoiser then predicts the latent updates, and a
flow-matching scheduler with training-aligned time shifting controls the denoising
trajectory. The VAE maps between image and latent space.

The model is released in several variants:

- **Base** (`Boogu/Boogu-Image-0.1-Base`) — text-to-image, full sampling schedule.
- **Turbo** (`Boogu/Boogu-Image-0.1-Turbo`) — DMD student model for few-step
  text-to-image generation.
- **Edit** (`Boogu/Boogu-Image-0.1-Edit`) — instruction-based image editing conditioned
  on one or more reference images.

FP8-quantized checkpoints are also available for each variant (the `-fp8` suffix).

There are two pipeline classes:

- [`BooguImagePipeline`] — text-to-image and instruction editing.
- [`BooguImageTurboPipeline`] — a subclass adding the DMD few-step inference path. It
  defaults the guidance scales to the DMD-required values (`text_guidance_scale=1.0`,
  `image_guidance_scale=1.0`, `empty_instruction_guidance_scale=0.0`).

## Usage examples

### Text-to-image

```python
import torch
from diffusers.pipelines.boogu import BooguImagePipeline

pipe = BooguImagePipeline.from_pretrained("Boogu/Boogu-Image-0.1-Base", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    instruction="A serene Chinese ink-wash landscape of the Guilin mountains bathed in golden light, layered peaks, mirror-like river, glowing golden contours.",
    height=1024,
    width=1024,
    num_inference_steps=50,
    text_guidance_scale=4.0,
).images[0]

image.save("base.png")
```

### Few-step generation (Turbo)

```python
import torch
from diffusers.pipelines.boogu import BooguImageTurboPipeline

pipe = BooguImageTurboPipeline.from_pretrained("Boogu/Boogu-Image-0.1-Turbo", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    instruction="A serene Chinese ink-wash landscape of the Guilin mountains bathed in golden light.",
    height=1024,
    width=1024,
    num_inference_steps=4,
).images[0]

image.save("turbo.png")
```

### Instruction-based editing

Pass one or more reference images through `input_images`:

```python
import torch
from PIL import Image
from diffusers.pipelines.boogu import BooguImagePipeline

pipe = BooguImagePipeline.from_pretrained("Boogu/Boogu-Image-0.1-Edit", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    instruction="Turn the image into a colored-pencil illustration.",
    input_images=[Image.open("base.png").convert("RGB")],
    height=1024,
    width=1024,
    num_inference_steps=50,
    text_guidance_scale=4.0,
    image_guidance_scale=1.0,
).images[0]

image.save("edit.png")
```

### FP8 checkpoints

FP8 weights are stored in a non-safetensors format, so load the transformer separately
with `use_safetensors=False` and pass it to the pipeline:

```python
import torch
from diffusers import BooguImageTransformer2DModel
from diffusers.pipelines.boogu import BooguImagePipeline

transformer = BooguImageTransformer2DModel.from_pretrained(
    "Boogu/Boogu-Image-0.1-Base-fp8",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_safetensors=False,
)
pipe = BooguImagePipeline.from_pretrained(
    "Boogu/Boogu-Image-0.1-Base-fp8", torch_dtype=torch.bfloat16, transformer=transformer
)
pipe = pipe.to("cuda")
```

Runnable scripts for every variant are available in
[`examples/boogu`](https://github.com/huggingface/diffusers/tree/main/examples/boogu).

> [!TIP]
> The transformer uses fused `triton` (RMSNorm) and `flash_attn` (SwiGLU, variable-length
> attention) kernels when they are installed, and falls back to pure PyTorch otherwise.

## BooguImagePipeline

[[autodoc]] pipelines.boogu.pipeline_boogu.BooguImagePipeline
  - all
  - __call__

## BooguImageTurboPipeline

[[autodoc]] pipelines.boogu.pipeline_boogu_turbo.BooguImageTurboPipeline
  - all
  - __call__

## FMPipelineOutput

[[autodoc]] pipelines.boogu.pipeline_boogu.FMPipelineOutput
