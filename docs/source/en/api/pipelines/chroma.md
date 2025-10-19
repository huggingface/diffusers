<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Chroma

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
  <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22">
</div>

Chroma is a text to image generation model based on Flux.

Original model checkpoints for Chroma can be found here:
* High-resolution finetune: [lodestones/Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD)
* Base model: [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base)
* Original repo with progress checkpoints: [lodestones/Chroma](https://huggingface.co/lodestones/Chroma) (loading this repo with `from_pretrained` will load a Diffusers-compatible version of the `unlocked-v37` checkpoint)

> [!TIP]
> Chroma can use all the same optimizations as Flux.

## Inference

```python
import torch
from diffusers import ChromaPipeline

pipe = ChromaPipeline.from_pretrained("lodestones/Chroma1-HD", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
negative_prompt =  ["low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.Generator("cpu").manual_seed(433),
    num_inference_steps=40,
    guidance_scale=3.0,
    num_images_per_prompt=1,
).images[0]
image.save("chroma.png")
```

## Loading from a single file

To use updated model checkpoints that are not in the Diffusers format, you can use the `ChromaTransformer2DModel` class to load the model from a single file in the original format. This is also useful when trying to load finetunes or quantized versions of the models that have been published by the community.

The following example demonstrates how to run Chroma from a single file.

Then run the following example

```python
import torch
from diffusers import ChromaTransformer2DModel, ChromaPipeline

model_id = "lodestones/Chroma1-HD"
dtype = torch.bfloat16

transformer = ChromaTransformer2DModel.from_single_file("https://huggingface.co/lodestones/Chroma1-HD/blob/main/Chroma1-HD.safetensors", torch_dtype=dtype)

pipe = ChromaPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=dtype)
pipe.enable_model_cpu_offload()

prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
negative_prompt =  ["low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.Generator("cpu").manual_seed(433),
    num_inference_steps=40,
    guidance_scale=3.0,
).images[0]

image.save("chroma-single-file.png")
```

## ChromaPipeline

[[autodoc]] ChromaPipeline
	- all
	- __call__

## ChromaImg2ImgPipeline

[[autodoc]] ChromaImg2ImgPipeline
	- all
	- __call__
