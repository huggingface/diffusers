<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Batch inference

Batch inference processes multiple prompts at a time to increase throughput. It is more efficient because processing multiple prompts at once maximizes GPU usage versus processing a single prompt and underutilizing the GPU.

The downside is increased latency because you must wait for the entire batch to complete, and more GPU memory is required for large batches.

To generate a batch of images, pass a list of prompts or images to the pipeline.

<hfoptions id="usage">
<hfoption id="text-to-image">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

prompts = [
    "cinematic photo of A beautiful sunset over mountains, 35mm photograph, film, professional, 4k, highly detailed",
    "cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
    "pixel-art a cozy coffee shop interior, low-res, blocky, pixel art style, 8-bit graphics"
]

images = pipeline(
    prompt=prompts,
).images

for i, image in enumerate(images):
    image.save(f"batch_image_{i}.png")
    print(f"Generated image {i+1} for prompt: {prompts[i]}")
```

</hfoption>
<hfoption id="image-to-image">

```py
import torch
from diffusers.utils import load_image
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

input_images = [
    load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"),
    load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"),
    load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/detail-prompt.png")
]

prompts = [
    "cinematic photo of a beautiful sunset over mountains, 35mm photograph, film, professional, 4k, highly detailed",
    "cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
    "pixel-art a cozy coffee shop interior, low-res, blocky, pixel art style, 8-bit graphics"
]

images = pipeline(
    prompt=prompts,
    image=input_images,
    guidance_scale=8.0,
    strength=0.5
).images

for i, image in enumerate(images):
    image.save(f"batch_image_{i}.png")
    print(f"Generated image {i+1} for prompt: {prompts[i]}")
```

</hfoption>
</hfoptions>

## Deterministic generation

Enable reproducible batch generation by passing a list of [Generator’s](https://pytorch.org/docs/stable/generated/torch.Generator.html) to the pipeline and tie each `Generator` to a seed to reuse it.

Use a list comprehension to iterate over the batch size specified in `range()` to create a unique `Generator` object for each image in the batch. Don't multiply the `Generator` by the batch size because that only creates one `Generator` object that is used sequentially for each image in the batch.

Pass the `geneator` to the pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(3)]
prompts = [
    "cinematic photo of A beautiful sunset over mountains, 35mm photograph, film, professional, 4k, highly detailed",
    "cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
    "pixel-art a cozy coffee shop interior, low-res, blocky, pixel art style, 8-bit graphics"
]

images = pipeline(
    prompt=prompts,
    generator=generator
).images

for i, image in enumerate(images):
    image.save(f"batch_image_{i}.png")
    print(f"Generated image {i+1} for prompt: {prompts[i]}")
```

You can use this to iteratively select an image associated with a seed and then improve on it by crafting a more detailed prompt.