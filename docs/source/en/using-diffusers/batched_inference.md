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

<hfoptions id="usage">
<hfoption id="text-to-image">

For text-to-image, pass a list of prompts to the pipeline.

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

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

To generate multiple variations of one prompt, use the `num_images_per_prompt` argument.

```py
import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

images = pipeline(
    prompt="pixel-art a cozy coffee shop interior, low-res, blocky, pixel art style, 8-bit graphics",
    num_images_per_prompt=4
).images

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Combine both approaches to generate different variations of different prompts.

```py
images = pipeline(
    prompt=prompts,
    num_images_per_prompt=2,
).images

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

</hfoption>
<hfoption id="image-to-image">

For image-to-image, pass a list of input images and prompts to the pipeline.

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

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

To generate multiple variations of one prompt, use the `num_images_per_prompt` argument.

```py
import torch
import matplotlib.pyplot as plt
from diffusers.utils import load_image
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/detail-prompt.png")

images = pipeline(
    prompt="pixel-art a cozy coffee shop interior, low-res, blocky, pixel art style, 8-bit graphics",
    image=input_image,
    num_images_per_prompt=4
).images

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Combine both approaches to generate different variations of different prompts.

```py
input_images = [
    load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"),
    load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/detail-prompt.png")
]

prompts = [
    "cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
    "pixel-art a cozy coffee shop interior, low-res, blocky, pixel art style, 8-bit graphics"
]

images = pipeline(
    prompt=prompts,
    image=input_images,
    num_images_per_prompt=2,
).images

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

</hfoption>
</hfoptions>

## Deterministic generation

Enable reproducible batch generation by passing a list of [Generator’s](https://pytorch.org/docs/stable/generated/torch.Generator.html) to the pipeline and tie each `Generator` to a seed to reuse it.

Use a list comprehension to iterate over the batch size specified in `range()` to create a unique `Generator` object for each image in the batch.

Don't multiply the `Generator` by the batch size because that only creates one `Generator` object that is used sequentially for each image in the batch.

```py
generator = [torch.Generator(device="cuda").manual_seed(0)] * 3
```

Pass the `generator` to the pipeline.

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

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

You can use this to iteratively select an image associated with a seed and then improve on it by crafting a more detailed prompt.