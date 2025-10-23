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

For text-to-image, pass a list of prompts to the pipeline and for image-to-image, pass a list of images and prompts to the pipeline. The example below demonstrates batched text-to-image inference.

```py
import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)

prompts = [
    "Cinematic shot of a cozy coffee shop interior, warm pastel light streaming through a window where a cat rests. Shallow depth of field, glowing cups in soft focus, dreamy lofi-inspired mood, nostalgic tones, framed like a quiet film scene.",
    "Polaroid-style photograph of a cozy coffee shop interior, bathed in warm pastel light. A cat sits on the windowsill near steaming mugs. Soft, slightly faded tones and dreamy blur evoke nostalgia, a lofi mood, and the intimate, imperfect charm of instant film.",
    "Soft watercolor illustration of a cozy coffee shop interior, pastel washes of color filling the space. A cat rests peacefully on the windowsill as warm light glows through. Gentle brushstrokes create a dreamy, lofi-inspired atmosphere with whimsical textures and nostalgic calm.",
    "Isometric pixel-art illustration of a cozy coffee shop interior in detailed 8-bit style. Warm pastel light fills the space as a cat rests on the windowsill. Blocky furniture and tiny mugs add charm, low-res retro graphics enhance the nostalgic, lofi-inspired game aesthetic."
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

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/batch-inference.png"/>
</div>

To generate multiple variations of one prompt, use the `num_images_per_prompt` argument.

```py
import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)

prompt="""
Isometric pixel-art illustration of a cozy coffee shop interior in detailed 8-bit style. Warm pastel light fills the
space as a cat rests on the windowsill. Blocky furniture and tiny mugs add charm, low-res retro graphics enhance the
nostalgic, lofi-inspired game aesthetic.
"""

images = pipeline(
    prompt=prompt,
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

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/batch-inference-2.png"/>
</div>

Combine both approaches to generate different variations of different prompts.

```py
images = pipeline(
    prompt=prompts,
    num_images_per_prompt=2,
).images

fig, axes = plt.subplots(2, 4, figsize=(12, 12))
axes = axes.flatten()

for i, image in enumerate(images):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/batch-inference-3.png"/>
</div>

## Deterministic generation

Enable reproducible batch generation by passing a list of [Generator’s](https://pytorch.org/docs/stable/generated/torch.Generator.html) to the pipeline and tie each `Generator` to a seed to reuse it.

> [!TIP]
> Refer to the [Reproducibility](./reusing_seeds) docs to learn more about deterministic algorithms and the `Generator` object.

Use a list comprehension to iterate over the batch size specified in `range()` to create a unique `Generator` object for each image in the batch. Don't multiply the `Generator` by the batch size because that only creates one `Generator` object that is used sequentially for each image in the batch.

```py
generator = [torch.Generator(device="cuda").manual_seed(0)] * 3
```

Pass the `generator` to the pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)

generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(3)]
prompts = [
    "Cinematic shot of a cozy coffee shop interior, warm pastel light streaming through a window where a cat rests. Shallow depth of field, glowing cups in soft focus, dreamy lofi-inspired mood, nostalgic tones, framed like a quiet film scene.",
    "Polaroid-style photograph of a cozy coffee shop interior, bathed in warm pastel light. A cat sits on the windowsill near steaming mugs. Soft, slightly faded tones and dreamy blur evoke nostalgia, a lofi mood, and the intimate, imperfect charm of instant film.",
    "Soft watercolor illustration of a cozy coffee shop interior, pastel washes of color filling the space. A cat rests peacefully on the windowsill as warm light glows through. Gentle brushstrokes create a dreamy, lofi-inspired atmosphere with whimsical textures and nostalgic calm.",
    "Isometric pixel-art illustration of a cozy coffee shop interior in detailed 8-bit style. Warm pastel light fills the space as a cat rests on the windowsill. Blocky furniture and tiny mugs add charm, low-res retro graphics enhance the nostalgic, lofi-inspired game aesthetic."
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

You can use this to select an image associated with a seed and iteratively improve on it by crafting a more detailed prompt.