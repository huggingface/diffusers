<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Batch inference

Batch inference processes multiple prompts at once to increase throughput. A batch keeps the GPU busy, while a single prompt leaves it underused.

The downside is higher latency, because you wait for the entire batch to finish. Large batches also need more GPU memory.

## Batch multiple prompts

Pass a list of prompts to the pipeline to generate one image for each prompt.

```py
import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
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

## Generate several images per prompt

To generate multiple variations of one prompt, use the `num_images_per_prompt` argument.

```py
import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
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

## Generate several images for several prompts

Combine both approaches to generate different variations of different prompts. This example reuses the `pipeline` and `prompts` from the examples above.

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

## Generate reproducible batches

To make a batch reproducible, pass the pipeline a list of [Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) objects, one for each image, each with its own seed.

> [!TIP]
> Refer to the [Reproducibility](./reusing_seeds) docs to learn more about deterministic algorithms and the `Generator` object.

Create one `Generator` per image with a list comprehension. Multiplying a list (`[generator] * 3`) gives you three references to the same `Generator`, so the images don't get independent seeds.

```py
# don't do this
generator = [torch.Generator(device="cuda").manual_seed(0)] * 3  # or "mps", "xpu", "cpu"
```

The number of generators must match the number of images in the batch. Pass the `generator` to the pipeline.

```py
import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="cuda"
)

prompts = [
    "Cinematic shot of a cozy coffee shop interior, warm pastel light streaming through a window where a cat rests. Shallow depth of field, glowing cups in soft focus, dreamy lofi-inspired mood, nostalgic tones, framed like a quiet film scene.",
    "Polaroid-style photograph of a cozy coffee shop interior, bathed in warm pastel light. A cat sits on the windowsill near steaming mugs. Soft, slightly faded tones and dreamy blur evoke nostalgia, a lofi mood, and the intimate, imperfect charm of instant film.",
    "Soft watercolor illustration of a cozy coffee shop interior, pastel washes of color filling the space. A cat rests peacefully on the windowsill as warm light glows through. Gentle brushstrokes create a dreamy, lofi-inspired atmosphere with whimsical textures and nostalgic calm.",
    "Isometric pixel-art illustration of a cozy coffee shop interior in detailed 8-bit style. Warm pastel light fills the space as a cat rests on the windowsill. Blocky furniture and tiny mugs add charm, low-res retro graphics enhance the nostalgic, lofi-inspired game aesthetic."
]
generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(len(prompts))]  # or "mps", "xpu", "cpu"

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

Each image is tied to a seed, so you can pick an image you like and regenerate it with the same seed while you refine its prompt.

## Next steps

- When a batch doesn't fit on one GPU, or you have more prompts than one GPU can process, spread the prompts across GPUs with [Data parallelism](../optimization/data_parallelism).