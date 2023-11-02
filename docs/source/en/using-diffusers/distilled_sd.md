<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Distilled Stable Diffusion inference

[[open-in-colab]]

Stable Diffusion inference can be a computationally intensive process because it must iteratively denoise the latents to generate an image. To reduce the computational burden, you can use a *distilled* version of the Stable Diffusion model from [Nota AI](https://huggingface.co/nota-ai). The distilled version of their Stable Diffusion model eliminates some of the residual and attention blocks from the UNet, reducing the model size by 51% and improving latency on CPU/GPU by 43%.

<Tip>

Read this [blog post](https://huggingface.co/blog/sd_distillation) to learn more about how knowledge distillation training works to produce a faster, smaller, and cheaper generative model.

</Tip>

Let's load the distilled Stable Diffusion model and compare it against the original Stable Diffusion model:

```py
from diffusers import StableDiffusionPipeline
import torch

distilled = StableDiffusionPipeline.from_pretrained(
    "nota-ai/bk-sdm-small", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")

original = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")
```

Given a prompt, get the inference time for the original model:

```py
import time

seed = 2023
generator = torch.manual_seed(seed)

NUM_ITERS_TO_RUN = 3
NUM_INFERENCE_STEPS = 25
NUM_IMAGES_PER_PROMPT = 4

prompt = "a golden vase with different flowers"

start = time.time_ns()
for _ in range(NUM_ITERS_TO_RUN):
    images = original(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    ).images
end = time.time_ns()
original_sd = f"{(end - start) / 1e6:.1f}"

print(f"Execution time -- {original_sd} ms\n")
"Execution time -- 45781.5 ms"
```

Time the distilled model inference:

```py
start = time.time_ns()
for _ in range(NUM_ITERS_TO_RUN):
    images = distilled(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    ).images
end = time.time_ns()

distilled_sd = f"{(end - start) / 1e6:.1f}"
print(f"Execution time -- {distilled_sd} ms\n")
"Execution time -- 29884.2 ms"
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/original_sd.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original Stable Diffusion (45781.5 ms)</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/distilled_sd.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">distilled Stable Diffusion (29884.2 ms)</figcaption>
  </div>
</div>

## Tiny AutoEncoder

To speed inference up even more, use a tiny distilled version of the [Stable Diffusion VAE](https://huggingface.co/sayakpaul/taesdxl-diffusers) to denoise the latents into images. Replace the VAE in the distilled Stable Diffusion model with the tiny VAE:

```py
from diffusers import AutoencoderTiny

distilled.vae = AutoencoderTiny.from_pretrained(
    "sayakpaul/taesd-diffusers", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")
```

Time the distilled model and distilled VAE inference:

```py
start = time.time_ns()
for _ in range(NUM_ITERS_TO_RUN):
    images = distilled(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    ).images
end = time.time_ns()

distilled_tiny_sd = f"{(end - start) / 1e6:.1f}"
print(f"Execution time -- {distilled_tiny_sd} ms\n")
"Execution time -- 27165.7 ms"
```

<div class="flex justify-center">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/distilled_sd_vae.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">distilled Stable Diffusion + Tiny AutoEncoder (27165.7 ms)</figcaption>
  </div>
</div>
