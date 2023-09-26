<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Inpainting

[[open-in-colab]]

Inpainting replaces or edits specified areas of an image. This makes it a useful tool for restoring old images, removing defects and artifacts, or even replacing areas of an image with something entirely new. Inpainting relies on a mask image to determine which regions of an image to fill in; the area to replace is represented by white pixels and the area to keep is represented by black pixels. The mask determines which areas containing the missing pixels to fill in with the prompt.

With 🤗 Diffusers, here is how you can quickly start inpainting:

1. Load an inpainting checkpoint with the [`AutoPipelineForInpainting`] class. This'll automatically detect the appropriate pipeline class to load based on the checkpoint:

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()
```

<Tip>

You'll notice throughout the guide, we use [`~DiffusionPipeline.enable_model_cpu_offload`] and [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`], to save memory and increase inference speed. If you're using PyTorch 2.0, then you don't need to call [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`] on your pipeline because it'll already be using PyTorch 2.0's native [scaled-dot product attention](/optimization/torch2.0#scaled-dot-product-attention).

</Tip>

2. Load your base image and mask image:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")
```

3. Create a prompt to inpaint the image with and pass it to the pipeline with the base and mask images:

```py
prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">base image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-cat.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

## Popular models

[Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint) are among the most popular models for inpainting. SDXL typically produces higher resolution images than Stable Diffusion v1.5, and Kandinsky 2.2 is also capable of generating high-quality images thanks to an image prior model that creates better embeddings.

### Stable Diffusion v1.5

Stable Diffusion v1.5 is a latent diffusion model finetuned on 512x512 images. It is a good starting point for inpainting because it is relatively fast and produces images with good quality. To use this model for inpainting, you'll need to pass a prompt, base image, and mask image to the pipeline:

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

generator = torch.Generator("cuda").manual_seed(13)
prompt = "concept art digital painting of a fantasy castle, inspired by lord of the rings, deviantart, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
image
```

### Stable Diffusion XL (SDXL)

SDXL is a larger and more powerful version of Stable Diffusion v1.5. It employs a two-stage model process (though each model can also be used alone); the base model generates an image, and a refiner model takes that image and further enhances the details and quality of it. Take a look at the [SDXL](sdxl) guide for a more comprehensive guide on how to use SDXL and configure it's parameters.

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

generator = torch.Generator("cuda").manual_seed(13)
prompt = "concept art digital painting of a fantasy castle, inspired by lord of the rings, deviantart, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
image
```

### Kandinsky 2.2

The Kandinsky model family is similar to SDXL in the sense that it uses two models; the image prior model generates image embeddings, and the diffusion model uses these embeddings to generate higher quality images. You can load the image prior and diffusion model separately, but the easiest way to use Kandinsky 2.2 is to load it into the [`AutoPipelineForInpainting`] class which uses the [`KandinskyV22InpaintCombinedPipeline`] under the hood.

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

generator = torch.Generator("cuda").manual_seed(13)
prompt = "concept art digital painting of a fantasy castle, inspired by lord of the rings, deviantart, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">base image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdv1.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion v1.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdxl.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion XL</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-kandinsky.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Kandinsky 2.2</figcaption>
  </div>
</div>
