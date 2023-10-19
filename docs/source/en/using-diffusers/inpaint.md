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

Inpainting replaces or edits specific areas of an image. This makes it a useful tool for image restoration like removing defects and artifacts, or even replacing an image area with something entirely new. Inpainting relies on a mask to determine which regions of an image to fill in; the area to inpaint is represented by white pixels and the area to keep is represented by black pixels. The white pixels are filled in by the prompt.

With 🤗 Diffusers, here is how you can do inpainting:

1. Load an inpainting checkpoint with the [`AutoPipelineForInpainting`] class. This'll automatically detect the appropriate pipeline class to load based on the checkpoint:

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()
```

<Tip>

You'll notice throughout the guide, we use [`~DiffusionPipeline.enable_model_cpu_offload`] and [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`], to save memory and increase inference speed. If you're using PyTorch 2.0, it's not necessary to call [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`] on your pipeline because it'll already be using PyTorch 2.0's native [scaled-dot product attention](../optimization/torch2.0#scaled-dot-product-attention).

</Tip>

2. Load the base and mask images:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")
```

3. Create a prompt to inpaint the image with and pass it to the pipeline with the base and mask images:

```py
prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
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

## Create a mask image

Throughout this guide, the mask image is provided in all of the code examples for convenience. You can inpaint on your own images, but you'll need to create a mask image for it. Use the Space below to easily create a mask image.

Upload a base image to inpaint on and use the sketch tool to draw a mask. Once you're done, click **Run** to generate and download the mask image.

<iframe
	src="https://stevhliu-inpaint-mask-maker.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

## Popular models

[Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting), [Stable Diffusion XL (SDXL) Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), and [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint) are among the most popular models for inpainting. SDXL typically produces higher resolution images than Stable Diffusion v1.5, and Kandinsky 2.2 is also capable of generating high-quality images.

### Stable Diffusion Inpainting

Stable Diffusion Inpainting is a latent diffusion model finetuned on 512x512 images on inpainting. It is a good starting point because it is relatively fast and generates good quality images. To use this model for inpainting, you'll need to pass a prompt, base and mask image to the pipeline:

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
```

### Stable Diffusion XL (SDXL) Inpainting

SDXL is a larger and more powerful version of Stable Diffusion v1.5. This model can follow a two-stage model process (though each model can also be used alone); the base model generates an image, and a refiner model takes that image and further enhances its details and quality. Take a look at the [SDXL](sdxl) guide for a more comprehensive guide on how to use SDXL and configure it's parameters.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
```

### Kandinsky 2.2 Inpainting

The Kandinsky model family is similar to SDXL because it uses two models as well; the image prior model creates image embeddings, and the diffusion model generates images from them. You can load the image prior and diffusion model separately, but the easiest way to use Kandinsky 2.2 is to load it into the [`AutoPipelineForInpainting`] class which uses the [`KandinskyV22InpaintCombinedPipeline`] under the hood.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">base image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdv1.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion Inpainting</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdxl.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion XL Inpainting</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-kandinsky.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Kandinsky 2.2 Inpainting</figcaption>
  </div>
</div>

## Configure pipeline parameters

Image features - like quality and "creativity" - are dependent on pipeline parameters. Knowing what these parameters do is important for getting the results you want. Let's take a look at the most important parameters and see how changing them affects the output.

### Strength

`strength` is a measure of how much noise is added to the base image, which influences how similar the output is to the base image.

* 📈 a high `strength` value means more noise is added to an image and the denoising process takes longer, but you'll get higher quality images that are more different from the base image
* 📉 a low `strength` value means less noise is added to an image and the denoising process is faster, but the image quality may not be as great and the generated image resembles the base image more

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.6).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-0.6.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.6</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-0.8.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.8</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-1.0.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 1.0</figcaption>
  </div>
</div>

### Guidance scale

`guidance_scale` affects how aligned the text prompt and generated image are.

* 📈 a high `guidance_scale` value means the prompt and generated image are closely aligned, so the output is a stricter interpretation of the prompt
* 📉 a low `guidance_scale` value means the prompt and generated image are more loosely aligned, so the output may be more varied from the prompt

You can use `strength` and `guidance_scale` together for more control over how expressive the model is. For example, a combination high `strength` and `guidance_scale` values gives the model the most creative freedom.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=2.5).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-2.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 2.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-7.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 7.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-12.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 12.5</figcaption>
  </div>
</div>

### Negative prompt

A negative prompt assumes the opposite role of a prompt; it guides the model away from generating certain things in an image. This is useful for quickly improving image quality and preventing the model from generating things you don't want.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
negative_prompt = "bad architecture, unstable, poor details, blurry"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
image
```

<div class="flex justify-center">
  <figure>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-negative.png" />
    <figcaption class="text-center">negative_prompt = "bad architecture, unstable, poor details, blurry"</figcaption>
  </figure>
</div>

## Preserve unmasked areas

The [`AutoPipelineForInpainting`] (and other inpainting pipelines) generally changes the unmasked parts of an image to create a more natural transition between the masked and unmasked region. If this behavior is undesirable, you can force the unmasked area to remain the same. However, forcing the unmasked portion of the image to remain the same may result in some unusual transitions between the unmasked and masked areas.

```py
import PIL
import numpy as np
import torch

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

device = "cuda"
pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to(device)

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
repainted_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
repainted_image.save("repainted_image.png")

# Convert mask to grayscale NumPy array
mask_image_arr = np.array(mask_image.convert("L"))
# Add a channel dimension to the end of the grayscale mask
mask_image_arr = mask_image_arr[:, :, None]
# Binarize the mask: 1s correspond to the pixels which are repainted
mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
mask_image_arr[mask_image_arr < 0.5] = 0
mask_image_arr[mask_image_arr >= 0.5] = 1

# Take the masked pixels from the repainted image and the unmasked pixels from the initial image
unmasked_unchanged_image_arr = (1 - mask_image_arr) * init_image + mask_image_arr * repainted_image
unmasked_unchanged_image = PIL.Image.fromarray(unmasked_unchanged_image_arr.round().astype("uint8"))
unmasked_unchanged_image.save("force_unmasked_unchanged.png")
```

## Chained inpainting pipelines

[`AutoPipelineForInpainting`] can be chained with other 🤗 Diffusers pipelines to edit their outputs. This is often useful for improving the output quality from your other diffusion pipelines, and if you're using multiple pipelines, it can be more memory-efficient to chain them together to keep the outputs in latent space and reuse the same pipeline components.

### Text-to-image-to-inpaint

Chaining a text-to-image and inpainting pipeline allows you to inpaint the generated image, and you don't have to provide a base image to begin with. This makes it convenient to edit your favorite text-to-image outputs without having to generate an entirely new image.

Start with the text-to-image pipeline to create a castle:

```py
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline("concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k").images[0]
```

Load the mask image of the output from above:

```py
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_text-chain-mask.png").convert("RGB")
```

And let's inpaint the masked area with a waterfall:

```py
pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

prompt = "digital painting of a fantasy waterfall, cloudy"
image = pipeline(prompt=prompt, image=image, mask_image=mask_image).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-text-chain.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">text-to-image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-text-chain-out.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">inpaint</figcaption>
  </div>
</div>


### Inpaint-to-image-to-image

You can also chain an inpainting pipeline before another pipeline like image-to-image or an upscaler to improve the quality.

Begin by inpainting an image:

```py
import torch
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# resize image to 1024x1024 for SDXL
image = image.resize((1024, 1024))
```

Now let's pass the image to another inpainting pipeline with SDXL's refiner model to enhance the image details and quality:

```py
pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt=prompt, image=image, mask_image=mask_image, output_type="latent").images[0]
```

<Tip>

It is important to specify `output_type="latent"` in the pipeline to keep all the outputs in latent space to avoid an unnecessary decode-encode step. This only works if the chained pipelines are using the same VAE. For example, in the [Text-to-image-to-inpaint](#text-to-image-to-inpaint) section, Kandinsky 2.2 uses a different VAE class than the Stable Diffusion model so it won't work. But if you use Stable Diffusion v1.5 for both pipelines, then you can keep everything in latent space because they both use [`AutoencoderKL`].

</Tip>

Finally, you can pass this image to an image-to-image pipeline to put the finishing touches on it. It is more efficient to use the [`~AutoPipelineForImage2Image.from_pipe`] method to reuse the existing pipeline components, and avoid unnecessarily loading all the pipeline components into memory again.

```py
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt=prompt, image=image).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-to-image-chain.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">inpaint</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-to-image-final.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">image-to-image</figcaption>
  </div>
</div>

Image-to-image and inpainting are actually very similar tasks. Image-to-image generates a new image that resembles the existing provided image. Inpainting does the same thing, but it only transforms the image area defined by the mask and the rest of the image is unchanged. You can think of inpainting as a more precise tool for making specific changes and image-to-image has a broader scope for making more sweeping changes.

## Control image generation

Getting an image to look exactly the way you want is challenging because the denoising process is random. While you can control certain aspects of generation by configuring parameters like `negative_prompt`, there are better and more efficient methods for controlling image generation.

### Prompt weighting

Prompt weighting provides a quantifiable way to scale the representation of concepts in a prompt. You can use it to increase or decrease the magnitude of the text embedding vector for each concept in the prompt, which subsequently determines how much of each concept is generated. The [Compel](https://github.com/damian0815/compel) library offers an intuitive syntax for scaling the prompt weights and generating the embeddings. Learn how to create the embeddings in the [Prompt weighting](../using-diffusers/weighted_prompts) guide.

Once you've generated the embeddings, pass them to the `prompt_embeds` (and `negative_prompt_embeds` if you're using a negative prompt) parameter in the [`AutoPipelineForInpainting`]. The embeddings replace the `prompt` parameter:

```py
import torch
from diffusers import AutoPipelineForInpainting

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt_emebds=prompt_embeds, # generated from Compel
    negative_prompt_embeds, # generated from Compel
    image=init_image,
    mask_image=mask_image
).images[0]
```

### ControlNet

ControlNet models are used with other diffusion models like Stable Diffusion, and they provide an even more flexible and accurate way to control how an image is generated. A ControlNet accepts an additional conditioning image input that guides the diffusion model to preserve the features in it.

For example, let's condition an image with a ControlNet pretrained on inpaint images:

```py
import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image

# load ControlNet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")

# pass ControlNet to the pipeline
pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").convert("RGB")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("RGB")

# prepare control image
def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image

control_image = make_inpaint_condition(init_image, mask_image)
```

Now generate an image from the base, mask and control images. You'll notice features of the base image are strongly preserved in the generated image.

```py
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image).images[0]
image
```

You can take this a step further and chain it with an image-to-image pipeline to apply a new [style](https://huggingface.co/nitrosocke/elden-ring-diffusion):

```py
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "nitrosocke/elden-ring-diffusion", torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

prompt = "elden ring style castle" # include the token "elden ring style" in the prompt
negative_prompt = "bad architecture, deformed, disfigured, poor details"

image = pipeline(prompt, negative_prompt=negative_prompt, image=image).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-controlnet.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">ControlNet inpaint</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-img2img.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">image-to-image</figcaption>
  </div>
</div>

## Optimize

It can be difficult and slow to run diffusion models if you're resource constrained, but it doesn't have to be with a few optimization tricks. One of the biggest (and easiest) optimizations you can enable is switching to memory-efficient attention. If you're using PyTorch 2.0, [scaled-dot product attention](../optimization/torch2.0#scaled-dot-product-attention) is automatically enabled and you don't need to do anything else. For non-PyTorch 2.0 users, you can install and use [xFormers](../optimization/xformers)'s implementation of memory-efficient attention. Both options reduce memory usage and accelerate inference.

You can also offload the model to the GPU to save even more memory:

```diff
+ pipeline.enable_xformers_memory_efficient_attention()
+ pipeline.enable_model_cpu_offload()
```

To speed-up your inference code even more, use [`torch_compile`](../optimization/torch2.0#torch.compile). You should wrap `torch.compile` around the most intensive component in the pipeline which is typically the UNet:

```py
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

Learn more in the [Reduce memory usage](../optimization/memory) and [Torch 2.0](../optimization/torch2.0) guides.