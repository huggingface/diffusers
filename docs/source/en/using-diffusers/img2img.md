<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Image-to-image

[[open-in-colab]]

Image-to-image is similar to [text-to-image](conditional_image_generation), but in addition to a prompt, you can also pass an initial image to use as a starting point for the diffusion process. The initial image is encoded to latent space and noise is added to it. Then the latent diffusion model takes a prompt and the noisy latent image, predicts the added noise, and then removes the predicted noise from the initial latent image to get the new latent image. Lastly, a decoder decodes the new latent image back into an image.

With 🤗 Diffusers, this is as easy as 1-2-3:

1. Load a checkpoint into the [`AutoPipelineForImage2Image`] class; this pipeline automatically handles loading the correct pipeline class to use based on the checkpoint:

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()
```

2. Prepare an image to pass to the pipeline:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
```

3. Pass a prompt and image to the pipeline to generate an image:

```py
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Kandinsky 2.2</figcaption>
  </div>
</div>

## Popular models

The most popular image-to-image models are [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Stable Diffusion XL (SDXL)](sdxl), and Kandinsky 2.2. The results from the Stable Diffusion and Kandinsky models are different because of their architecture and training process, and you can generally expect SDXL to produce higher quality images than Stable Diffusion v1.5. Let's take a quick look at how to use each of these models and compare their results.

### Stable Diffusion v1.5

Stable Diffusion v1.5 is a latent diffusion model intialized from an earlier checkpoint, and further finetuned for 595K steps on 512x512 images. To use this pipeline for image-to-image, you'll need to prepare an initial image to pass to the pipeline. Then you can pass a prompt and the image to the pipeline to generate a new image:

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdv1.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

### Stable Diffusion XL (SDXL)

SDXL is a more powerful version of the Stable Diffusion model. It uses a larger base model, and an additional refiner model to increase the quality of the generated image from the base model. Read the [SDXL](sdxl) guide for a more detailed walkthrough of how to use this model, and other techniques it uses to produce high quality images.

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src=""/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src=""/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

### Kandinsky 2.2

The Kandinsky model usage is different from the Stable Diffusion models because it uses an image prior model to create image embeddings. The embeddings help create a better alignment between text and images, allowing the latent diffusion model to generate higher quality images. For more details about how to use this model, take a look at the more comprehensive [Kandinsky]() guide.

The simplest way to use Kandinsky 2.2 is:

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-kandinsky.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

## Configure pipeline parameters

There are several important parameters you can configure in the pipeline that'll affect the image generation process and image quality. Let's take a look at some of these parameters.

### Strength

The `strength` parameter is one of the most important ones in the pipeline. It determines how much the generated image resembles the initial image. In other words:

- 📈 a higher `strength` value gives the model more "creativity" to generate an image that's different from the initial image; a `strength` value of 1.0 means the initial image is more or less ignored
- 📉 a lower `strength` value means the generated image is more similar to the initial image

The `strength` and `num_inference_steps` parameter are related because `strength` determines the number of noise steps to add. For example, if the `num_inference_steps` is 50 and `strength` is 0.8, then this means adding 40 (50 * 0.8) steps of noise to the initial image and then denoising for 40 steps to get the newly generated image.

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = init_image

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.8).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-0.4.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.4</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-0.6.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.6</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-1.0.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 1.0</figcaption>
  </div>
</div>

### Guidance scale

The `guidance_scale` parameter is used to control how closely the generated image resembles the text prompt. A higher `guidance_scale` value means your generated image is more faithful to the prompt, while a lower `guidance_scale` value means your generated image has more freedom to deviate from the prompt.

You can combine this with the `strength` parameter for even more fine-grained control on how expressive the model is. For example, combine a high `strength + guidance_scale` for maximum creativity or use a combination of low `strength` and low `guidance_scale` to generate an image that more closely resembles the initial image but is not as strictly bound to the prompt.

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, guidance_scale=8.0).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-0.1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 0.1</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-3.0.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 5.0</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-7.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 10.0</figcaption>
  </div>
</div>

### Negative prompt

A negative prompt conditions the model to *not* include things in an image, and it can be used to improve image quality or modify an image. For example, you can improve image quality by including negative prompts like "poor details" or "blurry" to encourage the model to generate a higher quality image. Or you can modify an image by specifying things to exclude from an image.

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

# pass prompt and image to pipeline
image = pipeline(prompt, negative_prompt=negative_prompt, image=init_image).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-negative-1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">negative prompt = "ugly, deformed, disfigured, poor details, bad anatomy"</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-negative-2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">negative prompt = "jungle"</figcaption>
  </div>
</div>

## Chained image-to-image pipelines

There are some interesting ways you can use an image-to-image pipeline aside from just generating an image (although that is pretty cool too). You can take it a step further and chain it after a text-to-image or another image-to-image pipeline. The key is to keep all the outputs in *latent* space to avoid an unnecessary decode-encode step. You can do this by specifying `output_type="latent"` in a pipeline.

### Text-to-image-to-image

Chaining a text-to-image and image-to-image pipeline allows you to generate an image from text, and then use that image as the initial image for the image-to-image pipeline. This is useful if you want to generate an image entirely from scratch. For example, let's chain a Stable Diffusion model and a Kandinsky model.

Start by generating an image with the text-to-image pipeline:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", output_type="latent").images
```

Now you can pass this generated image to the image-to-image pipeline:

```py
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=image).images[0]
image
```

### Image-to-image-to-image

You can also chain multiple image-to-image pipelines together to create more interesting images. For example, you can load checkpoints finetuned on a specific style in each pipeline to create an image that is a combination of several styles.

Start by generating an image:

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, output_type="latent").images[0]
```

Pass the latent output from this pipeline to the next pipeline:

```py
pipelne = AutoPipelineForImage2Image.from_pretrained(
    "nerijs/pixel-art-xl", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt, image=image, output_type="latent").images[0]
```

Repeat one more time to generate the final image:

```py
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "ogkalu/Comic-Diffusion", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt, image=image).images[0]
image
```

### Image-to-upscaler-to-super-resolution

Another way you can chain your image-to-image pipeline is with an upscaler and super-resolution pipeline to really increase the level of details in an image.

Start with an image-to-image pipeline:

```py
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image_1 = pipeline(prompt, image=init_image, output_type="latent").images[0]
```

Chain it to an upscaler pipeline to increase the image resolution:

```py
upscaler = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
upscaler.enable_model_cpu_offload()
upscaler.enable_xformers_memory_efficient_attention()

image_2 = upscaler(prompt, image=image_1, output_type="latent").images[0]
```

Finally, chain it to a super-resolution pipeline to further enhance the resolution:

```py
super_res = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
super_res.enable_model_cpu_offload()
super_res.enable_xformers_memory_efficient_attention()

image_3 = upscaler(prompt, image=image_2).images[0]
image_3
```

## Control image generation

Trying to generate an image that looks exactly the way you want can be difficult which is why controlled generation techniques and models are so useful. While you can use the `negative_prompt` to partially control image generation, there are more robust methods like prompt weighting and ControlNets.

### Prompt weighting

Prompt weighting allows you to scale the representation of each concept in a prompt. For example, in a prompt like "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", you can choose to increase or decrease the embeddings of "astronaut" and "jungle". The [Compel](https://github.com/damian0815/compel) library provides a simple syntax for adjusting prompt weights and generating the embeddings. You can learn how to create the embeddings in the [Prompt weighting](weighted_prompts) guide.

The [`AutoPipelineForImage2Image`] has a `prompt_embeds` (and `negative_prompt_embeds` if you're using a negative prompt) parameter where you can pass the embeddings which replaces the `prompt` parameter.

```py
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt_emebds=prompt_embeds, # generated from Compel
    negative_prompt_embeds, # generated from Compel
    image=init_image,
).images[0]
```

### ControlNet

ControlNets provide a more flexible and accurate way to control image generation because you can use an additional conditioning image. The conditioning image can be a canny image, depth map, image segmentation, and even scribbles! Whatever type of conditioning image you choose, the ControlNet generates an image that preserves the information in it.

For example, let's condition an image with a depth map to keep the spatial information in the image.

```py
# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((958, 960)) # resize to depth image dimensions
depth_image = load_image("https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png")
```

Load the ControlNet model conditioned on depth maps and the [`AutoPipelineForImage2Image`]:

```py
from diffusers import ControlNetModel, AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()
```

Generate a new image conditioned on the depth map, initial image, and prompt:

```py
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt, image=init_image, control_image=depth_image).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">depth image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-controlnet.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">ControlNet image</figcaption>
  </div>
</div>

## Optimize