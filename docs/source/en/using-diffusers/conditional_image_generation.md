<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text-to-image

[[open-in-colab]]

When you think of diffusion models, text-to-image is usually one of the first things that come to mind. Text-to-image generates an image from a text description (for example, "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k") which is also known as a *prompt*.

From a very high level, a diffusion model takes a prompt and some random initial noise, and iteratively removes the noise to construct an image. The *denoising* process is guided by the prompt, and once the denoising process ends after a predetermined number of time steps, the image representation is decoded into an image.

<Tip>

Read the [How does Stable Diffusion work?](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work) blog post to learn more about how a latent diffusion model works.

</Tip>

You can generate images from a prompt in 🤗 Diffusers in two steps:

1. Load a checkpoint into the [`AutoPipelineForText2Image`] class, which automatically detects the appropriate pipeline class to use based on the checkpoint:

```py
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
```

2. Pass a prompt to the pipeline to generate an image:

```py
image = pipeline(
	"stained glass of darth vader, backlight, centered composition, masterpiece, photorealistic, 8k"
).images[0]
```

<div class="flex justify-center">
	<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-vader.png"/>
</div>

## Popular models

The most common text-to-image models are [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder). There are also ControlNet models or adapters that can be used with text-to-image models for more direct control in generating images. The results from each model are slightly different because of their architecture and training process, but no matter which model you choose, their usage is more or less the same. Let's use the same prompt for each model and compare their results.

### Stable Diffusion v1.5

[Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) is a latent diffusion model initialized from [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), and finetuned for 595K steps on 512x512 images from the LAION-Aesthetics V2 dataset. You can use this model like:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", generator=generator).images[0]
```

### Stable Diffusion XL

SDXL is a much larger version of the previous Stable Diffusion models, and involves a two-stage model process that adds even more details to an image. It also includes some additional *micro-conditionings* to generate high-quality images centered subjects. Take a look at the more comprehensive [SDXL](sdxl) guide to learn more about how to use it. In general, you can use SDXL like:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", generator=generator).images[0]
```

### Kandinsky 2.2

The Kandinsky model is a bit different from the Stable Diffusion models because it also uses an image prior model to create embeddings that are used to better align text and images in the diffusion model.

The easiest way to use Kandinsky 2.2 is:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", generator=generator).images[0]
```

### ControlNet

ControlNet are auxiliary models or adapters that are finetuned on top of text-to-image models, such as [Stable Diffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). Using ControlNet models in combination with text-to-image models offers diverse options for more explicit control over how to generate an image. With ControlNet's, you add an additional conditioning input image to the model. For example, if you provide an image of a human pose (usually represented as multiple keypoints that are connected into a skeleton) as a conditioning input, the model generates an image that follows the pose of the image. Check out the more in-depth [ControlNet](controlnet) guide to learn more about other conditioning inputs and how to use them.

In this example, let's condition the ControlNet with a human pose estimation image. Load the ControlNet model pretrained on human pose estimations:

```py
from diffusers import ControlNetModel, AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained(
	"lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pose_image = load_image("https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/control.png")
```

Pass the `controlnet` to the [`AutoPipelineForText2Image`], and provide the prompt and pose estimation image:

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=pose_image, generator=generator).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion v1.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion XL</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Kandinsky 2.2</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-3.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">ControlNet (pose conditioning)</figcaption>
  </div>
</div>

## Configure pipeline parameters

There are a number of parameters that can be configured in the pipeline that affect how an image is generated. You can change the image's output size, specify a negative prompt to improve image quality, and more. This section dives deeper into how to use these parameters.

### Height and width

The `height` and `width` parameters control the height and width (in pixels) of the generated image. By default, the Stable Diffusion v1.5 model outputs 512x512 images, but you can change this to any size that is a multiple of 8. For example, to create a rectangular image:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
image = pipeline(
	"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", height=768, width=512
).images[0]
```

<div class="flex justify-center">
	<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-hw.png"/>
</div>

<Tip warning={true}>

Other models may have different default image sizes depending on the image size's in the training dataset. For example, SDXL's default image size is 1024x1024 and using lower `height` and `width` values may result in lower quality images. Make sure you check the model's API reference first!

</Tip>

### Guidance scale

The `guidance_scale` parameter affects how much the prompt influences image generation. A lower value gives the model "creativity" to generate images that are more loosely related to the prompt. Higher `guidance_scale` values push the model to follow the prompt more closely, and if this value is too high, you may observe some artifacts in the generated image.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
image = pipeline(
	"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", guidance_scale=3.5
).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-guidance-scale-2.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 2.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-guidance-scale-7.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 7.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-guidance-scale-10.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 10.5</figcaption>
  </div>
</div>

### Negative prompt

Just like how a prompt guides generation, a *negative prompt* steers the model away from things you don't want the model to generate. This is commonly used to improve overall image quality by removing poor or bad image features such as "low resolution" or "bad details". You can also use a negative prompt to remove or modify the content and style of an image.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
image = pipeline(
	prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", 
	negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-neg-prompt-1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">negative prompt = "ugly, deformed, disfigured, poor details, bad anatomy"</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/text2img-neg-prompt-2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">negative prompt = "astronaut"</figcaption>
  </div>
</div>

### Generator

A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#generator) object enables reproducibility in a pipeline by setting a manual seed. You can use a `Generator` to generate batches of images and iteratively improve on an image generated from a seed as detailed in the [Improve image quality with deterministic generation](reusing_seeds) guide.

You can set a seed and `Generator` as shown below. Creating an image with a `Generator` should return the same result each time instead of randomly generating a new image.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(30)
image = pipeline(
	"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", 
	generator=generator,
).images[0]
```

## Control image generation

There are several ways to exert more control over how an image is generated outside of configuring a pipeline's parameters, such as prompt weighting and ControlNet models.

### Prompt weighting

Prompt weighting is a technique for increasing or decreasing the importance of concepts in a prompt to emphasize or minimize certain features in an image. We recommend using the [Compel](https://github.com/damian0815/compel) library to help you generate the weighted prompt embeddings.

<Tip>

Learn how to create the prompt embeddings in the [Prompt weighting](weighted_prompts) guide. This example focuses on how to use the prompt embeddings in the pipeline.

</Tip>

Once you've created the embeddings, you can pass them to the `prompt_embeds` (and `negative_prompt_embeds` if you're using a negative prompt) parameter in the pipeline.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
image = pipeline(
	prompt_emebds=prompt_embeds, # generated from Compel
	negative_prompt_embeds=negative_prompt_embeds, # generated from Compel
).images[0]
```

### ControlNet

As you saw in the [ControlNet](#controlnet) section, these models offer a more flexible and accurate way to generate images by incorporating an additional conditioning image input. Each ControlNet model is pretrained on a particular type of conditioning image to generate new images that resemble it. For example, if you take a ControlNet pretrained on depth maps, you can give the model a depth map as a conditioning input and it'll generate an image that preserves the spatial information in it. This is quicker and easier than specifying the depth information in a prompt. You can even combine multiple conditioning inputs with a [MultiControlNet](controlnet#multicontrolnet)!

There are many types of conditioning inputs you can use, and 🤗 Diffusers supports ControlNet for Stable Diffusion and SDXL models. Take a look at the more comprehensive [ControlNet](controlnet) guide to learn how you can use these models.

## Optimize

Diffusion models are large, and the iterative nature of denoising an image is computationally expensive and intensive. But this doesn't mean you need access to powerful - or even many - GPUs to use them. There are many optimization techniques for running diffusion models on consumer and free-tier resources. For example, you can load model weights in half-precision to save GPU memory and increase speed or offload the entire model to the GPU to save even more memory.

PyTorch 2.0 also supports a more memory-efficient attention mechanism called [*scaled dot product attention*](../optimization/torch2.0#scaled-dot-product-attention) that is automatically enabled if you're using PyTorch 2.0. You can combine this with [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) to speed your code up even more:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overheard", fullgraph=True)
```

For more tips on how to optimize your code to save memory and speed up inference, read the [Memory and speed](../optimization/fp16) and [Torch 2.0](../optimization/torch2.0) guides.