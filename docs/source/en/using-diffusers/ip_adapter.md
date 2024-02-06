<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IP-Adapter

[IP-Adapter](https://hf.co/papers/2308.06721) is an image prompt adapter that can be plugged into diffusion models to enable image prompting without any changes to the underlying model. Furthermore, this adapter can be reused with other models finetuned from the same base model and it can be combined with other adapters like ControlNet. The key idea behind IP-Adapter is the *decoupled cross-attention* mechanism which adds a separate cross-attention layer just for image features instead of using the same cross-attention layer for both text and image features. This allows the model to learn more image-specific features.

> [!TIP]
> Learn how to load an IP-Adapter in the [Load adapters](../using-diffusers/loading_adapters#ip-adapter) guide.

This guide will walk you through using IP-Adapter for various tasks and use cases.

## General tasks

Let's take a look at how to use IP-Adapter's image prompting capabilities with the [`StableDiffusionXLPipeline`] for tasks like text-to-image, image-to-image, and inpainting. Feel free to use another pipeline such as Stable Diffusion, LCM-LoRA, ControlNet, T2I-Adapter, or AnimateDiff!

<hfoptions id="tasks">
<hfoption id="Text-to-image">

Crafting the precise text prompt to generate the image you want can be difficult because it may not always capture what you'd like to express. Adding an image alongside the text prompt helps the model better understand what it should generate and can lead to more accurate results.

Load a Stable Diffusion XL (SDXL) model and insert an IP-Adapter into the model with the [`~IPAdapterMixin.load_ip_adapter`] method. Use the `subfolder` parameter to load the weights for SDXL.

The [`~IPAdapterMixin.set_ip_adapter_scale`] method controls the amount of text or image conditioning to apply to the model. A value of `1.0` means the model is only conditioned on the image prompt. Lowering this value encourages the model to produce more diverse images, but they may not be as aligned with the image prompt. Typically, a value of `0.5` achieves a good balance between the two prompt types and produces good results.

```py
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)
```

Create a text prompt and load an image prompt before passing them to the pipeline to generate an image.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png")
generator = torch.Generator(device="cpu").manual_seed(0)
images = pipeline(
    prompt="a polar bear sitting in a chair drinking a milkshake", 
    ip_adapter_image=image,
    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    num_inference_steps=100, 
    generator=generator,
).images
images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner_2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="Image-to-image">

IP-Adapter can also help with image-to-image by guiding the model to generate an image that resembles the original image and the image prompt.

Load a Stable Diffusion XL (SDXL) model and insert an IP-Adapter into the model with the [`~IPAdapterMixin.load_ip_adapter`] method. Use the `subfolder` parameter to load the weights for SDXL.

The [`~IPAdapterMixin.set_ip_adapter_scale`] method controls the amount of text or image conditioning to apply to the model. A value of `1.0` means the model is only conditioned on the image prompt. Lowering this value encourages the model to produce more diverse images, but they may not be as aligned with the image prompt. Typically, a value of `0.5` achieves a good balance between the two prompt types and produces good results.

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)
```

Pass the original image and the IP-Adapter image prompt to the pipeline to generate an image. Providing a text prompt to the pipeline is optional, but in this example, a text prompt is used to increase image quality.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_2.png")

generator = torch.Generator(device="cpu").manual_seed(4)
images = pipeline(
    prompt="best quality, high quality",
    image=image,
    ip_adapter_image=ip_image,
    generator=generator,
    strength=0.6,
).images
images[0]
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_3.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="Inpainting">

IP-Adapter is also useful for inpainting because the image prompt allows you to be much more specific about what you'd like to generate.

Load a Stable Diffusion XL (SDXL) model and insert an IP-Adapter into the model with the [`~IPAdapterMixin.load_ip_adapter`] method. Use the `subfolder` parameter to load the weights for SDXL.

The [`~IPAdapterMixin.set_ip_adapter_scale`] method controls the amount of text or image conditioning to apply to the model. A value of `1.0` means the model is only conditioned on the image prompt. Lowering this value encourages the model to produce more diverse images, but they may not be as aligned with the image prompt. Typically, a value of `0.5` achieves a good balance between the two prompt types and produces good results.

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)
```

Pass a prompt, the original image, mask image, and the IP-Adapter image prompt to the pipeline to generate an image. Providing a text prompt to the pipeline is optional, but in this example, a text prompt is used to increase image quality.

```py
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_mask.png")
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png")

generator = torch.Generator(device="cpu").manual_seed(4)
images = pipeline(
    prompt="a cute gummy bear waving",
    image=image,
    mask_image=mask_image,
    ip_adapter_image=ip_image,
    generator=generator,
    num_inference_steps=100,
).images
images[0]
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="Video">

IP-Adapter can also help you generate videos that are more aligned with your text prompt. For example, let's load [AnimateDiff](../api/pipelines/animatediff) with it's motion adapter, and insert an IP-Adapter into the model with the [`~IPAdapterMixin.load_ip_adapter`] method.

The [`~IPAdapterMixin.set_ip_adapter_scale`] method controls the amount of text or image conditioning to apply to the model. A value of `1.0` means the model is only conditioned on the image prompt. Lowering this value encourages the model to produce more diverse images, but they may not be as aligned with the image prompt. Typically, a value of `0.5` achieves a good balance between the two prompt types and produces good results.

> [!WARNING]
> If you're planning on offloading the model to the CPU, make sure you run it after you've loaded the IP-Adapter. When you call [`~DiffusionPipeline.enable_model_cpu_offload`] before loading the IP-Adapter, it offloads the image encoder module to the CPU and it'll return an error when you try to run the pipeline.

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from diffusers.utils import load_image

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    "emilianJR/epiCRealism",
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.enable_vae_slicing()

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipeline.set_ip_adapter_scale(0.6)
pipeline.enable_model_cpu_offload()
```

Pass a prompt and an image prompt to the pipeline to generate a short video.

```py
ip_adapter_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png")

output = pipeline(
    prompt="A cute gummy bear waving",
    negative_prompt="bad quality, worse quality, low resolution",
    ip_adapter_image=ip_adapter_image,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=50,
    generator=torch.Generator(device="cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animation1.gif")
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src=""/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated video</figcaption>
  </div>
</div>

</hfoption>
</hfoptions>

## Specific use cases

IP-Adapters image prompting and compatibility with other adapters and models makes it a versatile tool for a variety of use cases. Let's have a look at some of them.

### Face model  

### Multi IP-Adapter

### Instant generation

### Structural control
