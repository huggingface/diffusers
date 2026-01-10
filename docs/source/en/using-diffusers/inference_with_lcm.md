<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Latent Consistency Model

[[open-in-colab]]

[Latent Consistency Models (LCMs)](https://hf.co/papers/2310.04378) enable fast high-quality image generation by directly predicting the reverse diffusion process in the latent rather than pixel space. In other words, LCMs try to predict the noiseless image from the noisy image in contrast to typical diffusion models that iteratively remove noise from the noisy image. By avoiding the iterative sampling process, LCMs are able to generate high-quality images in 2-4 steps instead of 20-30 steps.

LCMs are distilled from pretrained models which requires ~32 hours of A100 compute. To speed this up, [LCM-LoRAs](https://hf.co/papers/2311.05556) train a [LoRA adapter](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) which have much fewer parameters to train compared to the full model. The LCM-LoRA can be plugged into a diffusion model once it has been trained.

This guide will show you how to use LCMs and LCM-LoRAs for fast inference on tasks and how to use them with other adapters like ControlNet or T2I-Adapter.

> [!TIP]
> LCMs and LCM-LoRAs are available for Stable Diffusion v1.5, Stable Diffusion XL, and the SSD-1B model. You can find their checkpoints on the [Latent Consistency](https://hf.co/collections/latent-consistency/latent-consistency-models-weights-654ce61a95edd6dffccef6a8) Collections.

## Text-to-image

<hfoptions id="lcm-text2img">
<hfoption id="LCM">

To use LCMs, you need to load the LCM checkpoint for your supported model into [`UNet2DConditionModel`] and replace the scheduler with the [`LCMScheduler`]. Then you can use the pipeline as usual, and pass a text prompt to generate an image in just 4 steps.

A couple of notes to keep in mind when using LCMs are:

* Typically, batch size is doubled inside the pipeline for classifier-free guidance. But LCM applies guidance with guidance embeddings and doesn't need to double the batch size, which leads to faster inference. The downside is that negative prompts don't work with LCM because they don't have any effect on the denoising process.
* The ideal range for `guidance_scale` is [3., 13.] because that is what the UNet was trained with. However, disabling `guidance_scale` with a value of 1.0 is also effective in most cases.

```python
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdxl_t2i.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

To use LCM-LoRAs, you need to replace the scheduler with the [`LCMScheduler`] and load the LCM-LoRA weights with the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method. Then you can use the pipeline as usual, and pass a text prompt to generate an image in just 4 steps.

A couple of notes to keep in mind when using LCM-LoRAs are:

* Typically, batch size is doubled inside the pipeline for classifier-free guidance. But LCM applies guidance with guidance embeddings and doesn't need to double the batch size, which leads to faster inference. The downside is that negative prompts don't work with LCM because they don't have any effect on the denoising process.
* You could use guidance with LCM-LoRAs, but it is very sensitive to high `guidance_scale` values and can lead to artifacts in the generated image. The best values we've found are between [1.0, 2.0].
* Replace [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0) with any finetuned model. For example, try using the [animagine-xl](https://huggingface.co/Linaqruf/animagine-xl) checkpoint to generate anime images with SDXL.

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(42)
image = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdxl_t2i.png"/>
</div>

</hfoption>
</hfoptions>

## Image-to-image

<hfoptions id="lcm-img2img">
<hfoption id="LCM">

To use LCMs for image-to-image, you need to load the LCM checkpoint for your supported model into [`UNet2DConditionModel`] and replace the scheduler with the [`LCMScheduler`]. Then you can use the pipeline as usual, and pass a text prompt and initial image to generate an image in just 4 steps.

> [!TIP]
> Experiment with different values for `num_inference_steps`, `strength`, and `guidance_scale` to get the best results.

```python
import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image

unet = UNet2DConditionModel.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    subfolder="unet",
    torch_dtype=torch.float16,
)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "Lykon/dreamshaper-7",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
generator = torch.manual_seed(0)
image = pipe(
    prompt,
    image=init_image,
    num_inference_steps=4,
    guidance_scale=7.5,
    strength=0.5,
    generator=generator
).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-img2img.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

To use LCM-LoRAs for image-to-image, you need to replace the scheduler with the [`LCMScheduler`] and load the LCM-LoRA weights with the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method. Then you can use the pipeline as usual, and pass a text prompt and initial image to generate an image in just 4 steps.

> [!TIP]
> Experiment with different values for `num_inference_steps`, `strength`, and `guidance_scale` to get the best results.

```py
import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from diffusers.utils import make_image_grid, load_image

pipe = AutoPipelineForImage2Image.from_pretrained(
    "Lykon/dreamshaper-7",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

generator = torch.manual_seed(0)
image = pipe(
    prompt,
    image=init_image,
    num_inference_steps=4,
    guidance_scale=1,
    strength=0.6,
    generator=generator
).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-img2img.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

</hfoption>
</hfoptions>

## Inpainting

To use LCM-LoRAs for inpainting, you need to replace the scheduler with the [`LCMScheduler`] and load the LCM-LoRA weights with the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method. Then you can use the pipeline as usual, and pass a text prompt, initial image, and mask image to generate an image in just 4 steps.

```py
import torch
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers.utils import load_image, make_image_grid

pipe = AutoPipelineForInpainting.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    generator=generator,
    num_inference_steps=4,
    guidance_scale=4,
).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

## Adapters

LCMs are compatible with adapters like LoRA, ControlNet, T2I-Adapter, and AnimateDiff. You can bring the speed of LCMs to these adapters to generate images in a certain style or condition the model on another input like a canny image.

### LoRA

[LoRA](../tutorials/using_peft_for_inference) adapters can be rapidly finetuned to learn a new style from just a few images and plugged into a pretrained model to generate images in that style.

<hfoptions id="lcm-lora">
<hfoption id="LCM">

Load the LCM checkpoint for your supported model into [`UNet2DConditionModel`] and replace the scheduler with the [`LCMScheduler`]. Then you can use the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method to load the LoRA weights into the LCM and generate a styled image in a few steps.

```python
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

prompt = "papercut, a cute fox"
generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdx_lora_mix.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

Replace the scheduler with the [`LCMScheduler`]. Then you can use the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method to load the LCM-LoRA weights and the style LoRA you want to use. Combine both LoRA adapters with the [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] method and generate a styled image in a few steps.

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])

prompt = "papercut, a cute fox"
generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=4, guidance_scale=1, generator=generator).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdx_lora_mix.png"/>
</div>

</hfoption>
</hfoptions>

### ControlNet

[ControlNet](./controlnet) are adapters that can be trained on a variety of inputs like canny edge, pose estimation, or depth. The ControlNet can be inserted into the pipeline to provide additional conditioning and control to the model for more accurate generation.

You can find additional ControlNet models trained on other inputs in [lllyasviel's](https://hf.co/lllyasviel) repository.

<hfoptions id="lcm-controlnet">
<hfoption id="LCM">

Load a ControlNet model trained on canny images and pass it to the [`ControlNetModel`]. Then you can load a LCM model into [`StableDiffusionControlNetPipeline`] and replace the scheduler with the [`LCMScheduler`]. Now pass the canny image to the pipeline and generate an image.

> [!TIP]
> Experiment with different values for `num_inference_steps`, `controlnet_conditioning_scale`, `cross_attention_kwargs`, and `guidance_scale` to get the best results.

```python
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image, make_image_grid

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((512, 512))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)
image = pipe(
    "the mona lisa",
    image=canny_image,
    num_inference_steps=4,
    generator=generator,
).images[0]
make_image_grid([canny_image, image], rows=1, cols=2)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdv1-5_controlnet.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

Load a ControlNet model trained on canny images and pass it to the [`ControlNetModel`]. Then you can load a Stable Diffusion v1.5 model into [`StableDiffusionControlNetPipeline`] and replace the scheduler with the [`LCMScheduler`]. Use the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method to load the LCM-LoRA weights, and pass the canny image to the pipeline and generate an image.

> [!TIP]
> Experiment with different values for `num_inference_steps`, `controlnet_conditioning_scale`, `cross_attention_kwargs`, and `guidance_scale` to get the best results.

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((512, 512))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    variant="fp16"
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

generator = torch.manual_seed(0)
image = pipe(
    "the mona lisa",
    image=canny_image,
    num_inference_steps=4,
    guidance_scale=1.5,
    controlnet_conditioning_scale=0.8,
    cross_attention_kwargs={"scale": 1},
    generator=generator,
).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdv1-5_controlnet.png"/>
</div>

</hfoption>
</hfoptions>

### T2I-Adapter

[T2I-Adapter](./t2i_adapter) is an even more lightweight adapter than ControlNet, that provides an additional input to condition a pretrained model with. It is faster than ControlNet but the results may be slightly worse.

You can find additional T2I-Adapter checkpoints trained on other inputs in [TencentArc's](https://hf.co/TencentARC) repository.

<hfoptions id="lcm-t2i">
<hfoption id="LCM">

Load a T2IAdapter trained on canny images and pass it to the [`StableDiffusionXLAdapterPipeline`]. Then load a LCM checkpoint into [`UNet2DConditionModel`] and replace the scheduler with the [`LCMScheduler`]. Now pass the canny image to the pipeline and generate an image.

```python
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLAdapterPipeline, UNet2DConditionModel, T2IAdapter, LCMScheduler
from diffusers.utils import load_image, make_image_grid

# detect the canny map in low resolution to avoid high-frequency details
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((384, 384))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image).resize((1024, 1216))

adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    adapter=adapter,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "the mona lisa, 4k picture, high quality"
negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    num_inference_steps=4,
    guidance_scale=5,
    adapter_conditioning_scale=0.8,
    adapter_conditioning_factor=1,
    generator=generator,
).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-t2i.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

Load a T2IAdapter trained on canny images and pass it to the [`StableDiffusionXLAdapterPipeline`]. Replace the scheduler with the [`LCMScheduler`], and use the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method to load the LCM-LoRA weights. Pass the canny image to the pipeline and generate an image.

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLAdapterPipeline, UNet2DConditionModel, T2IAdapter, LCMScheduler
from diffusers.utils import load_image, make_image_grid

# detect the canny map in low resolution to avoid high-frequency details
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((384, 384))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image).resize((1024, 1024))

adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    adapter=adapter,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt = "the mona lisa, 4k picture, high quality"
negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    num_inference_steps=4,
    guidance_scale=1.5,
    adapter_conditioning_scale=0.8,
    adapter_conditioning_factor=1,
    generator=generator,
).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-t2i.png"/>
</div>

</hfoption>
</hfoptions>

### AnimateDiff

[AnimateDiff](../api/pipelines/animatediff) is an adapter that adds motion to an image. It can be used with most Stable Diffusion models, effectively turning them into "video generation" models. Generating good results with a video model usually requires generating multiple frames (16-24), which can be very slow with a regular Stable Diffusion model. LCM-LoRA can speed up this process by only taking 4-8 steps for each frame.

Load a [`AnimateDiffPipeline`] and pass a [`MotionAdapter`] to it. Then replace the scheduler with the [`LCMScheduler`], and combine both LoRA adapters with the [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] method. Now you can pass a prompt to the pipeline and generate an animated image.

```py
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler, LCMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")
pipe = AnimateDiffPipeline.from_pretrained(
    "frankjoshua/toonyou_beta6",
    motion_adapter=adapter,
).to("cuda")

# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# load LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", weight_name="diffusion_pytorch_model.safetensors", adapter_name="motion-lora")

pipe.set_adapters(["lcm", "motion-lora"], adapter_weights=[0.55, 1.2])

prompt = "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress"
generator = torch.manual_seed(0)
frames = pipe(
    prompt=prompt,
    num_inference_steps=5,
    guidance_scale=1.25,
    cross_attention_kwargs={"scale": 1},
    num_frames=24,
    generator=generator
).frames[0]
export_to_gif(frames, "animation.gif")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-animatediff.gif"/>
</div>
