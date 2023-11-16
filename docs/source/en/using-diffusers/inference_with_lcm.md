<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Latent Consistency Model

Latent Consistency Models (LCM) enable quality image generation in typically 2-4 steps making it possible to use diffusion models in almost real-time settings. 

From the [official website](https://latent-consistency-models.github.io/):

> LCMs can be distilled from any pre-trained Stable Diffusion (SD) in only 4,000 training steps (~32 A100 GPU Hours) for generating high quality 768 x 768 resolution images in 2~4 steps or even one step, significantly accelerating text-to-image generation. We employ LCM to distill the Dreamshaper-V7 version of SD in just 4,000 training iterations.

For a more technical overview of LCMs, refer to [the paper](https://huggingface.co/papers/2310.04378).

LCM distilled models are available for [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and the [SSD-1B](https://huggingface.co/segmind/SSD-1B) model. All the checkpoints can be found in this [collection](https://huggingface.co/collections/latent-consistency/latent-consistency-models-weights-654ce61a95edd6dffccef6a8).

This guide shows how to perform inference with LCMs for 
- text-to-image
- image-to-image
- combined with style LoRAs
- ControlNet/T2I-Adapter

## Text-to-image

You'll use the [`StableDiffusionXLPipeline`] pipeline with the [`LCMScheduler`] and then load the LCM-LoRA. Together with the LCM-LoRA and the scheduler, the pipeline enables a fast inference workflow, overcoming the slow iterative nature of diffusion models.

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
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdxl_t2i.png)

Notice that we use only 4 steps for generation which is way less than what's typically used for standard SDXL.

Some details to keep in mind:

* To perform classifier-free guidance, batch size is usually doubled inside the pipeline. LCM, however, applies guidance using guidance embeddings, so the batch size does not have to be doubled in this case. This leads to a faster inference time, with the drawback that negative prompts don't have any effect on the denoising process.
* The UNet was trained using the [3., 13.] guidance scale range. So, that is the ideal range for `guidance_scale`. However, disabling `guidance_scale` using a value of 1.0 is also effective in most cases.


## Image-to-image

LCMs can be applied to image-to-image tasks too. For this example, we'll use the [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) model, but the same steps can be applied to other LCM models as well.

```python
import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import make_image_grid, load_image

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

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
generator = torch.manual_seed(0)
image = pipe(
    prompt,
    image=init_image,
    num_inference_steps=4,
    guidance_scale=7.5,
    strength=0.5,
    generator=generator
).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdv1-5_i2i.png)


<Tip>

You can get different results based on your prompt and the image you provide. To get the best results, we recommend trying different values for `num_inference_steps`, `strength`, and `guidance_scale` parameters and choose the best one.

</Tip>


## Combine with style LoRAs

LCMs can be used with other styled LoRAs to generate styled-images in very few steps (4-8). In the following example, we'll use the [papercut LoRA](TheLastBen/Papercut_SDXL). 

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

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdx_lora_mix.png)


## ControlNet/T2I-Adapter

Let's look at how we can perform inference with ControlNet/T2I-Adapter and a LCM. 

### ControlNet
For this example, we'll use the [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) model with canny ControlNet, but the same steps can be applied to other LCM models as well.

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

# set scheduler
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

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdv1-5_controlnet.png)


<Tip>
The inference parameters in this example might not work for all examples, so we recommend trying different values for the `num_inference_steps`, `guidance_scale`, `controlnet_conditioning_scale`, and `cross_attention_kwargs` parameters and choosing the best one. 
</Tip>

### T2I-Adapter

This example shows how to use the `lcm-sdxl` with the [Canny T2I-Adapter](TencentARC/t2i-adapter-canny-sdxl-1.0).

```python
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLAdapterPipeline, UNet2DConditionModel, T2IAdapter, LCMScheduler
from diffusers.utils import load_image, make_image_grid

# Prepare image
# Detect the canny map in low resolution to avoid high-frequency details
image = load_image(
    "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_canny.jpg"
).resize((384, 384))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image).resize((1024, 1216))

# load adapter
adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")

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

prompt = "Mystical fairy in real, magic, 4k picture, high quality"
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
grid = make_image_grid([canny_image, image], rows=1, cols=2)
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdxl_t2iadapter.png)
