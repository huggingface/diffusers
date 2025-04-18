<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Diffusion XL Turbo

[[open-in-colab]]

SDXL Turbo is an adversarial time-distilled [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) (SDXL) model capable
of running inference in as little as 1 step.

This guide will show you how to use SDXL-Turbo for text-to-image and image-to-image.

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries in Colab
#!pip install -q diffusers transformers accelerate
```

## Load model checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [`~StableDiffusionXLPipeline.from_pretrained`] method:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to("cuda")
```

You can also use the [`~StableDiffusionXLPipeline.from_single_file`] method to load a model checkpoint stored in a single file format (`.ckpt` or `.safetensors`) from the Hub or locally. For this loading method, you need to set `timestep_spacing="trailing"` (feel free to experiment with the other scheduler config values to get better results):

```py
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors",
    torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to("cuda")
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
```

## Text-to-image

For text-to-image, pass a text prompt. By default, SDXL Turbo generates a 512x512 image, and that resolution gives the best results. You can try setting the `height` and `width` parameters to 768x768 or 1024x1024, but you should expect quality degradations when doing so.

Make sure to set `guidance_scale` to 0.0 to disable, as the model was trained without it. A single inference step is enough to generate high quality images.
Increasing the number of steps to 2, 3 or 4 should improve image quality.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline_text2image = pipeline_text2image.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sdxl-turbo-text2img.png" alt="generated image of a racoon in a robe"/>
</div>

## Image-to-image

For image-to-image generation, make sure that `num_inference_steps * strength` is larger or equal to 1.
The image-to-image pipeline will run for `int(num_inference_steps * strength)` steps, e.g. `0.5 * 2.0 = 1` step in
our example below.

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline_image2image = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
init_image = init_image.resize((512, 512))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipeline_image2image(prompt, image=init_image, strength=0.5, guidance_scale=0.0, num_inference_steps=2).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sdxl-turbo-img2img.png" alt="Image-to-image generation sample using SDXL Turbo"/>
</div>

## Speed-up SDXL Turbo even more

- Compile the UNet if you are using PyTorch version 2.0 or higher. The first inference run will be very slow, but subsequent ones will be much faster.

```py
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

- When using the default VAE, keep it in `float32` to avoid costly `dtype` conversions before and after each generation. You only need to do this one before your first generation:

```py
pipe.upcast_vae()
```

As an alternative, you can also use a [16-bit VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) created by community member [`@madebyollin`](https://huggingface.co/madebyollin) that does not need to be upcasted to `float32`.
