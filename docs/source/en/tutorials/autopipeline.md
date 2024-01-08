<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

🤗 Diffusers is able to complete many different tasks, and you can often reuse the same pretrained weights for multiple tasks such as text-to-image, image-to-image, and inpainting. If you're new to the library and diffusion models though, it may be difficult to know which pipeline to use for a task. For example, if you're using the [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) checkpoint for text-to-image, you might not know that you could also use it for image-to-image and inpainting by loading the checkpoint with the [`StableDiffusionImg2ImgPipeline`] and [`StableDiffusionInpaintPipeline`] classes respectively.

The `AutoPipeline` class is designed to simplify the variety of pipelines in 🤗 Diffusers. It is a generic, *task-first* pipeline that lets you focus on the task. The `AutoPipeline` automatically detects the correct pipeline class to use, which makes it easier to load a checkpoint for a task without knowing the specific pipeline class name.

<Tip>

Take a look at the [AutoPipeline](../api/pipelines/auto_pipeline) reference to see which tasks are supported. Currently, it supports text-to-image, image-to-image, and inpainting.

</Tip>

This tutorial shows you how to use an `AutoPipeline` to automatically infer the pipeline class to load for a specific task, given the pretrained weights.

## Choose an AutoPipeline for your task

Start by picking a checkpoint. For example, if you're interested in text-to-image with the [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) checkpoint, use [`AutoPipelineForText2Image`]:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
prompt = "peasant and dragon combat, wood cutting style, viking era, bevel with rune"

image = pipeline(prompt, num_inference_steps=25).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png" alt="generated image of peasant fighting dragon in wood cutting style"/>
</div>

Under the hood, [`AutoPipelineForText2Image`]:

1. automatically detects a `"stable-diffusion"` class from the [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json) file
2. loads the corresponding text-to-image [`StableDiffusionPipeline`] based on the `"stable-diffusion"` class name

Likewise, for image-to-image, [`AutoPipelineForImage2Image`] detects a `"stable-diffusion"` checkpoint from the `model_index.json` file and it'll load the corresponding [`StableDiffusionImg2ImgPipeline`] behind the scenes. You can also pass any additional arguments specific to the pipeline class such as `strength`, which determines the amount of noise or variation added to an input image:

```py
from diffusers import AutoPipelineForImage2Image
import torch
import requests
from PIL import Image
from io import BytesIO

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
prompt = "a portrait of a dog wearing a pearl earring"

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.thumbnail((768, 768))

image = pipeline(prompt, image, num_inference_steps=200, strength=0.75, guidance_scale=10.5).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png" alt="generated image of a vermeer portrait of a dog wearing a pearl earring"/>
</div>

And if you want to do inpainting, then [`AutoPipelineForInpainting`] loads the underlying [`StableDiffusionInpaintPipeline`] class in the same way:

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
image = pipeline(prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-inpaint.png" alt="generated image of a tiger sitting on a bench"/>
</div>

If you try to load an unsupported checkpoint, it'll throw an error:

```py
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", torch_dtype=torch.float16, use_safetensors=True
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```

## Use multiple pipelines

For some workflows or if you're loading many pipelines, it is more memory-efficient to reuse the same components from a checkpoint instead of reloading them which would unnecessarily consume additional memory. For example, if you're using a checkpoint for text-to-image and you want to use it again for image-to-image, use the [`~AutoPipelineForImage2Image.from_pipe`] method. This method creates a new pipeline from the components of a previously loaded pipeline at no additional memory cost.

The [`~AutoPipelineForImage2Image.from_pipe`] method detects the original pipeline class and maps it to the new pipeline class corresponding to the task you want to do. For example, if you load a `"stable-diffusion"` class pipeline for text-to-image:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
print(type(pipeline_text2img))
"<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'>"
```

Then [`~AutoPipelineForImage2Image.from_pipe`] maps the original `"stable-diffusion"` pipeline class to [`StableDiffusionImg2ImgPipeline`]:

```py
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(type(pipeline_img2img))
"<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline'>"
```

If you passed an optional argument - like disabling the safety checker - to the original pipeline, this argument is also passed on to the new pipeline:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    requires_safety_checker=False,
).to("cuda")

pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(pipeline_img2img.config.requires_safety_checker)
"False"
```

You can overwrite any of the arguments and even configuration from the original pipeline if you want to change the behavior of the new pipeline. For example, to turn the safety checker back on and add the `strength` argument:

```py
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img, requires_safety_checker=True, strength=0.3)
print(pipeline_img2img.config.requires_safety_checker)
"True"
```
