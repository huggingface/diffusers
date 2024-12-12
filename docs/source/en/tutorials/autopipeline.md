<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

Diffusers provides many pipelines for basic tasks like generating images, videos, audio, and inpainting. On top of these, there are specialized pipelines for adapters and features like upscaling, super-resolution, and more. Different pipeline classes can even use the same checkpoint because they share the same pretrained model! With so many different pipelines, it can be overwhelming to know which pipeline class to use.

The [AutoPipeline](../api/pipelines/auto_pipeline) class is designed to simplify the variety of pipelines in Diffusers. It is a generic *task-first* pipeline that lets you focus on a task ([`AutoPipelineForText2Image`], [`AutoPipelineForImage2Image`], and [`AutoPipelineForInpainting`]) without needing to know the specific pipeline class. The [AutoPipeline](../api/pipelines/auto_pipeline) automatically detects the correct pipeline class to use.

For example, let's use the [dreamlike-art/dreamlike-photoreal-2.0](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0) checkpoint.

Under the hood, [AutoPipeline](../api/pipelines/auto_pipeline):

1. Detects a `"stable-diffusion"` class from the [model_index.json](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0/blob/main/model_index.json) file.
2. Depending on the task you're interested in, it loads the [`StableDiffusionPipeline`], [`StableDiffusionImg2ImgPipeline`], or [`StableDiffusionInpaintPipeline`]. Any parameter (`strength`, `num_inference_steps`, etc.) you would pass to these specific pipelines can also be passed to the [AutoPipeline](../api/pipelines/auto_pipeline).

<hfoptions id="autopipeline">
<hfoption id="text-to-image">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

prompt = "cinematic photo of Godzilla eating sushi with a cat in a izakaya, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(37)
image = pipe_txt2img(prompt, generator=generator).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png"/>
</div>

</hfoption>
<hfoption id="image-to-image">

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png")

prompt = "cinematic photo of Godzilla eating burgers with a cat in a fast food restaurant, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(53)
image = pipe_img2img(prompt, image=init_image, generator=generator).images[0]
image
```

Notice how the [dreamlike-art/dreamlike-photoreal-2.0](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0) checkpoint is used for both text-to-image and image-to-image tasks? To save memory and avoid loading the checkpoint twice, use the [`~DiffusionPipeline.from_pipe`] method.

```py
pipe_img2img = AutoPipelineForImage2Image.from_pipe(pipe_txt2img).to("cuda")
image = pipeline(prompt, image=init_image, generator=generator).images[0]
image
```

You can learn more about the [`~DiffusionPipeline.from_pipe`] method in the [Reuse a pipeline](../using-diffusers/loading#reuse-a-pipeline) guide.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png"/>
</div>

</hfoption>
<hfoption id="inpainting">

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-mask.png")

prompt = "cinematic photo of a owl, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(38)
image = pipeline(prompt, image=init_image, mask_image=mask_image, generator=generator, strength=0.4).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-inpaint.png"/>
</div>

</hfoption>
</hfoptions>

## Unsupported checkpoints

The [AutoPipeline](../api/pipelines/auto_pipeline) supports [Stable Diffusion](../api/pipelines/stable_diffusion/overview), [Stable Diffusion XL](../api/pipelines/stable_diffusion/stable_diffusion_xl), [ControlNet](../api/pipelines/controlnet), [Kandinsky 2.1](../api/pipelines/kandinsky.md), [Kandinsky 2.2](../api/pipelines/kandinsky_v22), and [DeepFloyd IF](../api/pipelines/deepfloyd_if) checkpoints.

If you try to load an unsupported checkpoint, you'll get an error.

```py
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", torch_dtype=torch.float16, use_safetensors=True
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```
