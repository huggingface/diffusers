<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Diffusion pipelines

Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). Latent diffusion applies the diffusion process over a lower dimensional latent space to reduce memory and compute complexity. This specific type of diffusion model was proposed in [High-Resolution Image Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.

Stable Diffusion is trained on 512x512 images from a subset of the LAION-5B dataset. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and can run on consumer GPUs.

For more details about how Stable Diffusion works and how it differs from the base latent diffusion model, take a look at the Stability AI [announcement](https://stability.ai/blog/stable-diffusion-announcement) and our own [blog post](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work) for more technical details.

You can find the original codebase for Stable Diffusion v1.0 at [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) and Stable Diffusion v2.0 at [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion) as well as their original scripts for various tasks. Additional official checkpoints for the different Stable Diffusion versions and tasks can be found on the [CompVis](https://huggingface.co/CompVis), [Runway](https://huggingface.co/runwayml), and [Stability AI](https://huggingface.co/stabilityai) Hub organizations. Explore these organizations to find the best checkpoint for your use-case!

The table below summarizes the available Stable Diffusion pipelines, their supported tasks, and an interactive demo:

<div class="flex justify-center">
    <div class="rounded-xl border border-gray-200">
    <table class="min-w-full divide-y-2 divide-gray-200 bg-white text-sm">
        <thead>
        <tr>
            <th class="px-4 py-2 font-medium text-gray-900 text-left">
            Pipeline
            </th>
            <th class="px-4 py-2 font-medium text-gray-900 text-left">
            Supported tasks
            </th>
            <th class="px-4 py-2 font-medium text-gray-900 text-left">
            🤗 Space
            </th>
        </tr>
        </thead>
        <tbody class="divide-y divide-gray-200">
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./text2img">StableDiffusion</a>
            </td>
            <td class="px-4 py-2 text-gray-700">text-to-image</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./img2img">StableDiffusionImg2Img</a>
            </td>
            <td class="px-4 py-2 text-gray-700">image-to-image</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/huggingface/diffuse-the-rest"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./inpaint">StableDiffusionInpaint</a>
            </td>
            <td class="px-4 py-2 text-gray-700">inpainting</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./depth2img">StableDiffusionDepth2Img</a>
            </td>
            <td class="px-4 py-2 text-gray-700">depth-to-image</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/radames/stable-diffusion-depth2img"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./image_variation">StableDiffusionImageVariation</a>
            </td>
            <td class="px-4 py-2 text-gray-700">image variation</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./stable_diffusion_safe">StableDiffusionPipelineSafe</a>
            </td>
            <td class="px-4 py-2 text-gray-700">filtered text-to-image</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/AIML-TUDA/unsafe-vs-safe-stable-diffusion"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./stable_diffusion_2">StableDiffusion2</a>
            </td>
            <td class="px-4 py-2 text-gray-700">text-to-image, inpainting, depth-to-image, super-resolution</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./stable_diffusion_xl">StableDiffusionXL</a>
            </td>
            <td class="px-4 py-2 text-gray-700">text-to-image, image-to-image</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/RamAnanth1/stable-diffusion-xl"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./latent_upscale">StableDiffusionLatentUpscale</a>
            </td>
            <td class="px-4 py-2 text-gray-700">super-resolution</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/huggingface-projects/stable-diffusion-latent-upscaler"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./upscale">StableDiffusionUpscale</a>
            </td>
            <td class="px-4 py-2 text-gray-700">super-resolution</td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./ldm3d_diffusion">StableDiffusionLDM3D</a>
            </td>
            <td class="px-4 py-2 text-gray-700">text-to-rgb, text-to-depth, text-to-pano</td>
            <td class="px-4 py-2"><a href="https://huggingface.co/spaces/r23/ldm3d-space"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
            </td>
        </tr>
        <tr>
            <td class="px-4 py-2 text-gray-700">
            <a href="./ldm3d_diffusion">StableDiffusionUpscaleLDM3D</a>
            </td>
            <td class="px-4 py-2 text-gray-700">ldm3d super-resolution</td>
        </tr>
        </tbody>
    </table>
    </div>
</div>

## Tips

To help you get the most out of the Stable Diffusion pipelines, here are a few tips for improving performance and usability. These tips are applicable to all Stable Diffusion pipelines.

### Explore tradeoff between speed and quality

[`StableDiffusionPipeline`] uses the [`PNDMScheduler`] by default, but 🤗 Diffusers provides many other schedulers (some of which are faster or output better quality) that are compatible. For example, if you want to use the [`EulerDiscreteScheduler`] instead of the default:

```py
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# or
euler_scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=euler_scheduler)
```

### Reuse pipeline components to save memory

To save memory and use the same components across multiple pipelines, use the `.components` method to avoid loading weights into RAM more than once.

```py
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)

text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
inpaint = StableDiffusionInpaintPipeline(**text2img.components)

# now you can use text2img(...), img2img(...), inpaint(...) just like the call methods of each respective pipeline
```

### Create web demos using `gradio`

The Stable Diffusion pipelines are automatically supported in [Gradio](https://github.com/gradio-app/gradio/), a library that makes creating beautiful and user-friendly machine learning apps on the web a breeze. First, make sure you have Gradio installed:

```sh
pip install -U gradio
```

Then, create a web demo around any Stable Diffusion-based pipeline. For example, you can create an image generation pipeline in a single line of code with Gradio's [`Interface.from_pipeline`](https://www.gradio.app/docs/interface#interface-from-pipeline) function:

```py
from diffusers import StableDiffusionPipeline
import gradio as gr

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

gr.Interface.from_pipeline(pipe).launch()
```

which opens an intuitive drag-and-drop interface in your browser:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gradio-panda.png)

Similarly, you could create a demo for an image-to-image pipeline with:

```py
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr


pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

gr.Interface.from_pipeline(pipe).launch()
```

By default, the web demo runs on a local server. If you'd like to share it with others, you can generate a temporary public
link by setting `share=True` in `launch()`. Or, you can host your demo on [Hugging Face Spaces](https://huggingface.co/spaces)https://huggingface.co/spaces for a permanent link.