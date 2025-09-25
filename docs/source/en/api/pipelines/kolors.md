<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
  <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22">
</div>

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kolors/kolors_header_collage.png)

Kolors is a large-scale text-to-image generation model based on latent diffusion, developed by [the Kuaishou Kolors team](https://github.com/Kwai-Kolors/Kolors). Trained on billions of text-image pairs, Kolors exhibits significant advantages over both open-source and closed-source models in visual quality, complex semantic accuracy, and text rendering for both Chinese and English characters. Furthermore, Kolors supports both Chinese and English inputs, demonstrating strong performance in understanding and generating Chinese-specific content. For more details, please refer to this [technical report](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf).

The abstract from the technical report is:

*We present Kolors, a latent diffusion model for text-to-image synthesis, characterized by its profound understanding of both English and Chinese, as well as an impressive degree of photorealism. There are three key insights contributing to the development of Kolors. Firstly, unlike large language model T5 used in Imagen and Stable Diffusion 3, Kolors is built upon the General Language Model (GLM), which enhances its comprehension capabilities in both English and Chinese. Moreover, we employ a multimodal large language model to recaption the extensive training dataset for fine-grained text understanding. These strategies significantly improve Kolors’ ability to comprehend intricate semantics, particularly those involving multiple entities, and enable its advanced text rendering capabilities. Secondly, we divide the training of Kolors into two phases: the concept learning phase with broad knowledge and the quality improvement phase with specifically curated high-aesthetic data. Furthermore, we investigate the critical role of the noise schedule and introduce a novel schedule to optimize high-resolution image generation. These strategies collectively enhance the visual appeal of the generated high-resolution images. Lastly, we propose a category-balanced benchmark KolorsPrompts, which serves as a guide for the training and evaluation of Kolors. Consequently, even when employing the commonly used U-Net backbone, Kolors has demonstrated remarkable performance in human evaluations, surpassing the existing open-source models and achieving Midjourney-v6 level performance, especially in terms of visual appeal. We will release the code and weights of Kolors at <https://github.com/Kwai-Kolors/Kolors>, and hope that it will benefit future research and applications in the visual generation community.*

## Usage Example

```python
import torch

from diffusers import DPMSolverMultistepScheduler, KolorsPipeline

pipe = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

image = pipe(
    prompt='一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着"可图"',
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
).images[0]

image.save("kolors_sample.png")
```

### IP Adapter

Kolors needs a different IP Adapter to work, and it uses [Openai-CLIP-336](https://huggingface.co/openai/clip-vit-large-patch14-336) as an image encoder.

> [!TIP]
> Using an IP Adapter with Kolors requires more than 24GB of VRAM. To use it, we recommend using [`~DiffusionPipeline.enable_model_cpu_offload`] on consumer GPUs.

> [!TIP]
> While Kolors is integrated in Diffusers, you need to load the image encoder from a revision to use the safetensor files. You can still use the main branch of the original repository if you're comfortable loading pickle checkpoints.

```python
import torch
from transformers import CLIPVisionModelWithProjection

from diffusers import DPMSolverMultistepScheduler, KolorsPipeline
from diffusers.utils import load_image

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "Kwai-Kolors/Kolors-IP-Adapter-Plus",
    subfolder="image_encoder",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    revision="refs/pr/4",
)

pipe = KolorsPipeline.from_pretrained(
    "Kwai-Kolors/Kolors-diffusers", image_encoder=image_encoder, torch_dtype=torch.float16, variant="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

pipe.load_ip_adapter(
    "Kwai-Kolors/Kolors-IP-Adapter-Plus",
    subfolder="",
    weight_name="ip_adapter_plus_general.safetensors",
    revision="refs/pr/4",
    image_encoder_folder=None,
)
pipe.enable_model_cpu_offload()

ipa_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kolors/cat_square.png")

image = pipe(
    prompt="best quality, high quality",
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
    ip_adapter_image=ipa_image,
).images[0]

image.save("kolors_ipa_sample.png")
```

## KolorsPipeline

[[autodoc]] KolorsPipeline

- all
- __call__

## KolorsImg2ImgPipeline

[[autodoc]] KolorsImg2ImgPipeline

- all
- __call__

