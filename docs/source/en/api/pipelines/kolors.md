<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kolors/kolors_header_collage.png)

Kolors is a large-scale text-to-image generation model based on latent diffusion, developed by [the Kuaishou Kolors team](kwai-kolors@kuaishou.com). Trained on billions of text-image pairs, Kolors exhibits significant advantages over both open-source and closed-source models in visual quality, complex semantic accuracy, and text rendering for both Chinese and English characters. Furthermore, Kolors supports both Chinese and English inputs, demonstrating strong performance in understanding and generating Chinese-specific content. For more details, please refer to this [technical report](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf).

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

## KolorsPipeline

[[autodoc]] KolorsPipeline

- all
- __call__
