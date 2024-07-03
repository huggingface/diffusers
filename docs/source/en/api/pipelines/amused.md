<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# aMUSEd

aMUSEd was introduced in [aMUSEd: An Open MUSE Reproduction](https://huggingface.co/papers/2401.01808) by Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen.

Amused is a lightweight text to image model based off of the [MUSE](https://arxiv.org/abs/2301.00704) architecture. Amused is particularly useful in applications that require a lightweight and fast model such as generating many images quickly at once.

Amused is a vqvae token based transformer that can generate an image in fewer forward passes than many diffusion models. In contrast with muse, it uses the smaller text encoder CLIP-L/14 instead of t5-xxl. Due to its small parameter count and few forward pass generation process, amused can generate many images quickly. This benefit is seen particularly at larger batch sizes.

The abstract from the paper is:

*We present aMUSEd, an open-source, lightweight masked image model (MIM) for text-to-image generation based on MUSE. With 10 percent of MUSE's parameters, aMUSEd is focused on fast image generation. We believe MIM is under-explored compared to latent diffusion, the prevailing approach for text-to-image generation. Compared to latent diffusion, MIM requires fewer inference steps and is more interpretable. Additionally, MIM can be fine-tuned to learn additional styles with only a single image. We hope to encourage further exploration of MIM by demonstrating its effectiveness on large-scale text-to-image generation and releasing reproducible training code. We also release checkpoints for two models which directly produce images at 256x256 and 512x512 resolutions.*

| Model | Params |
|-------|--------|
| [amused-256](https://huggingface.co/amused/amused-256) | 603M |
| [amused-512](https://huggingface.co/amused/amused-512) | 608M |

## AmusedPipeline

[[autodoc]] AmusedPipeline
	- __call__
	- all
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention

[[autodoc]] AmusedImg2ImgPipeline
	- __call__
	- all
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention

[[autodoc]] AmusedInpaintPipeline
	- __call__
	- all
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention