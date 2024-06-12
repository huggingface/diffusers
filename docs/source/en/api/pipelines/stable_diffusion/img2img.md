<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Image-to-image

The Stable Diffusion model can also be applied to image-to-image generation by passing a text prompt and an initial image to condition the generation of new images.

The [`StableDiffusionImg2ImgPipeline`] uses the diffusion-denoising mechanism proposed in [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://huggingface.co/papers/2108.01073) by Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon.

The abstract from the paper is:

*Guided image synthesis enables everyday users to create and edit photo-realistic images with minimum effort. The key challenge is balancing faithfulness to the user input (e.g., hand-drawn colored strokes) and realism of the synthesized image. Existing GAN-based methods attempt to achieve such balance using either conditional GANs or GAN inversions, which are challenging and often require additional training data or loss functions for individual applications. To address these issues, we introduce a new image synthesis and editing method, Stochastic Differential Editing (SDEdit), based on a diffusion model generative prior, which synthesizes realistic images by iteratively denoising through a stochastic differential equation (SDE). Given an input image with user guide of any type, SDEdit first adds noise to the input, then subsequently denoises the resulting image through the SDE prior to increase its realism. SDEdit does not require task-specific training or inversions and can naturally achieve the balance between realism and faithfulness. SDEdit significantly outperforms state-of-the-art GAN-based methods by up to 98.09% on realism and 91.72% on overall satisfaction scores, according to a human perception study, on multiple tasks, including stroke-based image synthesis and editing as well as image compositing.*

<Tip>

Make sure to check out the Stable Diffusion [Tips](overview#tips) section to learn how to explore the tradeoff between scheduler speed and quality, and how to reuse pipeline components efficiently!

</Tip>

## StableDiffusionImg2ImgPipeline

[[autodoc]] StableDiffusionImg2ImgPipeline
	- all
	- __call__
	- enable_attention_slicing
	- disable_attention_slicing
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention
	- load_textual_inversion
	- from_single_file
	- load_lora_weights
	- save_lora_weights

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput

## FlaxStableDiffusionImg2ImgPipeline

[[autodoc]] FlaxStableDiffusionImg2ImgPipeline
	- all
	- __call__

## FlaxStableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput
