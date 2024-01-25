<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UVit2DModel

The [U-ViT](https://hf.co/papers/2301.11093) model is a vision transformer (ViT) based UNet. This model incorporates elements from ViT (considers all inputs such as time, conditions and noisy image patches as tokens) and a UNet (long skip connections between the shallow and deep layers). The skip connection is important for predicting pixel-level features. An additional 3x3 convolutional block is applied prior to the final output to improve image quality.

The abstract from the paper is:

*Currently, applying diffusion models in pixel space of high resolution images is difficult. Instead, existing approaches focus on diffusion in lower dimensional spaces (latent diffusion), or have multiple super-resolution levels of generation referred to as cascades. The downside is that these approaches add additional complexity to the diffusion framework. This paper aims to improve denoising diffusion for high resolution images while keeping the model as simple as possible. The paper is centered around the research question: How can one train a standard denoising diffusion models on high resolution images, and still obtain performance comparable to these alternate approaches? The four main findings are: 1) the noise schedule should be adjusted for high resolution images, 2) It is sufficient to scale only a particular part of the architecture, 3) dropout should be added at specific locations in the architecture, and 4) downsampling is an effective strategy to avoid high resolution feature maps. Combining these simple yet effective techniques, we achieve state-of-the-art on image generation among diffusion models without sampling modifiers on ImageNet.*

## UVit2DModel

[[autodoc]] UVit2DModel

## UVit2DConvEmbed

[[autodoc]] models.unets.uvit_2d.UVit2DConvEmbed

## UVitBlock

[[autodoc]] models.unets.uvit_2d.UVitBlock

## ConvNextBlock

[[autodoc]] models.unets.uvit_2d.ConvNextBlock

## ConvMlmLayer

[[autodoc]] models.unets.uvit_2d.ConvMlmLayer
