<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UViT2DModel

The [U-ViT](https://hf.co/papers/2209.12152) model is a vision transformer (ViT) based UNet. This model incorporates elements from ViT (considers all inputs such as time, conditions and noisy image patches as tokens) and a UNet (long skip connections between the shallow and deep layers). The skip connection is important for predicting pixel-level features. An additional 3x3 convolutional block is applied prior to the final output to improve image quality.

The abstract from the paper is:

*Vision transformers (ViT) have shown promise in various vision tasks while the U-Net based on a convolutional neural network (CNN) remains dominant in diffusion models. We design a simple and general ViT-based architecture (named U-ViT) for image generation with diffusion models. U-ViT is characterized by treating all inputs including the time, condition and noisy image patches as tokens and employing long skip connections between shallow and deep layers. We evaluate U-ViT in unconditional and class-conditional image generation, as well as text-to-image generation tasks, where U-ViT is comparable if not superior to a CNN-based U-Net of a similar size. In particular, latent diffusion models with U-ViT achieve record-breaking FID scores of 2.29 in class-conditional image generation on ImageNet 256x256, and 5.48 in text-to-image generation on MS-COCO, among methods without accessing large external datasets during the training of generative models. Our results suggest that, for diffusion-based image modeling, the long skip connection is crucial while the down-sampling and up-sampling operators in CNN-based U-Net are not always necessary. We believe that U-ViT can provide insights for future research on backbones in diffusion models and benefit generative modeling on large scale cross-modality datasets.*.

## UViT2DModel

[[autodoc]] UVit2DModel

## UViT2DConvEmbed

[[autodoc]] UViT2DConvEmbed

## UViTBlock

[[autodoc]] UViTBlock

## ConvNextBlock

[[autodoc]] ConvNextBlock

## ConvNextBlock

[[autodoc]] ConvNextBlock
