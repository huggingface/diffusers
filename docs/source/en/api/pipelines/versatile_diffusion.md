<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Versatile Diffusion

Versatile Diffusion was proposed in [Versatile Diffusion: Text, Images and Variations All in One Diffusion Model](https://huggingface.co/papers/2211.08332) by Xingqian Xu, Zhangyang Wang, Eric Zhang, Kai Wang, Humphrey Shi.

The abstract from the paper is:

*Recent advances in diffusion models have set an impressive milestone in many generation tasks, and trending works such as DALL-E2, Imagen, and Stable Diffusion have attracted great interest. Despite the rapid landscape changes, recent new approaches focus on extensions and performance rather than capacity, thus requiring separate models for separate tasks. In this work, we expand the existing single-flow diffusion pipeline into a multi-task multimodal network, dubbed Versatile Diffusion (VD), that handles multiple flows of text-to-image, image-to-text, and variations in one unified model. The pipeline design of VD instantiates a unified multi-flow diffusion framework, consisting of sharable and swappable layer modules that enable the crossmodal generality beyond images and text. Through extensive experiments, we demonstrate that VD successfully achieves the following: a) VD outperforms the baseline approaches and handles all its base tasks with competitive quality; b) VD enables novel extensions such as disentanglement of style and semantics, dual- and multi-context blending, etc.; c) The success of our multi-flow multimodal framework over images and text may inspire further diffusion-based universal AI research.*

## Tips

You can load the more memory intensive "all-in-one" [`VersatileDiffusionPipeline`] that supports all the tasks or use the individual pipelines which are more memory efficient.

| **Pipeline**                                         | **Supported tasks**               |
|------------------------------------------------------|-----------------------------------|
| [`VersatileDiffusionPipeline`]                       | all of the below                  |
| [`VersatileDiffusionTextToImagePipeline`]            | text-to-image                     |
| [`VersatileDiffusionImageVariationPipeline`]         | image variation                   |
| [`VersatileDiffusionDualGuidedPipeline`]             | image-text dual guided generation |

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## VersatileDiffusionPipeline
[[autodoc]] VersatileDiffusionPipeline

## VersatileDiffusionTextToImagePipeline
[[autodoc]] VersatileDiffusionTextToImagePipeline
	- all
	- __call__

## VersatileDiffusionImageVariationPipeline
[[autodoc]] VersatileDiffusionImageVariationPipeline
	- all
	- __call__

## VersatileDiffusionDualGuidedPipeline
[[autodoc]] VersatileDiffusionDualGuidedPipeline
	- all
	- __call__
