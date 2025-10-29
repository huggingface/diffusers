<!--
Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
Copyright 2024-2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Marigold Computer Vision

![marigold](https://marigoldmonodepth.github.io/images/teaser_collage_compressed.jpg)

Marigold was proposed in 
[Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://huggingface.co/papers/2312.02145), 
a CVPR 2024 Oral paper by 
[Bingxin Ke](http://www.kebingxin.com/), 
[Anton Obukhov](https://www.obukhov.ai/), 
[Shengyu Huang](https://shengyuh.github.io/), 
[Nando Metzger](https://nandometzger.github.io/), 
[Rodrigo Caye Daudt](https://rcdaudt.github.io/), and 
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en).
The core idea is to **repurpose the generative prior of Text-to-Image Latent Diffusion Models (LDMs) for traditional 
computer vision tasks**.
This approach was explored by fine-tuning Stable Diffusion for **Monocular Depth Estimation**, as demonstrated in the 
teaser above.

Marigold was later extended in the follow-up paper, 
[Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis](https://huggingface.co/papers/2312.02145), 
authored by 
[Bingxin Ke](http://www.kebingxin.com/), 
[Kevin Qu](https://www.linkedin.com/in/kevin-qu-b3417621b/?locale=en_US), 
[Tianfu Wang](https://tianfwang.github.io/), 
[Nando Metzger](https://nandometzger.github.io/), 
[Shengyu Huang](https://shengyuh.github.io/), 
[Bo Li](https://www.linkedin.com/in/bobboli0202/), 
[Anton Obukhov](https://www.obukhov.ai/), and 
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en).
This work expanded Marigold to support new modalities such as **Surface Normals** and **Intrinsic Image Decomposition** 
(IID), introduced a training protocol for **Latent Consistency Models** (LCM), and demonstrated **High-Resolution** (HR) 
processing capability.

> [!TIP]
> The early Marigold models (`v1-0` and earlier) were optimized for best results with at least 10 inference steps.
> LCM models were later developed to enable high-quality inference in just 1 to 4 steps.
> Marigold models `v1-1` and later use the DDIM scheduler to achieve optimal 
> results in as few as 1 to 4 steps.

## Available Pipelines

Each pipeline is tailored for a specific computer vision task, processing an input RGB image and generating a 
corresponding prediction.
Currently, the following computer vision tasks are implemented:

| Pipeline                                                                                                                                          | Recommended Model Checkpoints                                                                                                                                                                           |                              Spaces (Interactive Apps)                               | Predicted Modalities                                                                                                                                                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [MarigoldDepthPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py)           | [prs-eth/marigold-depth-v1-1](https://huggingface.co/prs-eth/marigold-depth-v1-1)                                                                                                                       |          [Depth Estimation](https://huggingface.co/spaces/prs-eth/marigold)          | [Depth](https://en.wikipedia.org/wiki/Depth_map), [Disparity](https://en.wikipedia.org/wiki/Binocular_disparity)                                                                   |
| [MarigoldNormalsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py)       | [prs-eth/marigold-normals-v1-1](https://huggingface.co/prs-eth/marigold-normals-v1-1)                                                                                                                   | [Surface Normals Estimation](https://huggingface.co/spaces/prs-eth/marigold-normals) | [Surface normals](https://en.wikipedia.org/wiki/Normal_mapping)                                                                                                                    |
| [MarigoldIntrinsicsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py) | [prs-eth/marigold-iid-appearance-v1-1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1),<br>[prs-eth/marigold-iid-lighting-v1-1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1) | [Intrinsic Image Decomposition](https://huggingface.co/spaces/prs-eth/marigold-iid)  | [Albedo](https://en.wikipedia.org/wiki/Albedo), [Materials](https://www.n.aiq3d.com/wiki/roughnessmetalnessao-map), [Lighting](https://en.wikipedia.org/wiki/Diffuse_reflection)   |

## Available Checkpoints

All original checkpoints are available under the [PRS-ETH](https://huggingface.co/prs-eth/) organization on Hugging Face.
They are designed for use with diffusers pipelines and the [original codebase](https://github.com/prs-eth/marigold), which can also be used to train 
new model checkpoints.
The following is a summary of the recommended checkpoints, all of which produce reliable results with 1 to 4 steps. 

| Checkpoint                                                                                          | Modality     | Comment                                                                                                                                                                              |
|-----------------------------------------------------------------------------------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [prs-eth/marigold-depth-v1-1](https://huggingface.co/prs-eth/marigold-depth-v1-1)                   | Depth        | Affine-invariant depth prediction assigns each pixel a value between 0 (near plane) and 1 (far plane), with both planes determined by the model during inference.                    |
| [prs-eth/marigold-normals-v0-1](https://huggingface.co/prs-eth/marigold-normals-v0-1)               | Normals      | The surface normals predictions are unit-length 3D vectors in the screen space camera, with values in the range from -1 to 1.                                                        |
| [prs-eth/marigold-iid-appearance-v1-1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) | Intrinsics   | InteriorVerse decomposition is comprised of Albedo and two BRDF material properties: Roughness and Metallicity.                                                                      | 
| [prs-eth/marigold-iid-lighting-v1-1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1)     | Intrinsics   | HyperSim decomposition of an image $I$ is comprised of Albedo $A$, Diffuse shading $S$, and Non-diffuse residual $R$: $I = A*S+R$. |

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff 
> between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to 
> efficiently load the same components into multiple pipelines. 
> Also, to know more about reducing the memory usage of this pipeline, refer to the ["Reduce memory usage"] section 
> [here](../../using-diffusers/svd#reduce-memory-usage).

> [!WARNING]
> Marigold pipelines were designed and tested with the scheduler embedded in the model checkpoint.
> The optimal number of inference steps varies by scheduler, with no universal value that works best across all cases.
> To accommodate this, the `num_inference_steps` parameter in the pipeline's `__call__` method defaults to `None` (see the 
> API reference).
> Unless set explicitly, it inherits the value from the `default_denoising_steps` field in the checkpoint configuration 
> file (`model_index.json`).
> This ensures high-quality predictions when invoking the pipeline with only the `image` argument.

See also Marigold [usage examples](../../using-diffusers/marigold_usage).

## Marigold Depth Prediction API

[[autodoc]] MarigoldDepthPipeline
	- __call__

[[autodoc]] pipelines.marigold.pipeline_marigold_depth.MarigoldDepthOutput

[[autodoc]] pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_depth

## Marigold Normals Estimation API
[[autodoc]] MarigoldNormalsPipeline
	- __call__

[[autodoc]] pipelines.marigold.pipeline_marigold_normals.MarigoldNormalsOutput

[[autodoc]] pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_normals

## Marigold Intrinsic Image Decomposition API

[[autodoc]] MarigoldIntrinsicsPipeline
	- __call__

[[autodoc]] pipelines.marigold.pipeline_marigold_intrinsics.MarigoldIntrinsicsOutput

[[autodoc]] pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_intrinsics
