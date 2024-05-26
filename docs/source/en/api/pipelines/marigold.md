<!--Copyright 2024 Marigold authors and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Marigold Pipelines for Computer Vision Tasks

![marigold](https://marigoldmonodepth.github.io/images/teaser_collage_compressed.jpg)

Marigold was proposed in [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://huggingface.co/papers/2312.02145), a CVPR 2024 Oral paper by [Bingxin Ke](http://www.kebingxin.com/), [Anton Obukhov](https://www.obukhov.ai/), [Shengyu Huang](https://shengyuh.github.io/), [Nando Metzger](https://nandometzger.github.io/), [Rodrigo Caye Daudt](https://rcdaudt.github.io/), and [Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en). 
In the nutshell, the idea is to repurpose the rich generative prior of Text-to-Image Latent Diffusion Models (LDMs) for traditional computer vision tasks. 
Initially, this was done to fine-tune Stable Diffusion 2.1 for monocular depth estimation, as shown in the teaser above. 
Later, 
- [Tianfu Wang](https://tianfwang.github.io/) trained the first Latent Consistency Model (LCM) of Marigold, which unlocked fast single-step inference;
- [Kevin Qu](https://www.linkedin.com/in/kevin-qu-b3417621b/?locale=en_US) extended the approach to surface normals estimation;
- [Anton Obukhov](https://www.obukhov.ai/) contributed the pipelines into diffusers (enabled and supported by [YiYi Xu](https://yiyixuxu.github.io/) and [Sayak Paul](https://sayak.dev/)).

The abstract from the paper is:

*Monocular depth estimation is a fundamental computer vision task. Recovering 3D depth from a single image is geometrically ill-posed and requires scene understanding, so it is not surprising that the rise of deep learning has led to a breakthrough. The impressive progress of monocular depth estimators has mirrored the growth in model capacity, from relatively modest CNNs to large Transformer architectures. Still, monocular depth estimators tend to struggle when presented with images with unfamiliar content and layout, since their knowledge of the visual world is restricted by the data seen during training, and challenged by zero-shot generalization to new domains. This motivates us to explore whether the extensive priors captured in recent generative diffusion models can enable better, more generalizable depth estimation. We introduce Marigold, a method for affine-invariant monocular depth estimation that is derived from Stable Diffusion and retains its rich prior knowledge. The estimator can be fine-tuned in a couple of days on a single GPU using only synthetic training data. It delivers state-of-the-art performance across a wide range of datasets, including over 20% performance gains in specific cases. Project page: https://marigoldmonodepth.github.io.*

## Available Pipelines

Each pipeline supports one Computer Vision task, which takes an input RGB image as input, and produces a *prediction* of the modality of interest, such as a depth map of the input image. 
Currently, the following tasks are implemented:

| Pipeline                                                                                                                                    | Predicted Modalities                                                                                             |                                                                       Demos                                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------:|
| [MarigoldDepthPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py)     | [Depth](https://en.wikipedia.org/wiki/Depth_map), [Disparity](https://en.wikipedia.org/wiki/Binocular_disparity) | [Fast Demo (LCM)](https://huggingface.co/spaces/prs-eth/marigold-lcm), [Slow Original Demo (DDIM)](https://huggingface.co/spaces/prs-eth/marigold) |
| [MarigoldNormalsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py) | [Surface normals](https://en.wikipedia.org/wiki/Normal_mapping)                                                  |                                   [Fast Demo (LCM)](https://huggingface.co/spaces/prs-eth/marigold-normals-lcm)                                    |

<Tip>
Marigold is a universal diffusion-based framework for precise dense regression. 
If you believe that an important dense prediction task (such as monocular depth or surface normals estimation) would benefit the community, consider voting for adding such functionality in a GitHub issue of diffusers.
</Tip>


## Available Checkpoints

The original checkpoints can be found under the [PRS-ETH](https://huggingface.co/prs-eth/) Hugging Face organization. 
These checkpoints are meant to work with diffusers pipelines and the [original codebase](https://github.com/prs-eth/marigold). 
The original code can also be used to train new checkpoints.

| Checkpoint                                                                                    | Modality | Comment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|-----------------------------------------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [prs-eth/marigold-v1-0](https://huggingface.co/prs-eth/marigold-v1-0)                         | Depth    | The first Marigold Depth checkpoint, which predicts *affine-invariant depth* maps. The performance of this checkpoint in benchmarks was studied in the original [paper](https://huggingface.co/papers/2312.02145). Designed to be used with the `DDIMScheduler` at inference, it requires at least 10 steps to get reliable predictions. Affine-invariant depth prediction has a range of values in each pixel between 0 (near plane) and 1 (far plane), both planes are chosen by the model as part of inference process. See `MarigoldImageProcessor` reference for visualization utilities. |
| [prs-eth/marigold-lcm-v1-0](https://huggingface.co/prs-eth/marigold-lcm-v1-0)                 | Depth    | The fast Marigold Depth checkpoint, fine-tuned from `prs-eth/marigold-v1-0`. Designed to be used with the `LCMScheduler` at inference, it requires as little as 1 step to get reliable predictions. The prediction reliability saturates at 4 steps and declines after that.                                                                                                                                                                                                                                                                                                                   |
| [prs-eth/marigold-normals-v0-1](https://huggingface.co/prs-eth/marigold-normals-v0-1)         | Normals  | A preview checkpoint for the Marigold Normals pipeline. Designed to be used with the `DDIMScheduler` at inference, it requires at least 10 steps to get reliable predictions. The surface normals predictions are unit-length 3D vectors with values in the range from -1 to 1. *This checkpoint will be phased out after the release of `v1-0` version.*                                                                                                                                                                                                                                      |
| [prs-eth/marigold-normals-lcm-v0-1](https://huggingface.co/prs-eth/marigold-normals-lcm-v0-1) | Normals  | The fast Marigold Normals checkpoint, fine-tuned from `prs-eth/marigold-normals-v0-1`. Designed to be used with the `LCMScheduler` at inference, it requires as little as 1 steps to get reliable predictions. The prediction reliability saturates at 4 steps and declines after that. *This checkpoint will be phased out after the release of `v1-0` version.*                                                                                                                                                                                                                              |

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines. Also, to know more about reducing the memory usage of this pipeline, refer to the ["Reduce memory usage"] section [here](../../using-diffusers/svd#reduce-memory-usage).

</Tip>

<Tip warning={true}>

Marigold pipelines were designed and tested only with `DDIMScheduler` and `LCMScheduler`. 
Additionally, the default values for `num_inference_steps` and `processing_resolution` in the `__call__` method of the pipeline are set to `None` (see the API reference). 
Unless set explicitly, these values will default to the ones stored in the checkpoint `model_index.json`. 
This is done to ensure high-quality predictions when calling the pipeline with just the `image` argument. 
See the Use Cases section to learn about various presets of the arguments. 

</Tip>

## MarigoldDepthPipeline
[[autodoc]] MarigoldDepthPipeline
	- all
	- __call__

## MarigoldNormalsPipeline
[[autodoc]] MarigoldNormalsPipeline
	- all
	- __call__

## MarigoldDepthOutput
[[autodoc]] pipelines.marigold.pipeline_marigold_depth.MarigoldDepthOutput

## MarigoldNormalsOutput
[[autodoc]] pipelines.marigold.pipeline_marigold_normals.MarigoldNormalsOutput