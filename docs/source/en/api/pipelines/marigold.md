<!--Copyright 2024 Marigold authors and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Marigold for Depth Estimation and Surface Normal Estimation

![marigold](https://marigoldmonodepth.github.io/images/teaser_collage_compressed.jpg)

[Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://huggingface.co/papers/2312.02145) by Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler.

The abstract from the paper is:

*Monocular depth estimation is a fundamental computer vision task. Recovering 3D depth from a single image is geometrically ill-posed and requires scene understanding, so it is not surprising that the rise of deep learning has led to a breakthrough. The impressive progress of monocular depth estimators has mirrored the growth in model capacity, from relatively modest CNNs to large Transformer architectures. Still, monocular depth estimators tend to struggle when presented with images with unfamiliar content and layout, since their knowledge of the visual world is restricted by the data seen during training, and challenged by zero-shot generalization to new domains. This motivates us to explore whether the extensive priors captured in recent generative diffusion models can enable better, more generalizable depth estimation. We introduce Marigold, a method for affine-invariant monocular depth estimation that is derived from Stable Diffusion and retains its rich prior knowledge. The estimator can be fine-tuned in a couple of days on a single GPU using only synthetic training data. It delivers state-of-the-art performance across a wide range of datasets, including over 20% performance gains in specific cases. Project page: https://marigoldmonodepth.github.io.*

The original codebase can be found [here](https://github.com/prs-eth/marigold). The model checkpoints can be found [here](https://huggingface.co/prs-eth/).

The Marigold pipelines were contributed by [toshas](https://huggingface.co/toshas), one of the co-authors of Marigold. 

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines. Also, to know more about reducing the memory usage of this pipeline, refer to the ["Reduce memory usage"] section [here](../../using-diffusers/svd#reduce-memory-usage).

</Tip>

Sample output with Marigold:

TODO

## Notes

TODO

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