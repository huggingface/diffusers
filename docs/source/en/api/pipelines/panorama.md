<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MultiDiffusion

[MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://huggingface.co/papers/2302.08113) is by Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel.

The abstract from the paper is:

*Recent advances in text-to-image generation with diffusion models present transformative capabilities in image quality. However, user controllability of the generated image, and fast adaptation to new tasks still remains an open challenge, currently mostly addressed by costly and long re-training and fine-tuning or ad-hoc adaptations to specific image generation tasks. In this work, we present MultiDiffusion, a unified framework that enables versatile and controllable image generation, using a pre-trained text-to-image diffusion model, without any further training or finetuning. At the center of our approach is a new generation process, based on an optimization task that binds together multiple diffusion generation processes with a shared set of parameters or constraints. We show that MultiDiffusion can be readily applied to generate high quality and diverse images that adhere to user-provided controls, such as desired aspect ratio (e.g., panorama), and spatial guiding signals, ranging from tight segmentation masks to bounding boxes.*

You can find additional information about MultiDiffusion on the [project page](https://multidiffusion.github.io/), [original codebase](https://github.com/omerbt/MultiDiffusion), and try it out in a [demo](https://huggingface.co/spaces/weizmannscience/MultiDiffusion).

## Tips

While calling [`StableDiffusionPanoramaPipeline`], it's possible to specify the `view_batch_size` parameter to be > 1.
For some GPUs with high performance, this can speedup the generation process and increase VRAM usage.

To generate panorama-like images make sure you pass the width parameter accordingly. We recommend a width value of 2048 which is the default.

Circular padding is applied to ensure there are no stitching artifacts when working with panoramas to ensure a seamless transition from the rightmost part to the leftmost part. By enabling circular padding (set `circular_padding=True`), the operation applies additional crops after the rightmost point of the image, allowing the model to "see” the transition from the rightmost part to the leftmost part. This helps maintain visual consistency in a 360-degree sense and creates a proper “panorama” that can be viewed using 360-degree panorama viewers. When decoding latents in Stable Diffusion, circular padding is applied to ensure that the decoded latents match in the RGB space.

For example, without circular padding, there is a stitching artifact (default):
![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/indoor_%20no_circular_padding.png)

But with circular padding, the right and the left parts are matching (`circular_padding=True`):
![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/indoor_%20circular_padding.png)

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionPanoramaPipeline
[[autodoc]] StableDiffusionPanoramaPipeline
	- __call__
	- all

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
