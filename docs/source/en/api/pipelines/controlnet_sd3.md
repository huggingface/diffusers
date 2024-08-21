<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet with Stable Diffusion 3

StableDiffusion3ControlNetPipeline is an implementation of ControlNet for Stable Diffusion 3.

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

The abstract from the paper is:

*We present ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models. ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with "zero convolutions" (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning. We test various conditioning controls, eg, edges, depth, segmentation, human pose, etc, with Stable Diffusion, using single or multiple conditions, with or without prompts. We show that the training of ControlNets is robust with small (<50k) and large (>1m) datasets. Extensive results show that ControlNet may facilitate wider applications to control image diffusion models.*

This controlnet code is mainly implemented by [The InstantX Team](https://huggingface.co/InstantX). The inpainting-related code was developed by [The Alimama Creative Team](https://huggingface.co/alimama-creative). You can find pre-trained checkpoints for SD3-ControlNet in the table below:


| ControlNet type | Developer | Link |
| -------- | ---------- | ---- |
| Canny | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/InstantX/SD3-Controlnet-Canny) |
| Pose | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/InstantX/SD3-Controlnet-Pose) |
| Tile | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/InstantX/SD3-Controlnet-Tile) |
| Inpainting | [The AlimamaCreative Team](https://huggingface.co/alimama-creative) | [link](https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting) |


<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusion3ControlNetPipeline
[[autodoc]] StableDiffusion3ControlNetPipeline
	- all
	- __call__

## StableDiffusion3ControlNetInpaintingPipeline
[[autodoc]] pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet_inpainting.StableDiffusion3ControlNetInpaintingPipeline
	- all
	- __call__

## StableDiffusion3PipelineOutput
[[autodoc]] pipelines.stable_diffusion_3.pipeline_output.StableDiffusion3PipelineOutput
