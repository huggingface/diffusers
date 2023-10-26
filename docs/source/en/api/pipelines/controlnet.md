<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

The abstract from the paper is:

*We present a neural network structure, ControlNet, to control pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small (< 50k). Moreover, training a ControlNet is as fast as fine-tuning a diffusion model, and the model can be trained on a personal devices. Alternatively, if powerful computation clusters are available, the model can scale to large amounts (millions to billions) of data. We report that large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like edge maps, segmentation maps, keypoints, etc. This may enrich the methods to control large diffusion models and further facilitate related applications.*

This model was contributed by [takuma104](https://huggingface.co/takuma104). ❤️

The original codebase can be found at [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet), and you can find official ControlNet checkpoints on [lllyasviel's](https://huggingface.co/lllyasviel) Hub profile.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionControlNetPipeline
[[autodoc]] StableDiffusionControlNetPipeline
	- all
	- __call__
	- enable_attention_slicing
	- disable_attention_slicing
	- enable_vae_slicing
	- disable_vae_slicing
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention
	- load_textual_inversion

## StableDiffusionControlNetImg2ImgPipeline
[[autodoc]] StableDiffusionControlNetImg2ImgPipeline
	- all
	- __call__
	- enable_attention_slicing
	- disable_attention_slicing
	- enable_vae_slicing
	- disable_vae_slicing
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention
	- load_textual_inversion

## StableDiffusionControlNetInpaintPipeline
[[autodoc]] StableDiffusionControlNetInpaintPipeline
	- all
	- __call__
	- enable_attention_slicing
	- disable_attention_slicing
	- enable_vae_slicing
	- disable_vae_slicing
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention
	- load_textual_inversion

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput

## FlaxStableDiffusionControlNetPipeline
[[autodoc]] FlaxStableDiffusionControlNetPipeline
	- all
	- __call__

## FlaxStableDiffusionControlNetPipelineOutput

[[autodoc]] pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput