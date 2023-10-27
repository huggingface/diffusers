<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet with Stable Diffusion XL

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

The abstract from the paper is:

*We present a neural network structure, ControlNet, to control pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small (< 50k). Moreover, training a ControlNet is as fast as fine-tuning a diffusion model, and the model can be trained on a personal devices. Alternatively, if powerful computation clusters are available, the model can scale to large amounts (millions to billions) of data. We report that large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like edge maps, segmentation maps, keypoints, etc. This may enrich the methods to control large diffusion models and further facilitate related applications.*

You can find additional smaller Stable Diffusion XL (SDXL) ControlNet checkpoints from the 🤗 [Diffusers](https://huggingface.co/diffusers) Hub organization, and browse [community-trained](https://huggingface.co/models?other=stable-diffusion-xl&other=controlnet) checkpoints on the Hub.

<Tip warning={true}>

🧪 Many of the SDXL ControlNet checkpoints are experimental, and there is a lot of room for improvement. Feel free to open an [Issue](https://github.com/huggingface/diffusers/issues/new/choose) and leave us feedback on how we can improve!

</Tip>

If you don't see a checkpoint you're interested in, you can train your own SDXL ControlNet with our [training script](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sdxl.md).

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionXLControlNetPipeline
[[autodoc]] StableDiffusionXLControlNetPipeline
	- all
	- __call__

## StableDiffusionXLControlNetImg2ImgPipeline
[[autodoc]] StableDiffusionXLControlNetImg2ImgPipeline
	- all
	- __call__

## StableDiffusionXLControlNetInpaintPipeline
[[autodoc]] StableDiffusionXLControlNetInpaintPipeline
	- all
	- __call__
## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput