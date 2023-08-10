<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet with Stable Diffusion XL

[Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

Using a pretrained model, we can provide control images (for example, a depth map) to control Stable Diffusion text-to-image generation so that it follows the structure of the depth image and fills in the details.

The abstract from the paper is:

*We present a neural network structure, ControlNet, to control pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small (< 50k). Moreover, training a ControlNet is as fast as fine-tuning a diffusion model, and the model can be trained on a personal devices. Alternatively, if powerful computation clusters are available, the model can scale to large amounts (millions to billions) of data. We report that large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like edge maps, segmentation maps, keypoints, etc. This may enrich the methods to control large diffusion models and further facilitate related applications.*

We provide support using ControlNets with [Stable Diffusion XL](./stable_diffusion/stable_diffusion_xl.md) (SDXL). 

There are not many ControlNet checkpoints that are compatible with SDXL at the moment. So, we trained one using Canny edge maps as the conditioning images. To know more, check out the [model card](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0). We encourage you to train custom ControlNets; we provide a [training script](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sdxl.md) for this.

You can find some results below:

<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/sdxl_controlnet_canny_grid.png" width=600/>


## StableDiffusionXLControlNetPipeline
[[autodoc]] StableDiffusionXLControlNetPipeline
	- all
	- __call__