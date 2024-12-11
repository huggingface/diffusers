<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNetUnion

ControlNetUnionModel is an implementation of ControlNet for Stable Diffusion XL.

The ControlNet model was introduced in [ControlNetPlus](https://github.com/xinsir6/ControlNetPlus) by xinsir6. It supports multiple conditioning inputs without increasing computation.

*We design a new architecture that can support 10+ control types in condition text-to-image generation and can generate high resolution images visually comparable with midjourney. The network is based on the original ControlNet architecture, we propose two new modules to: 1 Extend the original ControlNet to support different image conditions using the same network parameter. 2 Support multiple conditions input without increasing computation offload, which is especially important for designers who want to edit image in detail, different conditions use the same condition encoder, without adding extra computations or parameters.*


## StableDiffusionXLControlNetUnionPipeline
[[autodoc]] StableDiffusionXLControlNetUnionPipeline
	- all
	- __call__

## StableDiffusionXLControlNetUnionImg2ImgPipeline
[[autodoc]] StableDiffusionXLControlNetUnionImg2ImgPipeline
	- all
	- __call__

## StableDiffusionXLControlNetUnionInpaintPipeline
[[autodoc]] StableDiffusionXLControlNetUnionInpaintPipeline
	- all
	- __call__
