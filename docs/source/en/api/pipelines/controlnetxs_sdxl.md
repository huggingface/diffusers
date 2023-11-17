<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet with Stable Diffusion XL

ControlNet-XS was introduced in [ControlNet-XS](https://vislearn.github.io/ControlNet-XS/) by Denis Zavadski and Carsten Rother. It is based on the observation that the control model in the [original ControlNet](https://huggingface.co/papers/2302.05543) can be made much smaller and still produces good results.

As with the original ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

<Tip warning={true}>

🧪 Many of the SDXL ControlNet checkpoints are experimental, and there is a lot of room for improvement. Feel free to open an [Issue](https://github.com/huggingface/diffusers/issues/new/choose) and leave us feedback on how we can improve!

</Tip>

If you don't see a checkpoint you're interested in, you can train your own SDXL ControlNet with our [training script](../../../../../examples/controlnet/README_sdxl).

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionXLControlNetXSPipeline
[[autodoc]] StableDiffusionXLControlNetXSPipeline
	- all
	- __call__

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
