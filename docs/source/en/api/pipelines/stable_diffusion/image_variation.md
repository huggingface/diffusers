<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Image variation

The Stable Diffusion model can also generate variations from an input image. It uses a fine-tuned version of a Stable Diffusion model by [Justin Pinkney](https://www.justinpinkney.com/) from [Lambda](https://lambdalabs.com/).

The original codebase can be found at [LambdaLabsML/lambda-diffusers](https://github.com/LambdaLabsML/lambda-diffusers#stable-diffusion-image-variations) and additional official checkpoints for image variation can be found at [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).

<Tip>

Make sure to check out the Stable Diffusion [Tips](./overview#tips) section to learn how to explore the tradeoff between scheduler speed and quality, and how to reuse pipeline components efficiently!

</Tip>

## StableDiffusionImageVariationPipeline

[[autodoc]] StableDiffusionImageVariationPipeline
	- all
	- __call__
	- enable_attention_slicing
	- disable_attention_slicing
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
