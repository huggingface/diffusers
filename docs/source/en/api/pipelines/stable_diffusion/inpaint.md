<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Inpainting

The Stable Diffusion model can also be applied to inpainting which lets you edit specific parts of an image by providing a mask and a text prompt using Stable Diffusion.

## Tips

It is recommended to use this pipeline with checkpoints that have been specifically fine-tuned for inpainting, such
as [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting). Default
text-to-image Stable Diffusion checkpoints, such as
[runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) are also compatible but they might be less performant.

<Tip>

Make sure to check out the Stable Diffusion [Tips](overview#tips) section to learn how to explore the tradeoff between scheduler speed and quality, and how to reuse pipeline components efficiently!

If you're interested in using one of the official checkpoints for a task, explore the [CompVis](https://huggingface.co/CompVis), [Runway](https://huggingface.co/runwayml), and [Stability AI](https://huggingface.co/stabilityai) Hub organizations!

</Tip>

## StableDiffusionInpaintPipeline

[[autodoc]] StableDiffusionInpaintPipeline
	- all
	- __call__
	- enable_attention_slicing
	- disable_attention_slicing
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention
	- load_textual_inversion
	- load_lora_weights
	- save_lora_weights

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput

## FlaxStableDiffusionInpaintPipeline

[[autodoc]] FlaxStableDiffusionInpaintPipeline
	- all
	- __call__

## FlaxStableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput
