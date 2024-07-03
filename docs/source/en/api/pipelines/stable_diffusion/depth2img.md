<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Depth-to-image

The Stable Diffusion model can also infer depth based on an image using [MiDaS](https://github.com/isl-org/MiDaS). This allows you to pass a text prompt and an initial image to condition the generation of new images as well as a `depth_map` to preserve the image structure.

<Tip>

Make sure to check out the Stable Diffusion [Tips](overview#tips) section to learn how to explore the tradeoff between scheduler speed and quality, and how to reuse pipeline components efficiently!

If you're interested in using one of the official checkpoints for a task, explore the [CompVis](https://huggingface.co/CompVis), [Runway](https://huggingface.co/runwayml), and [Stability AI](https://huggingface.co/stabilityai) Hub organizations!

</Tip>

## StableDiffusionDepth2ImgPipeline

[[autodoc]] StableDiffusionDepth2ImgPipeline
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
