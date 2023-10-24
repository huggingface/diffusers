<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Attend-and-Excite

Attend-and-Excite for Stable Diffusion was proposed in [Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models](https://attendandexcite.github.io/Attend-and-Excite/) and provides textual attention control over image generation.

The abstract from the paper is:

*Text-to-image diffusion models have recently received a lot of interest for their astonishing ability to produce high-fidelity images from text only. However, achieving one-shot generation that aligns with the user's intent is nearly impossible, yet small changes to the input prompt often result in very different images. This leaves the user with little semantic control. To put the user in control, we show how to interact with the diffusion process to flexibly steer it along semantic directions. This semantic guidance (SEGA) allows for subtle and extensive edits, changes in composition and style, as well as optimizing the overall artistic conception. We demonstrate SEGA's effectiveness on a variety of tasks and provide evidence for its versatility and flexibility.*

You can find additional information about Attend-and-Excite on the [project page](https://attendandexcite.github.io/Attend-and-Excite/), the [original codebase](https://github.com/AttendAndExcite/Attend-and-Excite), or try it out in a [demo](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite).

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionAttendAndExcitePipeline

[[autodoc]] StableDiffusionAttendAndExcitePipeline
	- all
	- __call__

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput