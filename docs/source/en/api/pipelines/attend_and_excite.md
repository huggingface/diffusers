<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

> [!WARNING]
> This pipeline is deprecated but it can still be used. However, we won't test the pipeline anymore and won't accept any changes to it. If you run into any issues, reinstall the last Diffusers version that supported this model.

# Attend-and-Excite

Attend-and-Excite for Stable Diffusion was proposed in [Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models](https://attendandexcite.github.io/Attend-and-Excite/) and provides textual attention control over image generation.

The abstract from the paper is:

*Recent text-to-image generative models have demonstrated an unparalleled ability to generate diverse and creative imagery guided by a target text prompt. While revolutionary, current state-of-the-art diffusion models may still fail in generating images that fully convey the semantics in the given text prompt. We analyze the publicly available Stable Diffusion model and assess the existence of catastrophic neglect, where the model fails to generate one or more of the subjects from the input prompt. Moreover, we find that in some cases the model also fails to correctly bind attributes (e.g., colors) to their corresponding subjects. To help mitigate these failure cases, we introduce the concept of Generative Semantic Nursing (GSN), where we seek to intervene in the generative process on the fly during inference time to improve the faithfulness of the generated images. Using an attention-based formulation of GSN, dubbed Attend-and-Excite, we guide the model to refine the cross-attention units to attend to all subject tokens in the text prompt and strengthen - or excite - their activations, encouraging the model to generate all subjects described in the text prompt. We compare our approach to alternative approaches and demonstrate that it conveys the desired concepts more faithfully across a range of text prompts.*

You can find additional information about Attend-and-Excite on the [project page](https://attendandexcite.github.io/Attend-and-Excite/), the [original codebase](https://github.com/AttendAndExcite/Attend-and-Excite), or try it out in a [demo](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## StableDiffusionAttendAndExcitePipeline

[[autodoc]] StableDiffusionAttendAndExcitePipeline
	- all
	- __call__

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
