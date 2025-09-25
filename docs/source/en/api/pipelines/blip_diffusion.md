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

# BLIP-Diffusion

BLIP-Diffusion was proposed in [BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing](https://huggingface.co/papers/2305.14720). It enables zero-shot subject-driven generation and control-guided zero-shot generation.


The abstract from the paper is:

*Subject-driven text-to-image generation models create novel renditions of an input subject based on text prompts. Existing models suffer from lengthy fine-tuning and difficulties preserving the subject fidelity. To overcome these limitations, we introduce BLIP-Diffusion, a new subject-driven image generation model that supports multimodal control which consumes inputs of subject images and text prompts. Unlike other subject-driven generation models, BLIP-Diffusion introduces a new multimodal encoder which is pre-trained to provide subject representation. We first pre-train the multimodal encoder following BLIP-2 to produce visual representation aligned with the text. Then we design a subject representation learning task which enables a diffusion model to leverage such visual representation and generates new subject renditions. Compared with previous methods such as DreamBooth, our model enables zero-shot subject-driven generation, and efficient fine-tuning for customized subject with up to 20x speedup. We also demonstrate that BLIP-Diffusion can be flexibly combined with existing techniques such as ControlNet and prompt-to-prompt to enable novel subject-driven generation and editing applications. Project page at [this https URL](https://dxli94.github.io/BLIP-Diffusion-website/).*

The original codebase can be found at [salesforce/LAVIS](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion). You can find the official BLIP-Diffusion checkpoints under the [hf.co/SalesForce](https://hf.co/SalesForce) organization.

`BlipDiffusionPipeline` and `BlipDiffusionControlNetPipeline` were contributed by [`ayushtues`](https://github.com/ayushtues/).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.


## BlipDiffusionPipeline
[[autodoc]] BlipDiffusionPipeline
    - all
    - __call__

## BlipDiffusionControlNetPipeline
[[autodoc]] BlipDiffusionControlNetPipeline
    - all
    - __call__
