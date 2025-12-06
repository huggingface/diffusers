<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Ovis-Image

![concepts](https://github.com/AIDC-AI/Ovis-Image/blob/main/docs/imgs/ovis_image_case.png)

Ovis-Image is a 7B text-to-image model specifically optimized for high-quality text rendering, designed to operate efficiently under stringent computational constraints.

[Ovis-Image Technical Report](https://arxiv.org/abs/2511.22982) from Alibaba Group, by Guo-Hua Wang, Liangfu Cao, Tianyu Cui, Minghao Fu, Xiaohao Chen, Pengxin Zhan, Jianshan Zhao, Lan Li, Bowen Fu, Jiaqi Liu, Qing-Guo Chen.

The abstract from the paper is:

*We introduce Ovis-Image, a 7B text-to-image model specifically optimized for high-quality text rendering, designed to operate efficiently under stringent computational constraints. Built upon our previous Ovis-U1 framework, Ovis-Image integrates a diffusion-based visual decoder with the stronger Ovis 2.5 multimodal backbone, leveraging a text-centric training pipeline that combines large-scale pre-training with carefully tailored post-training refinements. Despite its compact architecture, Ovis-Image achieves text rendering performance on par with significantly larger open models such as Qwen-Image and approaches closed-source systems like Seedream and GPT4o. Crucially, the model remains deployable on a single high-end GPU with moderate memory, narrowing the gap between frontier-level text rendering and practical deployment. Our results indicate that combining a strong multimodal backbone with a carefully designed, text-focused training recipe is sufficient to achieve reliable bilingual text rendering without resorting to oversized or proprietary models.*

**Highlights**: 

*   **Strong text rendering at a compact 7B scale**: Ovis-Image is a 7B text-to-image model that delivers text rendering quality comparable to much larger 20B-class systems such as Qwen-Image and competitive with leading closed-source models like GPT4o in text-centric scenarios, while remaining small enough to run on widely accessible hardware.
*   **High fidelity on text-heavy, layout-sensitive prompts**: The model excels on prompts that demand tight alignment between linguistic content and rendered typography (e.g., posters, banners, logos, UI mockups, infographics), producing legible, correctly spelled, and semantically consistent text across diverse fonts, sizes, and aspect ratios without compromising overall visual quality.
*   **Efficiency and deployability**: With its 7B parameter budget and streamlined architecture, Ovis-Image fits on a single high-end GPU with moderate memory, supports low-latency interactive use, and scales to batch production serving, bringing near–frontier text rendering to applications where tens-of-billions–parameter models are impractical.


This pipeline was contributed by Ovis-Image Team. The original codebase can be found [here](https://github.com/AIDC-AI/Ovis-Image).

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`AIDC-AI/Ovis-Image-7B`](https://huggingface.co/AIDC-AI/Ovis-Image-7B) | `torch.bfloat16` |

Refer to [this](https://huggingface.co/collections/AIDC-AI/ovis-image) collection for more information.

## OvisImagePipeline

[[autodoc]] OvisImagePipeline
	- all
	- __call__

## OvisImagePipelineOutput

[[autodoc]] pipelines.ovis_image.pipeline_output.OvisImagePipelineOutput
