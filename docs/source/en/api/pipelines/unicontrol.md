<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UniControl

UniControl was introduced in [UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild
](https://huggingface.co/papers/2305.11147) by Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, and Ran Xu.

UniControl leverages a Mixture-Of-Experts architecture on top of the ControlNet architecture to enable a multi-task representation for a set of trained controlnet weights.

The abstract from the paper is:

*Achieving machine autonomy and human control often represent divergent objectives in the design of interactive AI systems. Visual generative foundation models such as Stable Diffusion show promise in navigating these goals, especially when prompted with arbitrary languages. However, they often fall short in generating images with spatial, structural, or geometric controls. The integration of such controls, which can accommodate various visual conditions in a single unified model, remains an unaddressed challenge. In response, we introduce UniControl, a new generative foundation model that consolidates a wide array of controllable condition-to-image (C2I) tasks within a singular framework, while still allowing for arbitrary language prompts. UniControl enables pixel-level-precise image generation, where visual conditions primarily influence the generated structures and language prompts guide the style and context. To equip UniControl with the capacity to handle diverse visual conditions, we augment pretrained text-to-image diffusion models and introduce a task-aware HyperNet to modulate the diffusion models, enabling the adaptation to different C2I tasks simultaneously. Trained on nine unique C2I tasks, UniControl demonstrates impressive zero-shot generation abilities with unseen visual conditions. Experimental results show that UniControl often surpasses the performance of single-task-controlled methods of comparable model sizes. This control versatility positions UniControl as a significant advancement in the realm of controllable visual generation.*

This model was contributed by [Abhinay1997](https://github.com/Abhinay1997) and [Can Qin](https://github.com/canqin001). ❤️

The original codebase can be found at [salesforce/UniControl](https://github.com/salesforce/UniControl), and you can find official ControlNet checkpoints on [Can Qin's](https://huggingface.co) Hub profile.

<Tip>

Make sure to check out the Schedulers [guide](/using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](/using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionUniControlPipeline
[[autodoc]] StableDiffusionUniControlPipeline
	- all
	- __call__
	- enable_attention_slicing
	- disable_attention_slicing
	- enable_vae_slicing
	- disable_vae_slicing
	- enable_xformers_memory_efficient_attention
	- disable_xformers_memory_efficient_attention
	- load_textual_inversion