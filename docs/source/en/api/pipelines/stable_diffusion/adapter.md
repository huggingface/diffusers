<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# T2I-Adapter

[T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.08453) by Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie.

Using the pretrained models we can provide control images (for example, a depth map) to control Stable Diffusion text-to-image generation so that it follows the structure of the depth image and fills in the details.

The abstract of the paper is the following:

*The incredible generative ability of large-scale text-to-image (T2I) models has demonstrated strong power of learning complex structures and meaningful semantics. However, relying solely on text prompts cannot fully take advantage of the knowledge learned by the model, especially when flexible and accurate controlling (e.g., color and structure) is needed. In this paper, we aim to ``dig out" the capabilities that T2I models have implicitly learned, and then explicitly use them to control the generation more granularly. Specifically, we propose to learn simple and lightweight T2I-Adapters to align internal knowledge in T2I models with external control signals, while freezing the original large T2I models. In this way, we can train various adapters according to different conditions, achieving rich control and editing effects in the color and structure of the generation results. Further, the proposed T2I-Adapters have attractive properties of practical value, such as composability and generalization ability. Extensive experiments demonstrate that our T2I-Adapter has promising generation quality and a wide range of applications.*

This model was contributed by the community contributor [HimariO](https://github.com/HimariO) ❤️ .

## StableDiffusionAdapterPipeline

[[autodoc]] StableDiffusionAdapterPipeline
    - all
    - __call__
    - enable_attention_slicing
    - disable_attention_slicing
    - enable_vae_slicing
    - disable_vae_slicing
    - enable_xformers_memory_efficient_attention
    - disable_xformers_memory_efficient_attention

## StableDiffusionXLAdapterPipeline

[[autodoc]] StableDiffusionXLAdapterPipeline
    - all
    - __call__
    - enable_attention_slicing
    - disable_attention_slicing
    - enable_vae_slicing
    - disable_vae_slicing
    - enable_xformers_memory_efficient_attention
    - disable_xformers_memory_efficient_attention
