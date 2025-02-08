<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Lumina2

[Lumina Image 2.0: A Unified and Efficient Image Generative Model](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) is a 2 billion parameter flow-based diffusion transformer capable of generating diverse images from text descriptions.

The abstract from the paper is:

*We introduce Lumina-Image 2.0, an advanced text-to-image model that surpasses previous state-of-the-art methods across multiple benchmarks, while also shedding light on its potential to evolve into a generalist vision intelligence model. Lumina-Image 2.0 exhibits three key properties: (1) Unification – it adopts a unified architecture that treats text and image tokens as a joint sequence, enabling natural cross-modal interactions and facilitating task expansion. Besides, since high-quality captioners can provide semantically better-aligned text-image training pairs, we introduce a unified captioning system, UniCaptioner, which generates comprehensive and precise captions for the model. This not only accelerates model convergence but also enhances prompt adherence, variable-length prompt handling, and task generalization via prompt templates. (2) Efficiency – to improve the efficiency of the unified architecture, we develop a set of optimization techniques that improve semantic learning and fine-grained texture generation during training while incorporating inference-time acceleration strategies without compromising image quality. (3) Transparency – we open-source all training details, code, and models to ensure full reproducibility, aiming to bridge the gap between well-resourced closed-source research teams and independent developers.*

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## Lumina2Text2ImgPipeline

[[autodoc]] Lumina2Text2ImgPipeline
  - all
  - __call__
