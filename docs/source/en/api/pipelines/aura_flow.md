<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AuraFlow

AuraFlow is inspired by [Stable Diffusion 3](../pipelines/stable_diffusion/stable_diffusion_3.md) and is by far the largest text-to-image generation model that comes with an Apache 2.0 license. This model achieves state-of-the-art results on the [GenEval](https://github.com/djghosh13/geneval) benchmark.

It was developed by the Fal team and more details about it can be found in [this blog post](https://blog.fal.ai/auraflow/).

<Tip>

AuraFlow can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details. 

</Tip>

## AuraFlowPipeline

[[autodoc]] AuraFlowPipeline
	- all
	- __call__