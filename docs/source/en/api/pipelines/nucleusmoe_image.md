<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
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

# NucleusMoE-Image

[NucleusMoE-Image](https://huggingface.co/NucleusAI/NucleusMoE-Image) is a text-to-image model that pairs a single-stream DiT with Mixture-of-Experts feed-forward layers, cross-attention to a Qwen3-VL text encoder, and a flow-matching Euler discrete scheduler.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## NucleusMoEImagePipeline

[[autodoc]] NucleusMoEImagePipeline
  - all
  - __call__

## NucleusMoEImagePipelineOutput

[[autodoc]] pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput
