<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Mage-Flow

[Mage-Flow](https://github.com/microsoft/Mage) is a 4B parameter text-to-image model from Microsoft's Mage team, based on the NR-MMDiT (Native-Resolution Multimodal DiT) dual-stream architecture. It supports any resolution from 512 to 2048 and arbitrary aspect ratios for high-quality image generation.

The model uses a Qwen3-VL text encoder and a custom MageVAE with 128-channel latents and 16x spatial downsampling.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## MageFlowPipeline

[[autodoc]] MageFlowPipeline
    - all
    - __call__

## MageFlowPipelineOutput

[[autodoc]] pipelines.mage_flow.pipeline_output.MageFlowPipelineOutput
