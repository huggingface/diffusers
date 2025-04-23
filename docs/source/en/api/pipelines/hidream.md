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

# HiDreamImage

[HiDream-I1](https://huggingface.co/HiDream-ai) by HiDream.ai

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## Available models

The following models are available for the [`HiDreamImagePipeline`](text-to-image) pipeline:

| Model name | Description |
|:---|:---|
| [`HiDream-ai/HiDream-I1-Full`](https://huggingface.co/HiDream-ai/HiDream-I1-Full) | - |
| [`HiDream-ai/HiDream-I1-Dev`](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) | - |
| [`HiDream-ai/HiDream-I1-Fast`](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) | - |

## HiDreamImagePipeline

[[autodoc]] HiDreamImagePipeline
  - all
  - __call__

## HiDreamImagePipelineOutput

[[autodoc]] pipelines.hidream_image.pipeline_output.HiDreamImagePipelineOutput
