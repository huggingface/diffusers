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

# Dance Diffusion

[Dance Diffusion](https://github.com/Harmonai-org/sample-generator) is by Zach Evans.

Dance Diffusion is the first in a suite of generative audio tools for producers and musicians released by [Harmonai](https://github.com/Harmonai-org).


> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## DanceDiffusionPipeline
[[autodoc]] DanceDiffusionPipeline
	- all
	- __call__

## AudioPipelineOutput
[[autodoc]] pipelines.AudioPipelineOutput
