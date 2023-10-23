<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Audio Diffusion

[Audio Diffusion](https://github.com/teticio/audio-diffusion) is by Robert Dargavel Smith, and it leverages the recent advances in image generation from diffusion models by converting audio samples to and from Mel spectrogram images.

The original codebase, training scripts and example notebooks can be found at [teticio/audio-diffusion](https://github.com/teticio/audio-diffusion).

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## AudioDiffusionPipeline
[[autodoc]] AudioDiffusionPipeline
	- all
	- __call__

## AudioPipelineOutput
[[autodoc]] pipelines.AudioPipelineOutput

## ImagePipelineOutput
[[autodoc]] pipelines.ImagePipelineOutput

## Mel
[[autodoc]] Mel
