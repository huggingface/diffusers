<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AceStepTransformer1DModel

A 1D Diffusion Transformer for music generation from [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). The model operates on the 25 Hz stereo latents produced by [`AutoencoderOobleck`] using flow matching, and is trained with a Qwen3-derived backbone (grouped-query attention, rotary position embedding, RMSNorm, AdaLN-Zero timestep conditioning) plus cross-attention to the text / lyric / timbre conditions built by `AceStepConditionEncoder`.

## AceStepTransformer1DModel

[[autodoc]] AceStepTransformer1DModel
