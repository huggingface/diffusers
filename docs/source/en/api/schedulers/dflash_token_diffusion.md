<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DFlashTokenDiffusionScheduler

[`DFlashTokenDiffusionScheduler`] implements the verification step for DFlash-style block-diffusion speculative
decoding. It samples a posterior block from the target logits, computes the acceptance length as the longest prefix
where the draft proposal matches the posterior, and exposes the resampled `next_token` for the first rejected
position. Used by [`DFlashPipeline`].

## DFlashTokenDiffusionScheduler
[[autodoc]] DFlashTokenDiffusionScheduler

## DFlashTokenDiffusionSchedulerOutput
[[autodoc]] schedulers.scheduling_dflash_token_diffusion.DFlashTokenDiffusionSchedulerOutput
