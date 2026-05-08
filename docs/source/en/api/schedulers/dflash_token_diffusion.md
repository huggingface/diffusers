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

The scheduler also owns three helpers used by the pipeline's verify loop on hybrid-attention targets:

- `cache_has_linear_attention(cache)` — detect whether a `DynamicCache` contains any linear-attention layers.
- `snapshot_cache(cache)` / `restore_cache(cache, snapshot)` — clone and restore the full per-layer state so a
  partial-accept block can be rolled back and the target re-advanced on just the accepted prefix.

These exist because `DynamicCache.crop()` silently no-ops on linear-attention layers, which would otherwise let
rejected speculative tokens permanently contaminate the recurrent state.

## DFlashTokenDiffusionScheduler
[[autodoc]] DFlashTokenDiffusionScheduler

## DFlashTokenDiffusionSchedulerOutput
[[autodoc]] schedulers.scheduling_dflash_token_diffusion.DFlashTokenDiffusionSchedulerOutput
