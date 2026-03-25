<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# BlockRefinementScheduler

The `BlockRefinementScheduler` manages block-wise iterative refinement for discrete token diffusion. At each step it
commits the most confident tokens and optionally edits already-committed tokens when the model predicts a different
token with high confidence.

This scheduler is used by [`LLaDA2Pipeline`].

## BlockRefinementScheduler
[[autodoc]] BlockRefinementScheduler

## BlockRefinementSchedulerOutput
[[autodoc]] schedulers.scheduling_block_refinement.BlockRefinementSchedulerOutput
