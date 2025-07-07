<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching methods

Cache methods speedup diffusion transformers by storing and reusing intermediate outputs of specific layers, such as attention and feedforward layers, instead of recalculating them at each inference step.

## CacheMixin

[[autodoc]] CacheMixin

## PyramidAttentionBroadcastConfig

[[autodoc]] PyramidAttentionBroadcastConfig

[[autodoc]] apply_pyramid_attention_broadcast

## FasterCacheConfig

[[autodoc]] FasterCacheConfig

[[autodoc]] apply_faster_cache

### FirstBlockCacheConfig

[[autodoc]] FirstBlockCacheConfig

[[autodoc]] apply_first_block_cache
