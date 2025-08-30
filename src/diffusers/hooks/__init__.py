# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# limitations under the License.

from ..utils import is_torch_available


if is_torch_available():
    from .context_parallel import apply_context_parallel
    from .faster_cache import FasterCacheConfig, apply_faster_cache
    from .first_block_cache import FirstBlockCacheConfig, apply_first_block_cache
    from .group_offloading import apply_group_offloading
    from .hooks import HookRegistry, ModelHook
    from .layer_skip import LayerSkipConfig, apply_layer_skip
    from .layerwise_casting import apply_layerwise_casting, apply_layerwise_casting_hook
    from .pyramid_attention_broadcast import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
    from .smoothed_energy_guidance_utils import SmoothedEnergyGuidanceConfig
