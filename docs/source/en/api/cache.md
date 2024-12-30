<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching Methods

## Pyramid Attention Broadcast

[Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588) from Xuanlei Zhao, Xiaolong Jin, Kai Wang, Yang You.

Pyramid Attention Broadcast (PAB) is a method that speeds up inference in diffusion models by systematically skipping attention computations between successive inference steps and reusing cached attention states. The attention states are not very different between successive inference steps. The most prominent difference is in the spatial attention blocks, not as much in the temporal attention blocks, and finally the least in the cross attention blocks. Therefore, many cross attention computation blocks can be skipped, followed by the temporal and spatial attention blocks. By combining other techniques like sequence parallelism and classifier-free guidance parallelism, PAB achieves near real-time video generation.

Enable PAB with [`~PyramidAttentionBroadcastConfig`] on any pipeline. For some benchmarks, refer to [this](https://github.com/huggingface/diffusers/pull/9562) pull request.

## PyramidAttentionBroadcastConfig

[[autodoc]] PyramidAttentionBroadcastConfig

[[autodoc]] apply_pyramid_attention_broadcast

[[autodoc]] apply_pyramid_attention_broadcast_on_module
