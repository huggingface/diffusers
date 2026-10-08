<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# BriaFibo2Transformer2DModel

The single-stream transformer of [Bria](https://huggingface.co/briaai)'s fibo-2, built on the Z-Image block. Its text conditioning comes from Qwen3-VL in two ways: a Perceiver resampler turns the text encoder's hidden states into gist tokens that share the stream with the image tokens, and five blocks add a gated cross-attention to their own bundle of Qwen3-VL layers. To edit, the latents of the images to edit join the stream as clean tokens, each image on its own RoPE plane.

## BriaFibo2Transformer2DModel

[[autodoc]] BriaFibo2Transformer2DModel
