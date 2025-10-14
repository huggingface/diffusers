<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LuminaDiT2DModel

The `LuminaDiT2DModel` is a Diffusion Transformer model for 2D image generation from [Lumina-T2I](https://arxiv.org/abs/2405.05945).

Lumina-T2I is a 5B parameter diffusion transformer that uses LLaMA-2-7B as its text encoder. It implements a rectified flow approach for efficient high-quality image generation. The model uses a DiT-Llama architecture with adaptive layer normalization (adaLN) for conditioning on timesteps and text embeddings.

The abstract from the paper is:

_Sora unveils the potential of scaling Diffusion Transformer for generating photorealistic images and videos at arbitrary resolutions, aspect ratios, and durations, yet it still lacks sufficient implementation details. In this technical report, we introduce the Lumina-T2X family - a series of Flow-based Large Diffusion Transformers (Flag-DiT) equipped with zero-initialized attention, as a unified framework designed to transform noise into images, videos, multi-view 3D objects, and audio clips conditioned on text instructions. By tokenizing the latent spatial-temporal space and incorporating learnable placeholders such as [nextline] and [nextframe] tokens, Lumina-T2X seamlessly unifies the representations of different modalities across various spatial-temporal resolutions._

## LuminaDiT2DModel

[[autodoc]] LuminaDiT2DModel
