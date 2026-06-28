<!--Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# StableAudio3DiTModel

A rectified-flow velocity-prediction Diffusion Transformer (DiT) for audio generation, used in
[Stable Audio 3](https://stability.ai/news/stable-audio-3).

Each [`StableAudio3DiTBlock`] performs:

1. **Self-attention** — differential multi-head attention with rotary position embeddings (RoPE).
2. **Cross-attention** — attends to the token sequence from the T5Gemma text encoder.
3. **Feed-forward** — SwiGLU projection.

The model is conditioned on a **timestep** (exponential Fourier features → linear projection) and a **global
conditioning vector** (duration embedding from [`StableAudio3DurationEmbedder`]).

## StableAudio3DiTModel

[[autodoc]] StableAudio3DiTModel
	- all
	- forward

## StableAudio3DiTBlock

[[autodoc]] StableAudio3DiTBlock

## StableAudio3DiTModelOutput

[[autodoc]] StableAudio3DiTModelOutput
