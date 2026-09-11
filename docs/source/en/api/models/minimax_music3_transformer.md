<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MiniMaxMusic3Transformer1DModel

The 2.4B flow-matching Diffusion Transformer of [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3). It
denoises 128-channel Flow-VAE audio latents conditioned on the per-frame hidden states of the model's autoregressive
language-model stage, prepending the flow-matching timestep as an extra sequence token (a Stable-Audio-lineage
continuous transformer with partial rotary attention and GLU feedforwards).

## MiniMaxMusic3Transformer1DModel

[[autodoc]] MiniMaxMusic3Transformer1DModel
