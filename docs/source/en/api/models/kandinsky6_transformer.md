<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 6 Transformers

Kandinsky 6 uses a multimodal diffusion transformer that denoises video and audio latents together for
text/image-to-video-and-audio generation, and a text-free diffusion transformer for video super-resolution.

## Kandinsky6Transformer3DModel

[[autodoc]] Kandinsky6Transformer3DModel
  - all
  - forward

## Kandinsky6Transformer3DModelOutput

[[autodoc]] models.transformers.transformer_kandinsky6.Kandinsky6Transformer3DModelOutput

## Kandinsky6SRTransformer3DModel

[[autodoc]] Kandinsky6SRTransformer3DModel
  - all
  - forward
