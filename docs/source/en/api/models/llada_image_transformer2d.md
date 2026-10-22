<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LLaDA-Image

The LLaDA-Image model family combines a denoising transformer with a QueryFormer, text projection model, and SigVQ
image tokenizer. Together, these components support text-to-image generation, VQ-conditioned generation, and
instruction-guided image editing.

The original code and checkpoints are available in the [LLaDA-Image repository](https://github.com/inclusionAI/LLaDA-Image).

## LLaDAImageTransformer2DModel

[[autodoc]] LLaDAImageTransformer2DModel

## LLaDAImageQueryFormerModel

[[autodoc]] LLaDAImageQueryFormerModel

## LLaDAImageTextProjectionModel

[[autodoc]] LLaDAImageTextProjectionModel

## LLaDAImageSigVQModel

[[autodoc]] LLaDAImageSigVQModel
