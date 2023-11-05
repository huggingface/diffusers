<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Overview

Generating high-quality outputs is computationally intensive, especially during each iterative step where you go from a noisy output to a less noisy output. One of 🤗 Diffuser's goals is to make this technology widely accessible to everyone, which includes enabling fast inference on consumer and specialized hardware.

This section will cover tips and tricks - like half-precision weights and sliced attention - for optimizing inference speed and reducing memory-consumption. You'll also learn how to speed up your PyTorch code with [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) or [ONNX Runtime](https://onnxruntime.ai/docs/), and enable memory-efficient attention with [xFormers](https://facebookresearch.github.io/xformers/). There are also guides for running inference on specific hardware like Apple Silicon, and Intel or Habana processors.
