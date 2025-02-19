<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Quantization

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn't be able to fit into memory, and speeding up inference. Diffusers supports 8-bit and 4-bit quantization with [bitsandbytes](https://huggingface.co/docs/bitsandbytes/en/index).

Quantization techniques that aren't supported in Transformers can be added with the [`DiffusersQuantizer`] class.

<Tip>

Learn how to quantize models in the [Quantization](../quantization/overview) guide.

</Tip>


## BitsAndBytesConfig

[[autodoc]] BitsAndBytesConfig

## GGUFQuantizationConfig

[[autodoc]] GGUFQuantizationConfig
## TorchAoConfig

[[autodoc]] TorchAoConfig

## DiffusersQuantizer

[[autodoc]] quantizers.base.DiffusersQuantizer
