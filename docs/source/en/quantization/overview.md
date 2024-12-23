<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Quantization

Quantization techniques focus on representing data with less information while also trying to not lose too much accuracy. This often means converting a data type to represent the same information with fewer bits. For example, if your model weights are stored as 32-bit floating points and they're quantized to 16-bit floating points, this halves the model size which makes it easier to store and reduces memory-usage. Lower precision can also speedup inference because it takes less time to perform calculations with fewer bits.

<Tip>

Interested in adding a new quantization method to Diffusers? Refer to the [Contribute new quantization method guide](https://huggingface.co/docs/transformers/main/en/quantization/contribute) to learn more about adding a new quantization method.

</Tip>

<Tip>

If you are new to the quantization field, we recommend you to check out these beginner-friendly courses about quantization in collaboration with DeepLearning.AI:

* [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

</Tip>

## When to use what?

Diffusers currently supports the following quantization methods.
- [BitsandBytes](./bitsandbytes)
- [TorchAO](./torchao)
- [GGUF](./gguf)

[This resource](https://huggingface.co/docs/transformers/main/en/quantization/overview#when-to-use-what) provides a good overview of the pros and cons of different quantization techniques.
