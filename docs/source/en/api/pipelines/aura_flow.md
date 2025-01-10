<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AuraFlow

AuraFlow is inspired by [Stable Diffusion 3](../pipelines/stable_diffusion/stable_diffusion_3) and is by far the largest text-to-image generation model that comes with an Apache 2.0 license. This model achieves state-of-the-art results on the [GenEval](https://github.com/djghosh13/geneval) benchmark.

It was developed by the Fal team and more details about it can be found in [this blog post](https://blog.fal.ai/auraflow/).

<Tip>

AuraFlow can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details.

</Tip>

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`AuraFlowPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, AuraFlowTransformer2DModel, AuraFlowPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "fal/AuraFlow",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = AuraFlowTransformer2DModel.from_pretrained(
    "fal/AuraFlow",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = AuraFlowPipeline.from_pretrained(
    "fal/AuraFlow",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt).images[0]
image.save("auraflow.png")
```

Loading [GGUF checkpoints](https://huggingface.co/docs/diffusers/quantization/gguf) are also supported:

```py
import torch
from diffusers import (
    AuraFlowPipeline,
    GGUFQuantizationConfig,
    AuraFlowTransformer2DModel,
)

transformer = AuraFlowTransformer2DModel.from_single_file(
    "https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/aura_flow_0.3-Q2_K.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipeline = AuraFlowPipeline.from_pretrained(
    "fal/AuraFlow-v0.3",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

prompt = "a cute pony in a field of flowers"
image = pipeline(prompt).images[0]
image.save("auraflow.png")
```

## AuraFlowPipeline

[[autodoc]] AuraFlowPipeline
	- all
	- __call__