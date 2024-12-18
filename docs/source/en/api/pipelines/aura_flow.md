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

Refer to the [Quantization](../../quantization/overview) to learn more about supported quantization backends (bitsandbytes, torchao, gguf) and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`AuraFlowPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, AuraFlowTransformer2DModel, AuraFlowPipeline
from diffusers.utils import export_to_video
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

prompt = "A refreshing scene where a glass of freshly squeezed orange juice stands prominently at the center, bathed in warm, golden sunlight that highlights the vibrant, citrus hues of the juice. The glass is intricately detailed, showing condensation droplets that glisten like tiny jewels. Surrounding the base of the glass, scattered orange slices and lush green leaves add a touch of natural beauty and freshness. Above the glass, a dynamic splash of orange juice is captured mid-air, forming the word 'Orange' in a fluid, playful script. The splash is so vivid and realistic that each droplet seems to dance in the air, creating a sense of movement and energy. In the background, a serene orchard with rows of orange trees stretches out under a clear blue sky, their branches heavy with ripe oranges ready for harvest. Rays of sunlight filter through the leaves, casting dappled shadows on the ground. A gentle breeze rustles the leaves, adding a sense of calm and tranquility to the scene. The entire scene evokes a sense of purity, freshness, and vitality, inviting viewers to experience the simple joy of a glass of fresh orange juice."
image = pipeline(prompt).images[0]
image.save("auraflow.png")
```

## AuraFlowPipeline

[[autodoc]] AuraFlowPipeline
	- all
	- __call__