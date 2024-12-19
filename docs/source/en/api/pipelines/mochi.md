<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Mochi

> [!TIP]
> Only a research preview of the model weights is available at the moment.

[Mochi 1](https://huggingface.co/genmo/mochi-1-preview) is a video generation model by Genmo with a strong focus on prompt adherence and motion quality. The model features a 10B parameter Asmmetric Diffusion Transformer (AsymmDiT) architecture, and uses non-square QKV and output projection layers to reduce inference memory requirements. A single T5-XXL model is used to encode prompts.

*Mochi 1 preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence in preliminary evaluation. This model dramatically closes the gap between closed and open video generation systems. The model is released under a permissive Apache 2.0 license.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) to learn more about supported quantization backends (bitsandbytes, torchao, gguf) and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`MochiPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, MochiTransformer3DModel, MochiPipeline
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "genmo/mochi-1-preview",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = MochiTransformer3DModel.from_pretrained(
    "genmo/mochi-1-preview",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

video = pipeline(
  "Close-up of a cats eye, with the galaxy reflected in the cats eye. Ultra high resolution 4k.",
  num_inference_steps=28,
  guidance_scale=3.5
).frames[0]
export_to_video(video, "cat.mp4")
```

## MochiPipeline

[[autodoc]] MochiPipeline
  - all
  - __call__

## MochiPipelineOutput

[[autodoc]] pipelines.mochi.pipeline_output.MochiPipelineOutput
