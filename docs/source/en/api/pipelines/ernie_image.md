<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Ernie-Image

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[ERNIE-Image] is a powerful and highly efficient image generation model with 8B parameters. Currently there's only one model with two more to be released:

|Model|Hugging Face|
|---|---|
|ERNIE-Image-Turbo|https://huggingface.co/baidu/ERNIE-Image-Turbo|

## Ernie-Image

ERNIE-Image-Turbo is a distilled version of ERNIE-Image that matches or exceeds leading competitors with only 8 NFEs (Number of Function Evaluations). It offers sub-second inference latency on enterprise-grade H800 GPUs and fits comfortably within 16G VRAM consumer devices. It excels in photorealistic image generation, bilingual text rendering (English & Chinese), and robust instruction adherence.

## ErnieImagePipeline

Use [`ErnieImagePipeline`] to generate an image based on a text prompt. If you do not want to use PE, please set use_pe=False.

```python
import torch
from diffusers import ErnieImagePipeline
from diffusers.utils import load_image

pipe = ErnieImagePipeline.from_pretrained("baidu/ERNIE-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")
# 如果显存不足，可以开启offload
pipe.enable_model_cpu_offload()

prompt = "一只黑白相间的中华田园犬"
images = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=5.0,
    generator=generator,
).images
images[0].save("ernie-image-turbo-output.png")
```