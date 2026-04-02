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

[Ernie-Image](https://huggingface.co/papers/2511.22699) is a powerful and highly efficient image generation model with 6B parameters. Currently there's only one model with two more to be released:

|Model|Hugging Face|
|---|---|
|Ernie-Image|https://huggingface.co/Tongyi-MAI/Ernie-Image-Turbo|

## Ernie-Image

Ernie-Image-Turbo is a distilled version of Ernie-Image that matches or exceeds leading competitors with only 8 NFEs (Number of Function Evaluations). It offers sub-second inference latency on enterprise-grade H800 GPUs and fits comfortably within 16G VRAM consumer devices. It excels in photorealistic image generation, bilingual text rendering (English & Chinese), and robust instruction adherence.

## ZImagePipeline

Use [`ZImageImg2ImgPipeline`] to transform an existing image based on a text prompt.

```python
import torch
from diffusers import ErnieImagePipeline
from diffusers.utils import load_image

pipe = ErnieImagePipeline.from_pretrained("Tongyi-MAI/Ernie-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "一只黑白相间的中华田园犬"
images = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=generator,
).images
images[0].save("ernie-image-output.png")
```

## ZImagePipeline

[[autodoc]] ErnieImagePipeline
	- all
	- __call__
