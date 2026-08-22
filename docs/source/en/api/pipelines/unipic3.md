<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UniPic-3

[UniPic-3](https://arxiv.org/abs/2601.15664) is a unified framework for **single-image editing** and **multi-image composition** by Skywork, built on the Qwen-Image-Edit architecture. It supports 1–6 input images with arbitrary output resolutions.

| Model | HuggingFace | Inference Steps |
|-------|-------------|-----------------|
| Base Model | [Skywork/Unipic3](https://huggingface.co/Skywork/Unipic3) | 50 steps |
| Consistency Model | [Skywork/Unipic3-Consistency-Model](https://huggingface.co/Skywork/Unipic3-Consistency-Model) | 8 steps |
| DMD Model | [Skywork/Unipic3-DMD](https://huggingface.co/Skywork/Unipic3-DMD) | 8 steps |

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality.

## Usage example

```python
import torch
from PIL import Image
from diffusers import UniPic3Pipeline
from diffusers.utils import load_image

pipe = UniPic3Pipeline.from_pretrained("Skywork/Unipic3", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image1 = load_image("https://example.com/pig.png").convert("RGB")
image2 = load_image("https://example.com/sunglasses.png").convert("RGB")

image = pipe(
    images=[image1, image2],
    prompt="A pig wearing sunglasses.",
    negative_prompt=" ",
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.manual_seed(0),
).images[0]

image.save("unipic3_output.png")
```

## UniPic3Pipeline

[[autodoc]] UniPic3Pipeline
  - all
  - __call__
