<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# JoyAI-Image-Edit-Plus

[JoyAI-Image](https://github.com/jd-opensource/JoyAI-Image) is a unified multimodal foundation model for image understanding, text-to-image generation, and instruction-guided image editing. It combines an 8B Multimodal Large Language Model (MLLM) with a 16B Multimodal Diffusion Transformer (MMDiT).

JoyAI-Image-Edit-Plus is a multi-image instruction-guided editing model that accepts **multiple reference images** and a text instruction to generate a new image that combines elements from the references according to the instruction. It supports 1–5 reference images per sample.

| Model | Description | Download |
|:-----:|:-----------:|:--------:|
| JoyAI-Image-Edit-Plus | Multi-image instruction-guided editing with element composition from multiple references | [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers) |

```python
import torch
from PIL import Image
from diffusers import JoyImageEditPlusPipeline

pipeline = JoyImageEditPlusPipeline.from_pretrained(
    "jdopensource/JoyAI-Image-Edit-Plus-Diffusers", torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

images = [
    Image.open("reference_0.png").convert("RGB"),
    Image.open("reference_1.png").convert("RGB"),
]

target_h, target_w = pipeline.vae_image_processor.get_default_height_width(images[-1])

output = pipeline(
    images=images,
    prompt="Combine the person from the second image with the scene from the first image.",
    negative_prompt="low quality, blurry, deformed",
    height=target_h,
    width=target_w,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
output.save("joyimage_edit_plus_output.png")
```

## JoyImageEditPlusPipeline

[[autodoc]] JoyImageEditPlusPipeline
  - all
  - __call__

## JoyImageEditPlusPipelineOutput

[[autodoc]] pipelines.joyimage.pipeline_output.JoyImageEditPlusPipelineOutput
