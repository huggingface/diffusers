<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# JoyAI-Image-Edit

[JoyAI-Image](https://github.com/jd-opensource/JoyAI-Image) is a unified multimodal foundation model for image understanding, text-to-image generation, and instruction-guided image editing. It combines an 8B Multimodal Large Language Model (MLLM) with a 16B Multimodal Diffusion Transformer (MMDiT). A central principle of JoyAI-Image is the closed-loop collaboration between understanding, generation, and editing.

JoyAI-Image-Edit supports general image editing as well as spatial editing capabilities including object move, object rotation, and camera control.

| Model | Description | Download |
|:-----:|:-----------:|:--------:|
| JoyAI-Image-Edit | Instruction-guided image editing with precise and controllable spatial manipulation | [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers) |

```python
import torch
from diffusers import JoyImageEditPipeline
from diffusers.utils import load_image

pipeline = JoyImageEditPipeline.from_pretrained(
    "jdopensource/JoyAI-Image-Edit-Diffusers", torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")
prompt = "Add wings to the astronaut."

output = pipeline(
    image=image,
    prompt=prompt,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
output.save("joyimage_edit_output.png")
```

## Spatial editing

JoyAI-Image supports three spatial editing prompt patterns: **Object Move**, **Object Rotation**, and **Camera Control**. For best results, follow the prompt templates below as closely as possible. For more information, refer to [SpatialEdit](https://github.com/EasonXiao-888/SpatialEdit).

### Object Move

Move a target object into a specified region marked by a red box in the input image.

```text
Move the <object> into the red box and finally remove the red box.
```

### Object Rotation

Rotate an object to a specific canonical view. Supported `<view>` values: `front`, `right`, `left`, `rear`, `front right`, `front left`, `rear right`, `rear left`.

```text
Rotate the <object> to show the <view> side view.
```

### Camera Control

Change the camera viewpoint while keeping the 3D scene unchanged.

```text
Move the camera.
- Camera rotation: Yaw {y_rotation}°, Pitch {p_rotation}°.
- Camera zoom: in/out/unchanged.
- Keep the 3D scene static; only change the viewpoint.
```

## JoyImageEditPipeline

[[autodoc]] JoyImageEditPipeline
  - all
  - __call__

## JoyImageEditPipelineOutput

[[autodoc]] pipelines.joyimage.pipeline_output.JoyImageEditPipelineOutput
