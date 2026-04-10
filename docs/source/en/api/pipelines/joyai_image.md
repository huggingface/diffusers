<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# JoyAI-Image

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

JoyAI-Image is a multimodal foundation model specialized in instruction-guided image editing. It enables precise and controllable edits by leveraging strong spatial understanding, including scene parsing, relational grounding, and instruction decomposition, allowing complex modifications to be applied accurately to specified regions.


### Key Features
- 🌟 **Unified Multimodal Understanding and Generation**: Combines powerful image understanding with generation capabilities in a single model.
- 🌟 **Spatial Editing**: Supports precise spatial editing including object movement, rotation, and camera control.
- 🌟 **Instruction Following**: Accurately interprets user instructions for image modifications while preserving image quality.
- 🌟 **Qwen2.5-VL Integration**: Leverages Qwen2.5-VL for enhanced multimodal understanding.

For more details, please refer to the [JoyAI-Image GitHub](https://github.com/jd-opensource/JoyAI-Image).


## Usage Example

```py
import torch
from diffusers import JoyAIImagePipeline

pipe = JoyAIImagePipeline.from_pretrained("path/to/converted/checkpoint", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "Move the apple into the red box and finally remove the red box."
image = pipe(
    prompt,
    image=input_image,
    num_inference_steps=30,
    guidance_scale=5.0,
).images[0]
image.save("./output.png")
```


### Supported Prompt Patterns

#### 1. Object Move
```text
Move the <object> into the red box and finally remove the red box.
```

#### 2. Object Rotation
```text
Rotate the <object> to show the <view> side view.
```
Supported views: front, right, left, rear, front right, front left, rear right, rear left

#### 3. Camera Control
```text
Move the camera.
- Camera rotation: Yaw {y_rotation}°, Pitch {p_rotation}°.
- Camera zoom: in/out/unchanged.
- Keep the 3D scene static; only change the viewpoint.
```

This pipeline was contributed by JDopensource Team. The original codebase can be found [here](https://github.com/jd-opensource/JoyAI-Image).


## Available Models
<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Models</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Type</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Description</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Download Link</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">JoyAI&#8209;Image&#8209;Edit</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Image Editing</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Final Release. Specialized model for instruction-guided image editing.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/jdopensource/JoyAI-Image-Edit">Huggingface</a></span>
        </td>
      </tr>
    </tbody>
  </table>
</div>

## Converting Original Checkpoint to Diffusers Format

If you have the original JoyAI checkpoint, you can convert it to diffusers format using the provided conversion script:

```bash
python scripts/convert_joyai_image_to_diffusers.py \
    --source_path /path/to/original/JoyAI-Image-Edit \
    --output_path /path/to/converted/checkpoint \
    --dtype bf16
```

After conversion, load the model with:

```py
from diffusers import JoyAIImagePipeline
pipe = JoyAIImagePipeline.from_pretrained("/path/to/converted/checkpoint")
```


## JoyAIImagePipeline

[[autodoc]] JoyAIImagePipeline
    - all
    - __call__


## JoyAIImagePipelineOutput

[[autodoc]] pipelines.joyai_image.pipeline_output.JoyAIImagePipelineOutput

