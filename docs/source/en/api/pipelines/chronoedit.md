<!-- Copyright 2025 The ChronoEdit Team and HuggingFace Team. All rights reserved.
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
# limitations under the License. -->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference" target="_blank" rel="noopener">
      <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
    </a>
  </div>
</div>

# ChronoEdit

[ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation](https://huggingface.co/papers/2510.04290) from NVIDIA and University of Toronto, by Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, Huan Ling.

> **TL;DR:** ChronoEdit reframes image editing as a video generation task, using input and edited images as start/end frames to leverage pretrained video models with temporal consistency. A temporal reasoning stage introduces reasoning tokens to ensure physically plausible edits and visualize the editing trajectory.

*Recent advances in large generative models have greatly enhanced both image editing and in-context image generation, yet a critical gap remains in ensuring physical consistency, where edited objects must remain coherent. This capability is especially vital for world simulation related tasks. In this paper, we present ChronoEdit, a framework that reframes image editing as a video generation problem. First, ChronoEdit treats the input and edited images as the first and last frames of a video, allowing it to leverage large pretrained video generative models that capture not only object appearance but also the implicit physics of motion and interaction through learned temporal consistency. Second, ChronoEdit introduces a temporal reasoning stage that explicitly performs editing at inference time. Under this setting, target frame is jointly denoised with reasoning tokens to imagine a plausible editing trajectory that constrains the solution space to physically viable transformations. The reasoning tokens are then dropped after a few steps to avoid the high computational cost of rendering a full video. To validate ChronoEdit, we introduce PBench-Edit, a new benchmark of image-prompt pairs for contexts that require physical consistency, and demonstrate that ChronoEdit surpasses state-of-the-art baselines in both visual fidelity and physical plausibility. Project page for code and models: [this https URL](https://research.nvidia.com/labs/toronto-ai/chronoedit).*

The ChronoEdit pipeline is developed by the ChronoEdit Team. The original code is available on [GitHub](https://github.com/nv-tlabs/ChronoEdit), and pretrained models can be found in the [nvidia/ChronoEdit](https://huggingface.co/collections/nvidia/chronoedit) collection on Hugging Face.


### Image Editing

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=50,
    guidance_scale=5.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

Optionally, enable **temporal reasoning** for improved physical consistency:
```py
output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=29,
    num_inference_steps=50,
    guidance_scale=5.0,
    enable_temporal_reasoning=True,
    num_temporal_reasoning_steps=50,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

### Inference with 8-Step Distillation Lora

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
lora_path = hf_hub_download(repo_id=model_id, filename="lora/chronoedit_distill_lora.safetensors")
pipe.load_lora_weights(lora_path)
pipe.fuse_lora(lora_scale=1.0)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=2.0)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=8,
    guidance_scale=1.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

## ChronoEditPipeline

[[autodoc]] ChronoEditPipeline
  - all
  - __call__

## ChronoEditPipelineOutput

[[autodoc]] pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput