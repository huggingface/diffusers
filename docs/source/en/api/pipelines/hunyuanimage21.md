<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
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

# HunyuanImage2.1


HunyuanImage-2.1 is a 17B text-to-image model that is capable of generating 2K (2048 x 2048) resolution images

HunyuanImage-2.1 comes in the following variants:

| model type | model id |
|:----------:|:--------:|
| HunyuanImage-2.1 | [hunyuanvideo-community/HunyuanImage-2.1-Diffusers](https://huggingface.co/hunyuanvideo-community/HunyuanImage-2.1-Diffusers) |
| HunyuanImage-2.1-Distilled | [hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers](https://huggingface.co/hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers) |
| HunyuanImage-2.1-Refiner | [hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers](https://huggingface.co/hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers) |

> [!TIP]
> [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

## HunyuanImage-2.1

HunyuanImage-2.1 applies [Adaptive Projected Guidance (APG)](https://huggingface.co/papers/2410.02416) combined with Classifier-Free Guidance (CFG) in the denoising loop. `HunyuanImagePipeline` has a `guider` component (read more about [Guider](../modular_diffusers/guiders.md)) and does not take a `guidance_scale` parameter at runtime. To change guider-related parameters, e.g., `guidance_scale`, you can update the `guider` configuration instead.

```python
import torch
from diffusers import HunyuanImagePipeline

pipe = HunyuanImagePipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
``` 

You can inspect the `guider` object:

```py
>>> pipe.guider
AdaptiveProjectedMixGuidance {
  "_class_name": "AdaptiveProjectedMixGuidance",
  "_diffusers_version": "0.36.0.dev0",
  "adaptive_projected_guidance_momentum": -0.5,
  "adaptive_projected_guidance_rescale": 10.0,
  "adaptive_projected_guidance_scale": 10.0,
  "adaptive_projected_guidance_start_step": 5,
  "enabled": true,
  "eta": 0.0,
  "guidance_rescale": 0.0,
  "guidance_scale": 3.5,
  "start": 0.0,
  "stop": 1.0,
  "use_original_formulation": false
}

State:
  step: None
  num_inference_steps: None
  timestep: None
  count_prepared: 0
  enabled: True
  num_conditions: 2
  momentum_buffer: None
  is_apg_enabled: False
  is_cfg_enabled: True
```

To update the guider with a different configuration, use the `new()` method. For example, to generate an image with `guidance_scale=5.0` while keeping all other default guidance parameters:

```py
import torch
from diffusers import HunyuanImagePipeline

pipe = HunyuanImagePipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

# Update the guider configuration
pipe.guider = pipe.guider.new(guidance_scale=5.0)

prompt = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, "
    "wearing a red knitted scarf and a red beret with the word 'Tencent' on it, holding a paintbrush with a "
    "focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."
)

image = pipe(
    prompt=prompt, 
    num_inference_steps=50, 
    height=2048, 
    width=2048,
).images[0]
image.save("image.png")
```


## HunyuanImage-2.1-Distilled

use `distilled_guidance_scale` with the guidance-distilled checkpoint, 

```py
import torch
from diffusers import HunyuanImagePipeline
pipe = HunyuanImagePipeline.from_pretrained("hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

prompt = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, "
    "wearing a red knitted scarf and a red beret with the word 'Tencent' on it, holding a paintbrush with a "
    "focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."
)

out = pipe(
    prompt,
    num_inference_steps=8,
    distilled_guidance_scale=3.25,
    height=2048,
    width=2048,
    generator=generator,
).images[0]

```


## HunyuanImagePipeline

[[autodoc]] HunyuanImagePipeline
  - all
  - __call__

## HunyuanImageRefinerPipeline

[[autodoc]] HunyuanImageRefinerPipeline
  - all
  - __call__


## HunyuanImagePipelineOutput

[[autodoc]] pipelines.hunyuan_image.pipeline_output.HunyuanImagePipelineOutput