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

# QwenImage

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

Qwen-Image from the Qwen team is an image generation foundation model in the Qwen series that achieves significant advances in complex text rendering and precise image editing. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.

Qwen-Image comes in the following variants:

| model type | model id |
|:----------:|:--------:|
| Qwen-Image | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen-Image-Edit | [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| Qwen-Image-Edit Plus | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |

> [!TIP]
> [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

## LoRA for faster inference

Use a LoRA from `lightx2v/Qwen-Image-Lightning` to speed up inference by reducing the
number of steps. Refer to the code snippet below:

<details>
<summary>Code</summary>

```py
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch 
import math

ckpt_id = "Qwen/Qwen-Image"

# From
# https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = DiffusionPipeline.from_pretrained(
    ckpt_id, scheduler=scheduler, torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors"
)

prompt = "a tiny astronaut hatching from an egg on the moon, Ultra HD, 4K, cinematic composition."
negative_prompt = " "
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=8,
    true_cfg_scale=1.0,
    generator=torch.manual_seed(0),
).images[0]
image.save("qwen_fewsteps.png")
```

</details>

> [!TIP]
> The `guidance_scale` parameter in the pipeline is there to support future guidance-distilled models when they come up. Note that passing `guidance_scale` to the pipeline is ineffective. To enable classifier-free guidance, please pass `true_cfg_scale` and `negative_prompt` (even an empty negative prompt like " ") should enable classifier-free guidance computations.

## Multi-image reference with QwenImageEditPlusPipeline

With [`QwenImageEditPlusPipeline`], one can provide multiple images as input reference.

```
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
).to("cuda")

image_1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/grumpy.jpg")
image_2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peng.png")
image = pipe(
    image=[image_1, image_2], 
    prompt='''put the penguin and the cat at a game show called "Qwen Edit Plus Games"''', 
    num_inference_steps=50
).images[0]
```

## QwenImagePipeline

[[autodoc]] QwenImagePipeline
  - all
  - __call__

## QwenImageImg2ImgPipeline

[[autodoc]] QwenImageImg2ImgPipeline
  - all
  - __call__

## QwenImageInpaintPipeline

[[autodoc]] QwenImageInpaintPipeline
  - all
  - __call__

## QwenImageEditPipeline

[[autodoc]] QwenImageEditPipeline
  - all
  - __call__

## QwenImageEditInpaintPipeline

[[autodoc]] QwenImageEditInpaintPipeline
  - all
  - __call__

## QwenImageControlNetPipeline

[[autodoc]] QwenImageControlNetPipeline
  - all
  - __call__

## QwenImageEditPlusPipeline

[[autodoc]] QwenImageEditPlusPipeline
  - all
  - __call__

## QwenImagePipelineOutput

[[autodoc]] pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput