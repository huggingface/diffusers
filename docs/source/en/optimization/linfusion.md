<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LinFusion
[LinFusion](https://huggingface.co/papers/2409.02097) accelerates [`DiffusionPipeline`] by replacing all the self-attention layers in a diffusion UNet with distilled Generalized Linear Attention layers. The distilled model is linear-complexity and highly compatible with existing diffusion plugins like ControlNet, IP-Adapter, LoRA, etc. The acceleration can be dramatic at high resolution. Strategical pipelines for high-resolution generation can be found in the [original codebase](https://github.com/Huage001/LinFusion).

You can use it with only 1 additional line:

```diff
import torch
from diffusers import StableDiffusionPipeline

repo_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16, variant="fp16").to("cuda")

+ pipe.load_linfusion(pipeline_name_or_path=repo_id)

image = pipe("a photo of an astronaut on a moon").images[0]
```

Currently, [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [`stabilityai/stable-diffusion-2-1`](https://huggingface.co/stabilityai/stable-diffusion-2-1), [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), models finetuned from them, and pipelines based on them are supported. If the `repo_id` is different from them, e.g., when using a fine-tuned model from the community, you need to specify `pipeline_name_or_path` explicitly to the model it is based on. Otherwise, this argument is optional and LinFusion will read it from the current pipeline. Alternatively, you can also specify the argument `pretrained_model_name_or_path_or_dict` to load LinFusion from other sources. You can also unload it with `pipe.unload_linfusion()` when unnecessary.

For a specific example:

```python
import time
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

repo_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16, variant="fp16").to("cuda")

# warming up
pipe("")

# load linfusion
pipe.load_linfusion()
# inference w linfusion
start_time = time.time()
image_linfusion = pipe(
    "A city street at night with glowing neon signs and rain-soaked pavement.",
    negative_prompt="over-exposure, under-exposure, saturated, duplicate, "
                    "out of frame, lowres, cropped, worst quality, low quality, "
                    "jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, "
                    "bad proportions, deformed, blurry, duplicate",
    height=768,
    width=3072,
    generator=torch.manual_seed(42)
).images[0]
time_used_linfusion = time.time() - start_time
print(f'Time used {time_used_linfusion}')

# unload linfusion
pipe.unload_linfusion()
# inference wo linfusion
start_time = time.time()
image_base = pipe(
    "A city street at night with glowing neon signs and rain-soaked pavement.",
    negative_prompt="over-exposure, under-exposure, saturated, duplicate, "
                    "out of frame, lowres, cropped, worst quality, low quality, "
                    "jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, "
                    "bad proportions, deformed, blurry, duplicate",
    height=768,
    width=3072,
    generator=torch.manual_seed(42)
).images[0]
time_used_base = time.time() - start_time
print(f'Time used {time_used_base}')

# visualization
plt.figure(dpi=500)
plt.subplot(2, 1, 1)
plt.imshow(image_linfusion)
plt.axis('off')
plt.title(f'with LinFusion, Latency {time_used_linfusion}')
plt.subplot(2, 1, 2)
plt.imshow(image_base)
plt.axis('off')
plt.title(f'without LinFusion, Latency {time_used_base}')
plt.subplots_adjust(hspace=-0.3)
plt.tight_layout()
plt.savefig('diffusers_linfusion_example.jpg')
plt.show()
```

You are expected to get the following results:

<div class="flex justify-center">
    <img src="https://github.com/Huage001/LinFusion/blob/main/assets/diffusers_linfusion_example.jpg">
</div>

The latency is measured on an A100 GPU with torch 2.4.0
