<!--Copyright 2024 The HuggingFace Team and AniMemory Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AniMemory-alpha
![img](https://github.com/animEEEmpire/AniMemory-alpha/raw/main/gallery_demo.png)



[Animemory Alpha](https://huggingface.co/animEEEmpire/AniMemory-alpha) is a bilingual model primarily focused on anime-style image generation. It utilizes a SDXL-type Unet
structure and a self-developed bilingual T5-XXL text encoder, achieving good alignment between Chinese and English. We
first developed our general model using billion-level data and then tuned the anime model through a series of
post-training strategies and curated data. By open-sourcing the Alpha version, we hope to contribute to the development
of the anime community, and we greatly value any feedback.

### Usage
```
from diffusers import AniMemoryPipeline
import torch

pipe = AniMemoryPipeline.from_pretrained("animEEEmpire/AniMemory-alpha", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "一只凶恶的狼，猩红的眼神，在午夜咆哮，月光皎洁"
negative_prompt = "nsfw, worst quality, low quality, normal quality, low resolution, monochrome, blurry, wrong, Mutated hands and fingers, text, ugly faces, twisted, jpeg artifacts, watermark, low contrast, realistic"

images = pipe(prompt=prompt,
              negative_prompt=negative_prompt,
              num_inference_steps=40,
              height=1024, width=1024,
              guidance_scale=7.0
              )[0]
images.save("output.png")
```

Use pipe.enable_sequential_cpu_offload() to offload the model into CPU for less GPU memory cost (about 14.25 G, compared to 25.67 G if CPU offload is not enabled), but the inference time will increase significantly(5.18s v.s. 17.74s on A100 40G).

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>



## AniMemoryPipeline

[[autodoc]] AniMemoryPipeline
	- all
	- __call__
