<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Controlling image quality

The components of a diffusion model, like the UNet and scheduler, can be optimized to improve the quality of generated images leading to better details. These techniques are especially useful if you don't have the resources to simply use a larger model for inference. You can enable these techniques during inference without any additional training.

This guide will show you how to turn these techniques on in your pipeline and how to configure them to improve the quality of your generated images.

## Details

[FreeU](https://hf.co/papers/2309.11497) improves image details by rebalancing the UNet's backbone and skip connection weights. The skip connections can cause the model to overlook some of the backbone semantics which may lead to unnatural image details in the generated image. This technique does not require any additional training and can be applied on the fly during inference for tasks like image-to-image and text-to-video.

Use the [`~pipelines.StableDiffusionMixin.enable_freeu`] method on your pipeline and configure the scaling factors for the backbone (`b1` and `b2`) and skip connections (`s1` and `s2`). The number after each scaling factor corresponds to the stage in the UNet where the factor is applied. Take a look at the [FreeU](https://github.com/ChenyangSi/FreeU#parameters) repository for reference hyperparameters for different models.

<hfoptions id="freeu">
<hfoption id="Stable Diffusion v1-5">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
generator = torch.Generator(device="cpu").manual_seed(33)
prompt = ""
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv15-no-freeu.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv15-freeu.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="Stable Diffusion v2-1">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.4, b2=1.6)
generator = torch.Generator(device="cpu").manual_seed(80)
prompt = "A squirrel eating a burger"
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv21-no-freeu.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv21-freeu.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="Stable Diffusion XL">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
generator = torch.Generator(device="cpu").manual_seed(13)
prompt = "A squirrel eating a burger"
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-no-freeu.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-freeu.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="Zeroscope">

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipeline = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16
).to("cuda")
# values come from https://github.com/lyn-rgb/FreeU_Diffusers#video-pipelines
pipeline.enable_freeu(b1=1.2, b2=1.4, s1=0.9, s2=0.2)
prompt = "Confident teddy bear surfer rides the wave in the tropics"
generator = torch.Generator(device="cpu").manual_seed(47)
video_frames = pipeline(prompt, generator=generator).frames[0]
export_to_video(video_frames, "teddy_bear.mp4", fps=10)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/video-no-freeu.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/video-freeu.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
  </div>
</div>

</hfoption>
</hfoptions>

Call the [`pipelines.StableDiffusionMixin.disable_freeu`] method to disable FreeU.

```py
pipeline.disable_freeu()
```
