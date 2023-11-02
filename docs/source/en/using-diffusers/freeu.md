<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Improve generation quality with FreeU

[[open-in-colab]]

The UNet is responsible for denoising during the reverse diffusion process, and there are two distinct features in its architecture: 

1. Backbone features primarily contribute to the denoising process
2. Skip features mainly introduce high-frequency features into the decoder module and can make the network overlook the semantics in the backbone features

However, the skip connection can sometimes introduce unnatural image details. [FreeU](https://hf.co/papers/2309.11497) is a technique for improving image quality by rebalancing the contributions from the UNet’s skip connections and backbone feature maps. 

FreeU is applied during inference and it does not require any additional training. The technique works for different tasks such as text-to-image, image-to-image, and text-to-video.

In this guide, you will apply FreeU to the [`StableDiffusionPipeline`], [`StableDiffusionXLPipeline`], and [`TextToVideoSDPipeline`]. You need to install Diffusers from source to run the examples below.

## StableDiffusionPipeline

Load the pipeline: 

```py
from diffusers import DiffusionPipeline
import torch 

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
```

Then enable the FreeU mechanism with the FreeU-specific hyperparameters. These values are scaling factors for the backbone and skip features.

```py
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
```

The values above are from the official FreeU [code repository](https://github.com/ChenyangSi/FreeU) where you can also find [reference hyperparameters](https://github.com/ChenyangSi/FreeU#range-for-more-parameters) for different models.

<Tip>

Disable the FreeU mechanism by calling `disable_freeu()` on a pipeline.

</Tip>

And then run inference:

```py
prompt = "A squirrel eating a burger"
seed = 2023
image = pipeline(prompt, generator=torch.manual_seed(seed)).images[0]
image
```

The figure below compares non-FreeU and FreeU results respectively for the same hyperparameters used above (`prompt` and `seed`):

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/freeu/sdv1_5_freeu.jpg)


Let's see how Stable Diffusion 2 results are impacted:

```py
from diffusers import DiffusionPipeline
import torch 

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, safety_checker=None
).to("cuda")

prompt = "A squirrel eating a burger"
seed = 2023

pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
image = pipeline(prompt, generator=torch.manual_seed(seed)).images[0]
image
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/freeu/sdv2_1_freeu.jpg)

## Stable Diffusion XL

Finally, let's take a look at how FreeU affects Stable Diffusion XL results:

```py
from diffusers import DiffusionPipeline
import torch 

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")

prompt = "A squirrel eating a burger"
seed = 2023

# Comes from
# https://wandb.ai/nasirk24/UNET-FreeU-SDXL/reports/FreeU-SDXL-Optimal-Parameters--Vmlldzo1NDg4NTUw
pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
image = pipeline(prompt, generator=torch.manual_seed(seed)).images[0]
image
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/freeu/sdxl_freeu.jpg)

## Text-to-video generation

FreeU can also be used to improve video quality:

```python
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import torch

model_id = "cerspense/zeroscope_v2_576w"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "an astronaut riding a horse on mars"
seed = 2023

# The values come from
# https://github.com/lyn-rgb/FreeU_Diffusers#video-pipelines
pipe.enable_freeu(b1=1.2, b2=1.4, s1=0.9, s2=0.2)
video_frames = pipe(prompt, height=320, width=576, num_frames=30, generator=torch.manual_seed(seed)).frames
export_to_video(video_frames, "astronaut_rides_horse.mp4")
```

Thanks to [kadirnar](https://github.com/kadirnar/) for helping to integrate the feature, and to [justindujardin](https://github.com/justindujardin) for the helpful discussions.
