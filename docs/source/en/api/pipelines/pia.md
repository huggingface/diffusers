<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

> [!WARNING]
> This pipeline is deprecated but it can still be used. However, we won't test the pipeline anymore and won't accept any changes to it. If you run into any issues, reinstall the last Diffusers version that supported this model.

# Image-to-Video Generation with PIA (Personalized Image Animator)

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

## Overview

[PIA: Your Personalized Image Animator via Plug-and-Play Modules in Text-to-Image Models](https://huggingface.co/papers/2312.13964) by Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, Kai Chen

Recent advancements in personalized text-to-image (T2I) models have revolutionized content creation, empowering non-experts to generate stunning images with unique styles. While promising, adding realistic motions into these personalized images by text poses significant challenges in preserving distinct styles, high-fidelity details, and achieving motion controllability by text. In this paper, we present PIA, a Personalized Image Animator that excels in aligning with condition images, achieving motion controllability by text, and the compatibility with various personalized T2I models without specific tuning. To achieve these goals, PIA builds upon a base T2I model with well-trained temporal alignment layers, allowing for the seamless transformation of any personalized T2I model into an image animation model. A key component of PIA is the introduction of the condition module, which utilizes the condition frame and inter-frame affinity as input to transfer appearance information guided by the affinity hint for individual frame synthesis in the latent space. This design mitigates the challenges of appearance-related image alignment within and allows for a stronger focus on aligning with motion-related guidance.

[Project page](https://pi-animator.github.io/)

## Available Pipelines

| Pipeline | Tasks | Demo
|---|---|:---:|
| [PIAPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pia/pipeline_pia.py) | *Image-to-Video Generation with PIA* |

## Available checkpoints

Motion Adapter checkpoints for PIA can be found under the [OpenMMLab org](https://huggingface.co/openmmlab/PIA-condition-adapter). These checkpoints are meant to work with any model based on Stable Diffusion 1.5

## Usage example

PIA works with a MotionAdapter checkpoint and a Stable Diffusion 1.5 model checkpoint. The MotionAdapter is a collection of Motion Modules that are responsible for adding coherent motion across image frames. These modules are applied after the Resnet and Attention blocks in the Stable Diffusion UNet. In addition to the motion modules, PIA also replaces the input convolution layer of the SD 1.5 UNet model with a 9 channel input convolution layer.

The following example demonstrates how to use PIA to generate a video from a single image.

```python
import torch
from diffusers import (
    EulerDiscreteScheduler,
    MotionAdapter,
    PIAPipeline,
)
from diffusers.utils import export_to_gif, load_image

adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter")
pipe = PIAPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", motion_adapter=adapter, torch_dtype=torch.float16)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
)
image = image.resize((512, 512))
prompt = "cat in a field"
negative_prompt = "wrong white balance, dark, sketches,worst quality,low quality"

generator = torch.Generator("cpu").manual_seed(0)
output = pipe(image=image, prompt=prompt, generator=generator)
frames = output.frames[0]
export_to_gif(frames, "pia-animation.gif")
```

Here are some sample outputs:

<table>
    <tr>
        <td><center>
        cat in a field.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pia-default-output.gif"
            alt="cat in a field"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>


> [!TIP]
> If you plan on using a scheduler that can clip samples, make sure to disable it by setting `clip_sample=False` in the scheduler as this can also have an adverse effect on generated samples. Additionally, the PIA checkpoints can be sensitive to the beta schedule of the scheduler. We recommend setting this to `linear`.

## Using FreeInit

[FreeInit: Bridging Initialization Gap in Video Diffusion Models](https://huggingface.co/papers/2312.07537) by Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, Ziwei Liu.

FreeInit is an effective method that improves temporal consistency and overall quality of videos generated using video-diffusion-models without any addition training. It can be applied to PIA, AnimateDiff, ModelScope, VideoCrafter and various other video generation models seamlessly at inference time, and works by iteratively refining the latent-initialization noise. More details can be found it the paper.

The following example demonstrates the usage of FreeInit.

```python
import torch
from diffusers import (
    DDIMScheduler,
    MotionAdapter,
    PIAPipeline,
)
from diffusers.utils import export_to_gif, load_image

adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter")
pipe = PIAPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", motion_adapter=adapter)

# enable FreeInit
# Refer to the enable_free_init documentation for a full list of configurable parameters
pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

# Memory saving options
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
)
image = image.resize((512, 512))
prompt = "cat in a field"
negative_prompt = "wrong white balance, dark, sketches,worst quality,low quality"

generator = torch.Generator("cpu").manual_seed(0)

output = pipe(image=image, prompt=prompt, generator=generator)
frames = output.frames[0]
export_to_gif(frames, "pia-freeinit-animation.gif")
```

<table>
    <tr>
        <td><center>
        cat in a field.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pia-freeinit-output-cat.gif"
            alt="cat in a field"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>


> [!WARNING]
> FreeInit is not really free - the improved quality comes at the cost of extra computation. It requires sampling a few extra times depending on the `num_iters` parameter that is set when enabling it. Setting the `use_fast_sampling` parameter to `True` can improve the overall performance (at the cost of lower quality compared to when `use_fast_sampling=False` but still better results than vanilla video generation models).

## PIAPipeline

[[autodoc]] PIAPipeline
	- all
	- __call__
    - enable_freeu
    - disable_freeu
    - enable_free_init
    - disable_free_init
    - enable_vae_slicing
    - disable_vae_slicing
    - enable_vae_tiling
    - disable_vae_tiling

## PIAPipelineOutput

[[autodoc]] pipelines.pia.PIAPipelineOutput