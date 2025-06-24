<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Video Diffusion

[[open-in-colab]]

[Stable Video Diffusion (SVD)](https://huggingface.co/papers/2311.15127) is a powerful image-to-video generation model that can generate 2-4 second high resolution (576x1024) videos conditioned on an input image.

This guide will show you how to use SVD to generate short videos from images.

Before you begin, make sure you have the following libraries installed:

```py
# Colab에서 필요한 라이브러리를 설치하기 위해 주석을 제외하세요
!pip install -q -U diffusers transformers accelerate
```

The are two variants of this model, [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt). The SVD checkpoint is trained to generate 14 frames and the SVD-XT checkpoint is further finetuned to generate 25 frames.

You'll use the SVD-XT checkpoint for this guide.

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"source image of a rocket"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"generated video from source image"</figcaption>
  </div>
</div>

## torch.compile

You can gain a 20-25% speedup at the expense of slightly increased memory by [compiling](../optimization/fp16#torchcompile) the UNet.

```diff
- pipe.enable_model_cpu_offload()
+ pipe.to("cuda")
+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

## Reduce memory usage

Video generation is very memory intensive because you're essentially generating `num_frames` all at once, similar to text-to-image generation with a high batch size. To reduce the memory requirement, there are multiple options that trade-off inference speed for lower memory requirement:

- enable model offloading: each component of the pipeline is offloaded to the CPU once it's not needed anymore.
- enable feed-forward chunking: the feed-forward layer runs in a loop instead of running a single feed-forward with a huge batch size.
- reduce `decode_chunk_size`: the VAE decodes frames in chunks instead of decoding them all together. Setting `decode_chunk_size=1` decodes one frame at a time and uses the least amount of memory (we recommend adjusting this value based on your GPU memory) but the video might have some flickering.

```diff
- pipe.enable_model_cpu_offload()
- frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
+ pipe.enable_model_cpu_offload()
+ pipe.unet.enable_forward_chunking()
+ frames = pipe(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
```

Using all these tricks together should lower the memory requirement to less than 8GB VRAM.

## Micro-conditioning

Stable Diffusion Video also accepts micro-conditioning, in addition to the conditioning image, which allows more control over the generated video:

- `fps`: the frames per second of the generated video.
- `motion_bucket_id`: the motion bucket id to use for the generated video. This can be used to control the motion of the generated video. Increasing the motion bucket id increases the motion of the generated video.
- `noise_aug_strength`: the amount of noise added to the conditioning image. The higher the values the less the video resembles the conditioning image. Increasing this value also increases the motion of the generated video.

For example, to generate a video with more motion, use the `motion_bucket_id` and `noise_aug_strength` micro-conditioning parameters:

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
  "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket_with_conditions.gif)
