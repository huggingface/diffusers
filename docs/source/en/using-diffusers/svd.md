<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Video Diffusion

[[open-in-colab]]

[Stable Video Diffusion](https://static1.squarespace.com/static/6213c340453c3f502425776e/t/655ce779b9d47d342a93c890/1700587395994/stable_video_diffusion.pdf) is a powerful image-to-video generation model that can generate high resolution (576x1024) 2-4 second videos conditioned on the input image.

This guide will show you how to use SVD to short generate videos from images.

Before you begin, make sure you have the following libraries installed:

```py
!pip install -q -U diffusers transformers accelerate 
```

## Image to Video Generation

The are two variants of SVD. [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) 
and [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt). The svd checkpoint is trained to generate 14 frames and the svd-xt checkpoint is further 
finetuned to generate 25 frames.

We will use the `svd-xt` checkpoint for this guide.

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

<video width="1024" height="576" controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket_generated.mp4?download=true" type="video/mp4">
</video>

<Tip>
Since generating videos is more memory intensive we can use the `decode_chunk_size` argument to control how many frames are decoded at once. This will reduce the memory usage. It's recommended to tweak this value based on your GPU memory.
Setting `decode_chunk_size=1` will decode one frame at a time and will use the least amount of memory but the video might have some flickering.

Additionally, we also use [model cpu offloading](../../optimization/memory#model-offloading) to reduce the memory usage.
</Tip>


### Micro-conditioning

Along with conditioning image Stable Diffusion Video also allows providing micro-conditioning that allows more control over the generated video.
It accepts the following arguments:

- `fps`: The frames per second of the generated video.
- `motion_bucket_id`: The motion bucket id to use for the generated video. This can be used to control the motion of the generated video. Increasing the motion bucket id will increase the motion of the generated video.
- `noise_aug_strength`: The amount of noise added to the conditioning image. The higher the values the less the video will resemble the conditioning image. Increasing this value will also increase the motion of the generated video.

Here is an example of using micro-conditioning to generate a video with more motion.

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```

<video width="1024" height="576" controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket_generated_motion.mp4?download=true" type="video/mp4">
</video>

