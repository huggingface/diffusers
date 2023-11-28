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

from diffusers import StableDiffusionVideoPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableDiffusionVideoPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, num_frames=25, frames_to_decode_at_once=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

<video width="1024" height="576" controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket_generated.mp4?download=true" type="video/mp4">
</video>

Since generating videos is more memory intensive we can use the `frames_to_decode_at_once` argument to control how many frames are decoded at once. This will reduce the memory usage. It's recommended to tweak this value based on your GPU memory.

Additionally, we also use [model cpu offloading](../../optimization/memory#model-offloading) to reduce the memory usage.

