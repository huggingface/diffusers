<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

[ControlNet](https://huggingface.co/papers/2302.05543) is an adapter that enables controllable generation such as generating an image of a cat in a *specific pose* or following the lines in a sketch of a *specific* cat. It works by adding a smaller network of "zero convolution" layers and progressively training these to avoid disrupting with the original model. The original model parameters are frozen to avoid retraining it.

A ControlNet is conditioned on extra visual information or "controls" (canny edge, depth maps, human pose, etc.) that can be combined with text prompts to generate images that are guided by the visual input.

Load a ControlNet conditioned on a specific control, such as canny edge, and pass it to the pipeline in [`~DiffusionPipeline.from_pretrained`].

<hfoptions id="usage">
<hfoption id="text-to-image">

Generate a canny image with [opencv-python](https://github.com/opencv/opencv-python).

```py
import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image

original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
```

Pass the canny image to the pipeline.

```py
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny", torch_dtype=torch.bfloat16
)
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16
).to("cuda")

prompt = """
A photorealistic overhead image of a cat reclining sideways in a flamingo pool floatie holding a margarita. 
The cat is floating leisurely in the pool and completely relaxed and happy.
"""

pipeline(
    prompt, 
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=50, 
    guidance_scale=3.5,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png" width="300" alt="Generated image (prompt only)"/>
    <figcaption style="text-align: center;">original image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png" width="300" alt="Control image (Canny edges)"/>
    <figcaption style="text-align: center;">canny image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat-generated.png" width="300" alt="Generated image (ControlNet + prompt)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>


</hfoption>
<hfoption id="image-to-image">

```py

```

</hfoption>
<hfoption id="inpainting">


</hfoption>
</hfoptions>

## guess_mode

[Guess mode](https://github.com/lllyasviel/ControlNet/discussions/188) generates an image from **only** the control input (canny edge, depth map, pose, etc.) and without guidance from a prompt. It adjusts the scale of the ControlNet's output residuals by a fixed ratio depending on block depth. The earlier `DownBlock` is only scaled by `0.1` and the `MidBlock` is fully scaled by `1.0`.

```py
import torch
from diffusers.utils import load_iamge
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
  "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  controlnet=controlnet,
  torch_dtype=torch.float16
).to("cuda")

canny_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png")
pipeline(
  "",
  image=canny_image,
  guess_mode=True
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png" width="300" alt="Control image (Canny edges)"/>
    <figcaption style="text-align: center;">canny image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guess_mode.png" width="300" alt="Generated image (Guess mode)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>