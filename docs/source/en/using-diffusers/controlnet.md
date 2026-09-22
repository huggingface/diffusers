<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

[ControlNet](https://huggingface.co/papers/2302.05543) steers a pretrained diffusion model with a structural control image (edges, depth, or pose) while you keep a text prompt. It freezes the base model and adds a parallel network whose residuals guide the denoiser, so you get layout control without fine-tuning.

```text
control image --> ControlNet --> residuals
                                    |
text --> encoder --------------+    |
                               v    v
                         denoiser blocks
                         (frozen base)
```

Load a ControlNet for the control you need (for example canny), then pass it as `controlnet=` to [`~DiffusionPipeline.from_pretrained`]. Use `controlnet_conditioning_scale` to set how strongly the control steers generation.

> [!TIP]
> ControlNets are available to many models such as [Flux](../api/pipelines/controlnet_flux), [Hunyuan-DiT](../api/pipelines/controlnet_hunyuandit), [Stable Diffusion 3](../api/pipelines/controlnet_sd3), and more. The examples in this guide use Flux and Stable Diffusion XL. For T2I-Adapter on older Stable Diffusion–family checkpoints, see [Legacy adapters](./legacy_adapters#t2i-adapter). Prefer ControlNet for new controllable-generation work.

The tabs below show ControlNet for text-to-image, image-to-image, and inpainting.

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

Pass the canny image to the pipeline. Use the `controlnet_conditioning_scale` parameter to determine how much weight to assign to the control.

```py
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny", dtype=torch.bfloat16
)
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"

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

Generate a depth map with a depth estimation pipeline from Transformers.

```py
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image


device = "cuda"  # or "mps", "xpu", "cpu"
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"
).resize((1024, 1024))
depth_image = get_depth_map(image)
```

Pass the depth map to the pipeline. Use the `controlnet_conditioning_scale` parameter to determine how much weight to assign to the control.

```py
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=torch.float16)
pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    dtype=torch.float16,
).to(device)

prompt = """
A photorealistic overhead image of a cat reclining sideways in a flamingo pool floatie holding a margarita. 
The cat is floating leisurely in the pool and completely relaxed and happy.
"""
controlnet_conditioning_scale = 0.5
pipeline(
    prompt,
    image=image,
    control_image=depth_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    strength=0.99,
    num_inference_steps=100,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png" width="300" alt="Generated image (prompt only)"/>
    <figcaption style="text-align: center;">original image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_depth_image.png" width="300" alt="Depth map"/>
    <figcaption style="text-align: center;">depth map</figcaption>
  </figure>
  <figure> 
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_depth_cat.png" width="300" alt="Generated image (ControlNet + prompt)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

</hfoption>
<hfoption id="inpainting">

Generate a mask image that marks which pixels to replace, then pass it as `mask_image` with the control image.

```py
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel

init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"
)
init_image = init_image.resize((1024, 1024))
mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat_mask.png"
)
mask_image = mask_image.resize((1024, 1024))

def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

control_image = make_canny_condition(init_image)
```

Pass the mask and control image to the pipeline. Use the `controlnet_conditioning_scale` parameter to determine how much weight to assign to the control.

```py
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", dtype=torch.float16
)
pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline(
    "a cute and fluffy bunny rabbit",
    num_inference_steps=100,
    strength=0.99,
    controlnet_conditioning_scale=0.5,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png" width="300" alt="Generated image (prompt only)"/>
    <figcaption style="text-align: center;">original image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat_mask.png" width="300" alt="Mask image"/>
    <figcaption style="text-align: center;">mask image</figcaption>
  </figure>
  <figure> 
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_rabbit_inpaint.png" width="300" alt="Generated image (ControlNet + prompt)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

</hfoption>
</hfoptions>

## Multi-ControlNet

Compose several ControlNets (for example canny + depth) by passing them as a list. Mask overlapping regions when you can, and tune each `controlnet_conditioning_scale`. Build `canny_image` with OpenCV Canny (as in text-to-image) and `depth_image` with DPT (as in image-to-image). Resize every control image to the pipeline size before you call the pipeline.

```py
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image

original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"
)
image = np.array(original_image)
canny_image = Image.fromarray(
    np.concatenate([cv2.Canny(image, 100, 200)[:, :, None]] * 3, axis=2)
)

device = "cuda"  # or "mps", "xpu", "cpu"
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

depth_image = get_depth_map(original_image.resize((1024, 1024)))

controlnets = [
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", dtype=torch.float16
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small", dtype=torch.float16
    ),
]

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=torch.float16)
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnets, vae=vae, dtype=torch.float16
).to(device)

prompt = """
a relaxed rabbit sitting on a striped towel next to a pool with a tropical drink nearby, 
bright sunny day, vacation scene, 35mm photograph, film, professional, 4k, highly detailed
"""
negative_prompt = "lowres, bad anatomy, worst quality, low quality, deformed, ugly"

images = [canny_image.resize((1024, 1024)), depth_image.resize((1024, 1024))]

pipeline(
    prompt,
    negative_prompt=negative_prompt,
    image=images,
    num_inference_steps=100,
    controlnet_conditioning_scale=[0.5, 0.5],
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png" width="300" alt="Canny image"/>
    <figcaption style="text-align: center;">canny image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multicontrolnet_depth.png" width="300" alt="Depth map"/>
    <figcaption style="text-align: center;">depth map</figcaption>
  </figure>
  <figure> 
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_multi_controlnet.png" width="300" alt="Generated image (ControlNet + prompt)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

## Guess mode

[Guess mode](https://github.com/lllyasviel/ControlNet/discussions/188) generates an image from a control alone when you don't have (or don't want) a text prompt. It scales ControlNet residuals by block depth, from about `0.1` in early down blocks up to `1.0` at the mid block. That keeps early layers from over-constraining while deeper blocks still carry the structure.

```py
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
  "diffusers/controlnet-canny-sdxl-1.0", dtype=torch.float16
)
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  controlnet=controlnet,
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

canny_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png")
pipeline(
  "",
  image=canny_image,
  guess_mode=True
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png" width="300" alt="Canny image"/>
    <figcaption style="text-align: center;">canny image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guess_mode.png" width="300" alt="Generated image (Guess mode)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>