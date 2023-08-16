# ControlNet

ControlNet is a type of model for controlling image diffusion models by conditioning the model with an additional input image. There are many types of conditioning inputs (canny edge, user sketching, human pose, depth, and more) to control a diffusion model. This is hugely useful because it affords you greater control over image generation, making it easier to generate specific images without fiddling with text prompts or denoising values as much.

<Tip>

Check out Section 3.5 of the [ControlNet](https://huggingface.co/papers/2302.05543) paper for a list of ControlNet implementations of various conditioning inputs. You can find all available ControlNet conditioned models on [lllyasviel](https://huggingface.co/lllyasviel)'s Hub profile.

</Tip>

A ControlNet model has two sets of weights (or blocks) connected by a zero-convolution layer:

- a *locked copy* which keeps everything a large pretrained diffusion model has learned
- a *trainable copy* which is trained on the additional conditioning input

Since the locked copy preserves the pretrained model, training and implementing a ControlNet on a new conditioning input is as fast as finetuning any other model because you aren't training the model from scratch.

This guide will show you how to use ControlNet for text-to-image, image-to-image, inpainting, and more! There are many types of ControlNet conditioning inputs to choose from, but in this guide we'll only focus on several of them. Feel free to experiment with other conditioning inputs!

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries in Colab
#!pip install transformers accelerate safetensors opencv-contrib-python
```

## Text-to-image

For text-to-image, you normally only pass a text prompt to the model. But with ControlNet you can specify an additional conditioning input. Let's condition the model with a canny image, a white outline of an image on a black background. This way, the ControlNet can use the canny image as a control to guide the model to generate an image with the same outline.

Load an image and use the [opencv-python](https://github.com/opencv/opencv-python) library to extract the canny image:

```py
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vermeer_canny_edged.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">canny image</figcaption>
  </div>
</div>

Load the ControlNet model conditioned on canny edge detection, and pass it to the [`StableDiffusionControlNetPipeline`]. Use the faster [`UniPCMultistepScheduler`] and enable model offloading to speed up inference and reduce memory usage.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

Now pass your prompt and canny image to the pipeline:

```py
output = pipe(
    "spiderman wearing sunglasses", num_inference_steps=20, image=canny_image
).images[0]
```

## Image-to-image

For image-to-image, you would typically pass an initial image and a prompt to the pipeline to generate a new image conditioned on the initial image. With a ControlNet, you can pass an additional conditioning input to steer the model. Let's condition the model with a depth map, an image which contains spatial information. This way, the ControlNet can use the depth map as a control to guide the model to generate an image that preserves the spatial information.

Load an image and use the `depth-estimation` [`~transformers.Pipeline`] from 🤗 Transformers to extract the depth map of an image:

```py
import torch
import numpy as np

from transformers import pipeline
from diffusers.utils import load_image

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(image)

def make_hint(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint

depth_estimator = pipeline("depth-estimation")
control_image = make_hint(img, depth_estimator).unsqueeze(0).half().to("cuda")
```

Load the ControlNet model conditioned on depth maps, and pass it to the [`StableDiffusionControlNetImg2ImgPipeline`]. Use the faster [`UniPCMultistepScheduler`] and enable model offloading to speed up inference and reduce memory usage.

```py
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

Now pass your prompt, initial image, and depth map to the pipeline:

```py
output = pipe(
    "lego batman and robin", image=image, control_image=depth_map,
).images[0]
```

## Inpainting

For inpainting, you need an initial image, a mask image, and a prompt describing what to replace the mask with. For ControlNet models, you can add another control image to condition a model with. Let’s condition the model with a canny image, a white outline of an image on a black background. This way, the ControlNet can use the canny image as a control to guide the model to generate an image with the same outline.

Load an initial image and a mask image:

```py
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

init_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
)
init_image = init_image.resize((512, 512))

mask_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
)
mask_image = mask_image.resize((512, 512))
```

Create a function to prepare the control image from the initial and mask images. This'll create a tensor to mark the pixels in `init_image` as masked if the corresponding pixel in `mask_image` is over a certain threshold.

```py
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1],
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


control_image = make_inpaint_condition(init_image, mask_image)
```

Load the ControlNet model conditioned on inpainting images, and pass it to the [`StableDiffusionControlNetInpaintPipeline`]. Use the faster [`UniPCMultistepScheduler`] and enable model offloading to speed up inference and reduce memory usage.

```py
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

Now pass your prompt, initial image, mask image, and control image to the pipeline:

```py
output = pipe(
    "a handsome man with ray-ban sunglasses",
    num_inference_steps=20,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
).images[0]
```

## Combine conditioning inputs

It is also possible to compose multiple ControlNet's conditioned on different image inputs. To get better results, it is often helpful to:

1. mask conditionings such that they don't overlap (for example, mask the area of a canny image where the pose conditioning is located)
2. experiment with the [`controlnet_conditioning_scale`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale) parameter to determine how much weight to assign to each conditioning input

In this example, you'll combine a canny image and a human pose estimation image to generate a new image.

Prepare the canny image conditioning:

Prepare the human pose estimation conditioning:

Load a list of the ControlNet models that correspond to each conditioning, and pass them to the [`StableDiffusionControlNetPipeline`]. Use the faster [`UniPCMultistepScheduler`] and enable model offloading to speed up inference and reduce memory usage.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = [
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16, use_safetensors=True),
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True),
]

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

Now you can pass your prompt (an optional negative prompt if you're using one), canny image, and pose image to the pipeline:

```py
prompt = "a giant standing in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

images = [openpose_image, canny_image]

image = pipe(
    prompt,
    images,
    num_inference_steps=20,
    negative_prompt=negative_prompt,
    controlnet_conditioning_scale=[1.0, 0.8],
).images[0]
```

## Guess mode

[Guess mode](https://github.com/lllyasviel/ControlNet/discussions/188) does not require supplying a prompt to the ControlNet at all! 

## ControlNet with Stable Diffusion XL

