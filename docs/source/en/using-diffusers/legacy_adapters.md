<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Legacy adapters

These methods are still supported, especially on Stable Diffusion–family checkpoints. For new work, prefer [LoRA](../tutorials/using_peft_for_inference), [IP-Adapter](./ip_adapter), and [ControlNet](./controlnet).

## T2I-Adapter

[T2I-Adapter](https://huggingface.co/papers/2302.08453) is an adapter that enables controllable generation like [ControlNet](./controlnet). A T2I-Adapter works by learning a *mapping* between a control signal (for example, a depth map) and a pretrained model's internal knowledge. The adapter is plugged into the base model to provide extra guidance based on the control signal during generation.

For new controllable-generation work, prefer [ControlNet](./controlnet).

Load a T2I-Adapter conditioned on a specific control, such as canny edge. Then load the pipeline with [`~DiffusionPipeline.from_pretrained`] and pass the adapter with `adapter=`.

```py
import torch
from diffusers import T2IAdapter, StableDiffusionXLAdapterPipeline, AutoencoderKL

t2i_adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-canny-sdxl-1.0",
    dtype=torch.float16,
)
```

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

Pass the canny image to the pipeline to generate an image.

```py
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=torch.float16)
pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    adapter=t2i_adapter,
    vae=vae,
    dtype=torch.float16,
).to("cuda")  # or "mps", "xpu", "cpu"

prompt = """
A photorealistic overhead image of a cat reclining sideways in a flamingo pool floatie holding a margarita. 
The cat is floating leisurely in the pool and completely relaxed and happy.
"""

pipeline(
    prompt, 
    image=canny_image,
    num_inference_steps=100, 
    guidance_scale=10,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png" width="300" alt="Original image"/>
    <figcaption style="text-align: center;">original image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png" width="300" alt="Control image (Canny edges)"/>
    <figcaption style="text-align: center;">canny image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2i-canny-cat-generated.png" width="300" alt="Generated image (T2I-Adapter + prompt)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

### MultiAdapter

You can compose multiple controls, such as canny image and a depth map, with the [`MultiAdapter`] class.

The example below composes a canny image and depth map.

Load the control images and T2I-Adapters as a list.

```py
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionXLAdapterPipeline, AutoencoderKL, MultiAdapter, T2IAdapter

canny_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png"
)
depth_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_depth_image.png"
)
controls = [canny_image, depth_image]
prompt = ["""
a relaxed rabbit sitting on a striped towel next to a pool with a tropical drink nearby, 
bright sunny day, vacation scene, 35mm photograph, film, professional, 4k, highly detailed
"""]

adapters = MultiAdapter(
    [
        T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", dtype=torch.float16),
        T2IAdapter.from_pretrained("TencentARC/t2i-adapter-depth-midas-sdxl-1.0", dtype=torch.float16),
    ]
)
```

Pass the adapters, prompt, and control images to [`StableDiffusionXLAdapterPipeline`]. Use the `adapter_conditioning_scale` parameter to determine how much weight to assign to each control.

```py
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=torch.float16)
pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    vae=vae,
    adapter=adapters,
).to("cuda")  # or "mps", "xpu", "cpu"

pipeline(
    prompt,
    image=controls,
    height=1024,
    width=1024,
    adapter_conditioning_scale=[0.7, 0.7]
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png" width="300" alt="Control image (Canny edges)"/>
    <figcaption style="text-align: center;">canny image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_depth_image.png" width="300" alt="Control image (depth map)"/>
    <figcaption style="text-align: center;">depth map</figcaption>
  </figure>
  <figure> 
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2i-multi-rabbit.png" width="300" alt="Generated image (MultiAdapter + prompt)"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

## DreamBooth

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method for generating personalized images of a specific instance. It works by fine-tuning the model on 3-5 images of the subject (for example, a cat) that is associated with a unique identifier (`sks cat`). This allows you to use `sks cat` in your prompt to trigger the model to generate images of your cat in different settings, lighting, poses, and styles.

DreamBooth checkpoints are typically a few GBs in size because they contain the full model weights.

To train one, see [Train DreamBooth](../training/dreambooth).

Load the DreamBooth checkpoint with [`~DiffusionPipeline.from_pretrained`] and include the unique identifier in the prompt to activate its generation.

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "sd-dreambooth-library/herge-style",
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
prompt = "A cute sks herge_style brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
pipeline(prompt).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_dreambooth.png" />
</div>

## Textual inversion

[Textual Inversion](https://huggingface.co/papers/2208.01618) is a method for generating personalized images of a concept. It works by fine-tuning a model's word embeddings on 3-5 images of the concept (for example, pixel art) that is associated with a unique token (`<sks>`). This allows you to use the `<sks>` token in your prompt to trigger the model to generate pixel art images.

Textual Inversion weights are very lightweight and typically only a few KBs because they're only word embeddings. However, this also means the word embeddings need to be loaded after loading a model with [`~DiffusionPipeline.from_pretrained`].

To train embeddings, see [Train textual inversion](../training/text_inversion).

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
```

Load the word embeddings with [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] and include the unique token in the prompt to activate its generation.

```py
pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, <gta5-artwork> style"
pipeline(prompt).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_txt_embed.png" />
</div>

Textual Inversion can also be trained to learn *negative embeddings* to steer generation away from unwanted characteristics such as "blurry" or "ugly". It is useful for improving image quality.

EasyNegative is a widely used negative embedding that contains multiple learned negative concepts. Load the negative embeddings and specify the file name and token associated with the negative embeddings. Pass the token to `negative_prompt` in your pipeline to activate it.

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_textual_inversion(
    "EvilEngine/easynegative",
    weight_name="easynegative.safetensors",
    token="easynegative"
)
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
negative_prompt = "easynegative"
pipeline(prompt, negative_prompt=negative_prompt).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png" />
</div>
