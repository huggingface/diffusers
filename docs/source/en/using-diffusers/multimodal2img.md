<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Multi-modal instruction to image

[[open-in-colab]]



Multimodal instructions mean you can input any sequence of mixed text and images to guide image generation. You can input multiple images and use prompts to describe the desired output. This approach is more flexible than using only text or images.

## Examples


Take `OmniGenPipeline` as an example: the input can be a text-image sequence to create new images, he input can be a text-image sequence, with images inserted into the text prompt via special placeholder `<img><|image_i|></img>`.

```py   
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="A man and a woman are sitting at a classroom desk. The man is the man with yellow hair in <img><|image_1|></img>. The woman is the woman on the left of <img><|image_2|></img>"
input_image_1 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/3.jpg")
input_image_2 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/4.jpg")
input_images=[input_image_1, input_image_2]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    height=1024,
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    generator=torch.Generator(device="cpu").manual_seed(666)).images[0]
image
```
<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/3.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">input_image_1</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/4.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">input_image_2</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/id2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>


```py
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")


prompt="A woman is walking down the street, wearing a white long-sleeve blouse with lace details on the sleeves, paired with a blue pleated skirt. The woman is <img><|image_1|></img>. The long-sleeve blouse and a pleated skirt are <img><|image_2|></img>."
input_image_1 = load_image("/share/junjie/code/VISTA2/produce_data/laion_net/diffgpt/OmniGen/docs_img/emma.jpeg")
input_image_2 = load_image("/share/junjie/code/VISTA2/produce_data/laion_net/diffgpt/OmniGen/docs_img/dress.jpg")
input_images=[input_image_1, input_image_2]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    height=1024,
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    generator=torch.Generator(device="cpu").manual_seed(666)).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/emma.jpeg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">person image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/dress.jpg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">clothe image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/tryon.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>


The output image is a [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) object that can be saved:

```py
image.save("generated_image.png")
```