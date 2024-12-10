<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
# OmniGen

OmniGen is an image generation model. Unlike existing text-to-image models, OmniGen is designed to handle a variety of tasks (e.g., text-to-image, image editing, controllable generation) within a single model. It has the following features:
- Minimalist model architecture, consisting of only a VAE and a transformer module, for joint modeling of text and images.
- Support for multimodal inputs. It can process any text-image mixed data as instructions for image generation, rather than relying solely on text.

For more information, please refer to the [paper](https://arxiv.org/pdf/2409.11340).
This guide will walk you through using OmniGen for various tasks and use cases.

## Load model checkpoints
Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [`~DiffusionPipeline.from_pretrained`] method.

```py
import torch
from diffusers import OmniGenPipeline
pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
```



## Text-to-image

For text-to-image, pass a text prompt. By default, OmniGen generates a 1024x1024 image. 
You can try setting the `height` and `width` parameters to generate images with different size.

```py
import torch
from diffusers import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt = "Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light, with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD."
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3,
    generator=torch.Generator(device="cpu").manual_seed(111),
).images[0]
image
```
<div class="flex justify-center">
    <img src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/t2i_woman_with_book.png" alt="generated image"/>
</div>

## Image edit

OmniGen supports for multimodal inputs. 
When the input includes an image, you need to add a placeholder `<img><|image_1|></img>` in the text prompt to represent the image. 
It is recommended to enable 'use_input_image_size_as_output' to keep the edited image the same size as the original image.

```py
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="<img><|image_1|></img> Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/t2i_woman_with_book.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(222)).images[0]
image
```
<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/t2i_woman_with_book.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">edited image</figcaption>
  </div>
</div>

OmniGen has some interesting features, such as the ability to infer user needs, as shown in the example below.
```py
prompt="If the woman is thirsty, what should she take? Find it in the image and highlight it in blue. <img><|image_1|></img>"
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(0)).images[0]
image
```
<div class="flex justify-center">
    <img src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/reasoning.png" alt="generated image"/>
</div>


## Controllable generation

 OmniGen can handle several classic computer vision tasks. 
 As shown below, OmniGen can detect human skeletons in input images, which can be used as control conditions to generate new images.

```py
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="Detect the skeleton of human in this image: <img><|image_1|></img>"
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image1 = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(333)).images[0]
image1

prompt="Generate a new photo using the following picture and text as conditions: <img><|image_1|></img>\n A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/skeletal.png")]
image2 = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(333)).images[0]
image2
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/skeletal.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">detected skeleton</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/skeletal2img.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">skeleton to image</figcaption>
  </div>
</div>


OmniGen can also directly use relevant information from input images to generate new images.
```py
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="Following the pose of this image <img><|image_1|></img>, generate a new photo: A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(0)).images[0]
image
```
<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/same_pose.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>


## ID and object preserving

OmniGen can generate multiple images based on the people and objects in the input image and supports inputting multiple images simultaneously. 
Additionally, OmniGen can extract desired objects from an image containing multiple objects based on instructions.

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
input_image_1 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/3.png")
input_image_2 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/4.png")
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
input_image_1 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/emma.jpeg")
input_image_2 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/dress.jpg")
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


## Optimization when inputting multiple images 

For text-to-image task, OmniGen requires minimal memory and time costs (9GB memory and 31s for a 1024*1024 image on A800 GPU). 
However, when using input images, the computational cost increases. 

Here are some guidelines to help you reduce computational costs when input multiple images. The experiments are conducted on A800 GPU and input two images to OmniGen.


### inference speed

- `use_kv_cache=True`:   
   `use_kv_cache` will store key and value states of the input conditions to compute attention without redundant computations. 
    The default value is True, and OmniGen will offload the kv cache to cpu default.
  - `use_kv_cache=False`: the inference time is 3m21s.
  - `use_kv_cache=True`: the inference time is 1m30s.

- `max_input_image_size`:   
  the maximum size of input image, which will be used to crop the input image
  - `max_input_image_size=1024`: the inference time is 1m30s.
  - `max_input_image_size=512`: the inference time is 58s.

### Memory 

- `pipe.enable_model_cpu_offload()`:
  - Without enabling cpu offloading, memory usage is `31 GB`
  - With enabling cpu offloading, memory usage is `28 GB`

- `offload_transformer_block=True`:
  - offload transformer block to reduce memory usage
  - When enabled, memory usage is under `25 GB`

- `pipe.enable_sequential_cpu_offload()`:
  - significantly reduce memory usage at the cost of slow inference
  - When enabled, memory usage is under `11 GB`

