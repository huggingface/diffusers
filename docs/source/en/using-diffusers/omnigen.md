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


## Text-to-Image


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

prompt = "An elderly gentleman, with a serene expression, sits at the water's edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that's propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist's canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea."
pipe.enable_model_cpu_offload()

image = pipe(
    prompt=prompt,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image
```
<div class="flex justify-center">
    <img src="https://github.com/VectorSpaceLab/OmniGen/blob/main/imgs/demo_cases/t2i_woman_with_book.png" alt="generated image of an astronaut in a jungle"/>
</div>
For text-to-image, pass a text prompt. By default, CogVideoX generates a 720x480 video for the best results.



## Optimization

### inference speed

### Memory 