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

prompt = "A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light, with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD."
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