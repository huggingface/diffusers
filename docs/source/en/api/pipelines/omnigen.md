<!--Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# OmniGen

[OmniGen: Unified Image Generation](https://huggingface.co/papers/2409.11340) from BAAI, by Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, Zheng Liu.

The abstract from the paper is:

*The emergence of Large Language Models (LLMs) has unified language  generation tasks and revolutionized human-machine interaction.  However, in the realm of image generation, a unified model capable of handling various tasks within a single framework remains largely unexplored. In this work, we introduce OmniGen, a new diffusion model for unified image generation. OmniGen is characterized by the following features: 1) Unification: OmniGen not only demonstrates text-to-image generation capabilities but also inherently supports various downstream tasks, such as image editing, subject-driven generation, and visual conditional generation. 2) Simplicity: The architecture of OmniGen is highly simplified, eliminating the need for additional plugins. Moreover, compared to existing diffusion models, it is more user-friendly and can complete complex tasks end-to-end through instructions without the need for extra intermediate steps, greatly simplifying the image generation workflow. 3) Knowledge Transfer: Benefit from learning in a unified format, OmniGen effectively transfers knowledge across different tasks, manages unseen tasks and domains, and exhibits novel capabilities. We also explore the model’s reasoning capabilities and potential applications of the chain-of-thought mechanism.  This work represents the first attempt at a general-purpose image generation model,  and we will release our resources at https://github.com/VectorSpaceLab/OmniGen to foster future advancements.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [staoxiao](https://github.com/staoxiao). The original codebase can be found [here](https://github.com/VectorSpaceLab/OmniGen). The original weights can be found under [hf.co/shitao](https://huggingface.co/Shitao/OmniGen-v1).

## Inference

First, load the pipeline:

```python
import torch
from diffusers import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1-diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

For text-to-image, pass a text prompt. By default, OmniGen generates a 1024x1024 image. 
You can try setting the `height` and `width` parameters to generate images with different size.

```python
prompt = "Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light, with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD."
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3,
    generator=torch.Generator(device="cpu").manual_seed(111),
).images[0]
image.save("output.png")
```

OmniGen supports multimodal inputs. 
When the input includes an image, you need to add a placeholder `<img><|image_1|></img>` in the text prompt to represent the image. 
It is recommended to enable `use_input_image_size_as_output` to keep the edited image the same size as the original image.

```python
prompt="<img><|image_1|></img> Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/t2i_woman_with_book.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(222)).images[0]
image.save("output.png")
```

## OmniGenPipeline

[[autodoc]] OmniGenPipeline
  - all
  - __call__
