<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LongCat-Image

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>


We introduce LongCat-Image, a pioneering open-source and bilingual (Chinese-English) foundation model for image generation, designed to address core challenges in multilingual text rendering, photorealism, deployment efficiency, and developer accessibility prevalent in current leading models.


### Key Features
- 🌟 **Exceptional Efficiency and Performance**: With only **6B parameters**, LongCat-Image surpasses numerous open-source models that are several times larger across multiple benchmarks, demonstrating the immense potential of efficient model design.
- 🌟 **Superior Editing Performance**: LongCat-Image-Edit model achieves state-of-the-art performance among open-source models, delivering leading instruction-following and image quality with superior visual consistency.
- 🌟 **Powerful Chinese Text Rendering**: LongCat-Image demonstrates superior accuracy and stability in rendering common Chinese characters compared to existing SOTA open-source models and achieves industry-leading coverage of the Chinese dictionary.
- 🌟 **Remarkable Photorealism**: Through an innovative data strategy and training framework, LongCat-Image achieves remarkable photorealism in generated images.
- 🌟 **Comprehensive Open-Source Ecosystem**: We provide a complete toolchain, from intermediate checkpoints to full training code, significantly lowering the barrier for further research and development.

For more details, please refer to the comprehensive [***LongCat-Image Technical Report***](https://arxiv.org/abs/2412.11963)


## Usage Example

```py
import torch
import diffusers
from diffusers import LongCatImagePipeline

weight_dtype = torch.bfloat16
pipe = LongCatImagePipeline.from_pretrained("meituan-longcat/LongCat-Image", torch_dtype=torch.bfloat16 )
pipe.to('cuda')
# pipe.enable_model_cpu_offload()

prompt = '一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上，营造出一种宁静而温馨的氛围。镜头采用中距离视角，突出她的神态和服饰的细节。光线柔和地打在她的脸上，强调她的五官和饰品的质感，增加画面的层次感与亲和力。整个画面构图简洁，砖墙的纹理与阳光的光影效果相得益彰，突显出人物的优雅与从容。'
image = pipe(
    prompt,
    height=768,
    width=1344,
    guidance_scale=4.0,
    num_inference_steps=50,
    num_images_per_prompt=1,
    generator=torch.Generator("cpu").manual_seed(43),
    enable_cfg_renorm=True,
    enable_prompt_rewrite=True,
).images[0]
image.save(f'./longcat_image_t2i_example.png')
```


This pipeline was contributed by LongCat-Image Team. The original codebase can be found [here](https://github.com/meituan-longcat/LongCat-Image).

Available models:
<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Models</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Type</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Description</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Download Link</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">LongCat&#8209;Image</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Text&#8209;to&#8209;Image</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Final Release. The standard model for out&#8209;of&#8209;the&#8209;box inference.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/meituan-longcat/LongCat-Image">Huggingface</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">LongCat&#8209;Image&#8209;Dev</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Text&#8209;to&#8209;Image</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Development. Mid-training checkpoint, suitable for fine-tuning.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/meituan-longcat/LongCat-Image-Dev">Huggingface</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">LongCat&#8209;Image&#8209;Edit</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Image Editing</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Specialized model for image editing.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/meituan-longcat/LongCat-Image-Edit">Huggingface</a></span>
        </td>
      </tr>
    </tbody>
  </table>
</div>

## LongCatImagePipeline

[[autodoc]] LongCatImagePipeline
- all
- __call__

## LongCatImagePipelineOutput

[[autodoc]] pipelines.longcat_image.pipeline_output.LongCatImagePipelineOutput



