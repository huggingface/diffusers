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

# GLM-Image

## Overview

GLM-Image is an image generation model adopts a hybrid autoregressive + diffusion decoder architecture, effectively pushing the upper bound of visual fidelity and fine-grained details. In general image generation quality, it aligns with industry-standard LDM-based approaches, while demonstrating significant advantages in knowledge-intensive image generation scenarios.

Model architecture: a hybrid autoregressive + diffusion decoder design、

+ Autoregressive generator: a 9B-parameter model initialized from [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414), with an expanded vocabulary to incorporate visual tokens. The model first generates a compact encoding of approximately 256 tokens, then expands to 1K–4K tokens, corresponding to 1K–2K high-resolution image outputs. You can check AR model in class `GlmImageForConditionalGeneration` of transformers library.
+ Diffusion Decoder: a 7B-parameter decoder based on a single-stream DiT architecture for latent-space image decoding. It is equipped with a Glyph Encoder text module, significantly improving accurate text rendering within images.

Post-training with decoupled reinforcement learning: the model introduces a fine-grained, modular feedback strategy using the GRPO algorithm, substantially enhancing both semantic understanding and visual detail quality.

+ Autoregressive module: provides low-frequency feedback signals focused on aesthetics and semantic alignment, improving instruction following and artistic expressiveness.
+ Decoder module: delivers high-frequency feedback targeting detail fidelity and text accuracy, resulting in highly realistic textures, lighting, and color reproduction, as well as more precise text rendering.

GLM-Image supports both text-to-image and image-to-image generation within a single model

+ Text-to-image: generates high-detail images from textual descriptions, with particularly strong performance in information-dense scenarios.
+ Image-to-image: supports a wide range of tasks, including image editing, style transfer, multi-subject consistency, and identity-preserving generation for people and objects.

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The codebase can be found [here](https://huggingface.co/zai-org/GLM-Image).

## Usage examples

### Text to Image Generation

```python
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
def main():
    pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image",torch_dtype=torch.bfloat16,device_map="cuda")
    prompt = "现代美食杂志风格的甜点制作教程图，主题为覆盆子慕斯蛋糕。整体布局干净明亮，分为四个主要区域：顶部左侧是黑色粗体标题“覆盆子慕斯蛋糕制作指南”，右侧搭配光线柔和的成品蛋糕特写照片，蛋糕呈淡粉色，表面点缀新鲜覆盆子与薄荷叶；左下方为配料清单区域，标题“配料”使用简洁字体，下方列有“面粉 150g”“鸡蛋 3个”“细砂糖 120g”“覆盆子果泥 200g”“明胶片 10g”“淡奶油 300ml”“新鲜覆盆子”等配料，每种配料旁配有简约线图标（如面粉袋、鸡蛋、糖罐等）；右下方是四个等大的步骤方框，每个方框内含高清微距实拍图及对应操作说明，从上到下依次为：步骤1展示打蛋器打发白色泡沫（对应说明“打发蛋白至干性发泡”），步骤2展示红白相间的混合物被刮刀翻拌（对应说明“轻柔翻拌果泥与面糊”），步骤3展示粉色液体被倒入圆形模具（对应说明“倒入模具并冷藏4小时”），步骤4展示成品蛋糕表面装饰覆盆子与薄荷叶（对应说明“用覆盆子和薄荷装饰”）；底部边缘设浅棕色信息条，左侧图标分别代表“准备时间：30分钟”“烹饪时间：20分钟”“份量：8人份”。整体色调以奶油白、淡粉色为主，背景带轻微纸质纹理，图文排版紧凑有序，信息层级分明。"
    image = pipe(
        prompt=prompt,
        height=89 * 32,
        width=45 * 32,
        num_inference_steps=30,
        guidance_scale=1.5,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("output_t2i.png")

if __name__ == "__main__":
    main()
```

### Image to Image Generation

```python
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image

def main():
    pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image",torch_dtype=torch.bfloat16,device_map="cuda")
    image_path = "cond.jpg" 
    prompt = "Replace the background of the snow forest with an underground station featuring an automatic escalator."
    image = Image.open(image_path).convert("RGB")
    image = pipe(
        prompt=prompt,
        image=[image], # can input multiple images for multi-image-to-image generation such as [image, image1]
        height=33 * 32,
        width=32 * 32,
        num_inference_steps=30,
        guidance_scale=1.5,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("output_t2i.png")

if __name__ == "__main__":
    main()
```

+ Since the AR model used in GLM-Image is configured with `do_sample=True` and a temperature of `0.95` by default, the generated images can vary significantly across runs. We do not recommend setting do_sample=False, as this may lead to incorrect or degenerate outputs from the AR model.

## GlmImagePipeline

[[autodoc]] GlmImagePipeline
  - all
  - __call__

## GlmImagePipelineOutput

[[autodoc]] pipelines.cogview4.pipeline_output.GlmImagePipelineOutput
