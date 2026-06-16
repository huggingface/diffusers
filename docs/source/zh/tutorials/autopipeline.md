<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

[AutoPipeline](../api/models/auto_model) 是一种按*任务和模型*选择的pipeline，会根据任务自动选择正确的pipeline子类。这样你就不用提前知道具体的pipeline子类名称，也能加载不同类型的pipeline。

这和 [`DiffusionPipeline`] 不同。后者是只按*模型*选择的pipeline，会根据模型自动选择pipeline子类。

[`AutoPipelineForImage2Image`] 会返回某个特定的pipeline子类，例如 [`StableDiffusionXLImg2ImgPipeline`]，它只能用于 image-to-image 任务。

```py
import torch
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.bfloat16, device_map="cuda",
)
print(pipeline)
"StableDiffusionXLImg2ImgPipeline {
  "_class_name": "StableDiffusionXLImg2ImgPipeline",
  ...
"
```

如果用同一个模型加载 [`DiffusionPipeline`]，则会返回 [`StableDiffusionXLPipeline`] 子类。它可以根据输入用于 text-to-image、image-to-image 或 inpainting 任务。

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.bfloat16, device_map="cuda",
)
print(pipeline)
"StableDiffusionXLPipeline {
  "_class_name": "StableDiffusionXLPipeline",
  ...
"
```

你可以查看 [mappings](https://github.com/huggingface/diffusers/blob/130fd8df54f24ffb006d84787b598d8adc899f23/src/diffusers/pipelines/auto_pipeline.py#L114)，确认某个模型是否受支持。

如果尝试加载不受支持的模型，就会报错。

```py
import torch
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", torch_dtype=torch.float16,
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```

[AutoPipeline](../api/models/auto_model) 一共有四种类型：

- [`AutoPipelineForText2Image`]
- [`AutoPipelineForImage2Image`]
- [`AutoPipelineForInpainting`]
- [`AutoPipelineForText2Audio`]

这些类都带有预定义的映射关系，会把某个pipeline关联到对应任务的子类上。

调用 [`~AutoPipelineForText2Image.from_pretrained`] 时，它会从 `model_index.json` 文件中提取类名，并根据映射关系为该任务选择合适的pipeline子类。
