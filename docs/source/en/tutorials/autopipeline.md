<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

[AutoPipeline](../api/models/auto_model) is a *task-and-model* pipeline that automatically selects the correct pipeline subclass based on the task. It handles the complexity of loading different pipeline subclasses without needing to know the specific pipeline subclass name.

This is unlike [`DiffusionPipeline`], a *model-only* pipeline that automatically selects the pipeline subclass based on the model.

[`AutoPipelineForImage2Image`] returns a specific pipeline subclass, (for example, [`StableDiffusionXLImg2ImgPipeline`]), which can only be used for image-to-image tasks.

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

Loading the same model with [`DiffusionPipeline`] returns the [`StableDiffusionXLPipeline`] subclass. It can be used for text-to-image, image-to-image, or inpainting tasks depending on the inputs.

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

Check the [mappings](https://github.com/huggingface/diffusers/blob/130fd8df54f24ffb006d84787b598d8adc899f23/src/diffusers/pipelines/auto_pipeline.py#L114) to see whether a model is supported or not.

Trying to load an unsupported model returns an error.

```py
import torch
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", torch_dtype=torch.float16,
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```

There are three types of [AutoPipeline](../api/models/auto_model) classes, [`AutoPipelineForText2Image`], [`AutoPipelineForImage2Image`] and [`AutoPipelineForInpainting`]. Each of these classes have a predefined mapping, linking a pipeline to their task-specific subclass.

When [`~AutoPipelineForText2Image.from_pretrained`] is called, it extracts the class name from the `model_index.json` file and selects the appropriate pipeline subclass for the task based on the mapping.