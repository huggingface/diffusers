<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

[AutoPipeline](../api/pipelines/auto_pipeline) is a *task-and-model* pipeline that automatically selects the correct pipeline subclass based on the task. It handles the complexity of loading different pipeline subclasses without needing to know the specific pipeline subclass name.

This is unlike [`DiffusionPipeline`], a *model-only* pipeline that automatically selects the pipeline subclass based on the model.

```text
AutoPipelineForImage2Image.from_pretrained(model_id)
        |
        +-- read model_index.json  (e.g. StableDiffusionXLPipeline)
        +-- task mapping           (image-to-image)
        |
        v
StableDiffusionXLImg2ImgPipeline   // returned instance
```

[`AutoPipelineForImage2Image`] returns the task-specific subclass (for example, [`StableDiffusionXLImg2ImgPipeline`]), which can only be used for image-to-image tasks.

```py
import torch
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "RunDiffusion/Juggernaut-XL-v9", dtype=torch.bfloat16, device_map="cuda",  # or "mps", "xpu", "cpu"
)
print(pipeline)
# StableDiffusionXLImg2ImgPipeline {
#   "_class_name": "StableDiffusionXLImg2ImgPipeline",
#   ...
# }
```

Loading the same model with [`DiffusionPipeline`] returns the default text-to-image subclass, [`StableDiffusionXLPipeline`]. That pipeline is for text-to-image. For image-to-image or inpainting, load a task AutoPipeline such as [`AutoPipelineForImage2Image`] or [`AutoPipelineForInpainting`], or the matching task-specific subclass.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "RunDiffusion/Juggernaut-XL-v9", dtype=torch.bfloat16, device_map="cuda",  # or "mps", "xpu", "cpu"
)
print(pipeline)
# StableDiffusionXLPipeline {
#   "_class_name": "StableDiffusionXLPipeline",
#   ...
# }
```

## Switch tasks with from_pipe

Load a task AutoPipeline once, then switch tasks with [`~AutoPipelineForImage2Image.from_pipe`] without downloading the weights again. Components are reused from the source pipeline.

```py
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
  "RunDiffusion/Juggernaut-XL-v9", dtype=torch.bfloat16, device_map="cuda",  # or "mps", "xpu", "cpu"
)
pipeline_i2i = AutoPipelineForImage2Image.from_pipe(pipeline_t2i)
```

See [Reusing models in multiple pipelines](../using-diffusers/loading#reusing-models-in-multiple-pipelines) for more details.

Check the [mappings](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/auto_pipeline.py) to see whether a model is supported or not. Trying to load an unsupported model returns an error.

```py
import torch
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", dtype=torch.float16,
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```

There are four types of [AutoPipeline](../api/pipelines/auto_pipeline) classes:

- [`AutoPipelineForText2Image`]
- [`AutoPipelineForImage2Image`]
- [`AutoPipelineForInpainting`]
- [`AutoPipelineForText2Audio`]

Each of these classes has a predefined mapping, linking a pipeline to their task-specific subclass.

When [`~AutoPipelineForText2Image.from_pretrained`] is called, it extracts the class name from the `model_index.json` file and selects the appropriate pipeline subclass for the task based on the mapping.