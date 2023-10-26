<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

`AutoPipeline` is designed to:

1. make it easy for you to load a checkpoint for a task without knowing the specific pipeline class to use
2. use multiple pipelines in your workflow

Based on the task, the `AutoPipeline` class automatically retrieves the relevant pipeline given the name or path to the pretrained weights with the `from_pretrained()` method.

To seamlessly switch between tasks with the same checkpoint without reallocating additional memory, use the `from_pipe()` method to transfer the components from the original pipeline to the new one.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = pipeline(prompt, num_inference_steps=25).images[0]
```

<Tip>

Check out the [AutoPipeline](/tutorials/autopipeline) tutorial to learn how to use this API!

</Tip>

`AutoPipeline` supports text-to-image, image-to-image, and inpainting for the following diffusion models:

- [Stable Diffusion](./stable_diffusion)
- [ControlNet](./controlnet)
- [Stable Diffusion XL (SDXL)](./stable_diffusion/stable_diffusion_xl)
- [DeepFloyd IF](./if) 
- [Kandinsky](./kandinsky)
- [Kandinsky 2.2](./kandinsky#kandinsky-22)


## AutoPipelineForText2Image

[[autodoc]] AutoPipelineForText2Image
	- all
	- from_pretrained
	- from_pipe


## AutoPipelineForImage2Image

[[autodoc]] AutoPipelineForImage2Image
	- all
	- from_pretrained
	- from_pipe

## AutoPipelineForInpainting

[[autodoc]] AutoPipelineForInpainting
	- all
	- from_pretrained
	- from_pipe


