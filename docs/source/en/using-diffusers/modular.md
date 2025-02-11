<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Modular Diffusers

Modular Diffusers is a unified pipeline that greatly simplifies how you work with diffusion models. There are two main advantages of using modular Diffusers:

* Avoid rewriting an entire pipeline from scratch. Reuse existing blocks and only create new blocks for the functionality you need.
* Flexibility. Compose pipeline blocks for one workflow and mix and match them for another workflow where a specific block works better.

Create a [`ComponentManager`] to manage the components in the pipeline. The example below adds the [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) weights. Reduce memory usage by automatically offloading unused components to the CPU and loading them back on the GPU when they're needed.

```py
import torch
from diffusers import ModularPipeline, StableDiffusionXLAutoPipeline
from diffusers.pipelines.components_manager import ComponentsManager

# Load model weights
components = ComponentsManager()
components.add_from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
components.enable_auto_cpu_offload(device="cuda:0")
```

Use `from_block` to load the [`StableDiffusionXLAutoPipeline`] block into [`ModularPipeline`], and then use [`update_states`] to update it with the [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) weights.

```py
# Create pipeline
auto_pipe = ModularPipeline.from_block(StableDiffusionXLAutoPipeline())
auto_pipe.update_states(**components.components)
auto_pipe.to("cuda")
```

Pass your prompt to the pipeline to generate an image.

```py
image = pipe(prompt="an astronaut", height=1024, width=2014, num_inference_steps=30)images[0]
```
