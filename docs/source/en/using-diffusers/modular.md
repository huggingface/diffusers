<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Modular Diffusers

Modular Diffusers is a unified pipeline that simplifies how you work with diffusion models. There are two main advantages of using modular Diffusers:

* Avoid rewriting an entire pipeline from scratch. Reuse existing blocks and only create new blocks for the functionalities you need.
* Flexibility. Compose pipeline blocks for one workflow and mix and match them for another workflow where a specific block works better.

The example below composes a pipeline with an [IP-Adapter](./loading_adapters#ip-adapter) to enable image prompting.

Create a [`ComponentsManager`] to manage the components (text encoders, UNets, VAE, etc.) in the pipeline. Add the [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) weights with [`add_from_pretrained`], and load the image encoder and feature extractor for the IP-Adapter with [`add`].

> [!TIP]
> Reduce memory usage by automatically offloading unused components to the CPU and loading them back on the GPU when they're needed.

```py
import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import ModularPipeline, StableDiffusionXLAutoPipeline
from diffusers.pipelines.components_manager import ComponentsManager
from diffusers.utils import load_image

components = ComponentsManager()
components.add_from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)

image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="sdxl_models/image_encoder", torch_dtype=torch.float16)
feature_extractor = CLIPImageProcessor(size=224, crop_size=224)

components.add("image_encoder", image_encoder)
components.add("feature_extractor", feature_extractor)
components.enable_auto_cpu_offload(device="cuda:0")
```

Use [`from_block`] to load the [`StableDiffusionXLAutoPipeline`] block into [`ModularPipeline`], and use [`update_states`] to update it with the components in [`ComponentsManager`].

```py
auto_pipe = ModularPipeline.from_block(StableDiffusionXLAutoPipeline())
auto_pipe.update_states(**components.components)
auto_pipe.update_states(**components.get(["image_encoder", "feature_extractor"]))
auto_pipe.to("cuda")
```

Load and set the IP-Adapter weights in the pipeline.

```py
auto_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
auto_pipe.set_ip_adapter_scale(0.6)
```

[`ModularPipeline`] automatically adapts to your input (text, image, mask image, IP-Adapter, etc.). You don't need to choose a specific pipeline for a task.

```py
ip_adapter_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy/img5.png")
output = auto_pipe(prompt="An astronaut eating a cake in space", ip_adapter_image=ip_adapter_image, output="images").images[0]
output
```

## Pipeline blocks

[`StableDiffusionXLAutoPipeline`] is a preset arrangement of pipeline blocks. It can be broken down into more modular blocks and rearranged.

This example will show you how to recreate the same setup with [`StableDiffusionXLAutoPipeline`] in a more modular way.
