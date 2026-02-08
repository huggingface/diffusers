<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ModularPipeline

[`ModularPipeline`] converts [`~modular_pipelines.ModularPipelineBlocks`] into an executable pipeline that loads models and performs the computation steps defined in the blocks. It is the main interface for running a pipeline and the API is very similar to [`DiffusionPipeline`] but with a few key differences.

**Loading is lazy.** With [`DiffusionPipeline`], [`~DiffusionPipeline.from_pretrained`] creates the pipeline and loads all models at the same time. With [`ModularPipeline`], creating and loading are two separate steps: [`~ModularPipeline.from_pretrained`] reads the configuration and knows where to load each component from, but doesn't actually load the model weights. You load the models later with [`~ModularPipeline.load_components`], which is where you pass loading arguments like `torch_dtype` and `quantization_config`.

**Two ways to create a pipeline.** You can use [`~ModularPipeline.from_pretrained`] with an existing diffusers model repository — it automatically maps to the default pipeline blocks and then converts to a [`ModularPipeline`] with no extra setup. Currently supported models include SDXL, Wan, Qwen, Z-Image, Flux, and Flux2. You can also assemble your own pipeline from [`ModularPipelineBlocks`] and convert it with the [`~ModularPipelineBlocks.init_pipeline`] method (see [Creating a pipeline](#creating-a-pipeline) for more details).

**Running the pipeline is the same.** Once loaded, you call the pipeline with the same arguments you're used to. A single [`ModularPipeline`] can support multiple workflows (text-to-image, image-to-image, inpainting, etc.) when the pipeline blocks use [`AutoPipelineBlocks`](./auto_pipeline) to automatically select the workflow based on your inputs.

Below are complete examples for text-to-image, image-to-image, and inpainting with SDXL.

<hfoptions id="example">
<hfoption id="text-to-image">

```py
import torch
from diffusers import ModularPipeline

pipeline = ModularPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

image = pipeline(prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k").images[0]
image.save("modular_t2i_out.png")
```

</hfoption>
<hfoption id="image-to-image">

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipeline = ModularPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
init_image = load_image(url)
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt=prompt, image=init_image, strength=0.8).images[0]
image.save("modular_i2i_out.png")
```

</hfoption>
<hfoption id="inpainting">

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipeline = ModularPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "A deep sea diver floating"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85).images[0]
image.save("modular_inpaint_out.png")
```

</hfoption>
</hfoptions>

This guide will show you how to create a [`ModularPipeline`], manage the components in it, and run it.

## Creating a pipeline

There are two ways to create a [`ModularPipeline`]. Assemble and create a pipeline from [`ModularPipelineBlocks`] with [`~ModularPipelineBlocks.init_pipeline`], or load an existing pipeline with [`~ModularPipeline.from_pretrained`].

You should also initialize a [`ComponentsManager`] to handle device placement and memory and component management.

> [!TIP]
> Refer to the [ComponentsManager](./components_manager) doc for more details about how it can help manage components across different workflows.

### init_pipeline

[`~ModularPipelineBlocks.init_pipeline`] converts any [`ModularPipelineBlocks`] into a [`ModularPipeline`].

Let's define a minimal block to see how it works:

```py
from transformers import CLIPTextModel
from diffusers.modular_pipelines import (
    ComponentSpec,
    ModularPipelineBlocks,
    PipelineState,
)

class MyBlock(ModularPipelineBlocks):
    @property
    def expected_components(self):
        return [
            ComponentSpec(
                name="text_encoder",
                type_hint=CLIPTextModel,
                pretrained_model_name_or_path="openai/clip-vit-large-patch14",
            ),
        ]

    def __call__(self, components, state: PipelineState) -> PipelineState:
        return components, state
```

Call [`~ModularPipelineBlocks.init_pipeline`] to convert it into a pipeline. The `blocks` attribute on the pipeline is the blocks it was created from — it determines the expected inputs, outputs, and computation logic.

```py
block = MyBlock()
pipe = block.init_pipeline()
pipe.blocks
```

```
MyBlock {
  "_class_name": "MyBlock",
  "_diffusers_version": "0.37.0.dev0"
}
```

> [!WARNING]
> Blocks are mutable — you can freely add, remove, or swap blocks before creating a pipeline. However, once a pipeline is created, modifying `pipeline.blocks` won't affect the pipeline because it returns a copy. If you want a different block structure, create a new pipeline after modifying the blocks.

When you call [`~ModularPipelineBlocks.init_pipeline`] without a repository, it uses the `pretrained_model_name_or_path` defined in the block's [`ComponentSpec`] to determine where to load each component from. Printing the pipeline shows the component loading configuration.

```py
pipe
ModularPipeline {
  "_blocks_class_name": "MyBlock",
  "_class_name": "ModularPipeline",
  "_diffusers_version": "0.37.0.dev0",
  "text_encoder": [
    null,
    null,
    {
      "pretrained_model_name_or_path": "openai/clip-vit-large-patch14",
      "revision": null,
      "subfolder": "",
      "type_hint": [
        "transformers",
        "CLIPTextModel"
      ],
      "variant": null
    }
  ]
}
```

If you pass a repository to [`~ModularPipelineBlocks.init_pipeline`], it overrides the loading path by matching your block's components against the pipeline config in that repository (`model_index.json` or `modular_model_index.json`).

In the example below, the `pretrained_model_name_or_path` will be updated to `"stabilityai/stable-diffusion-xl-base-1.0"`.

```py
pipe = block.init_pipeline("stabilityai/stable-diffusion-xl-base-1.0")
pipe
ModularPipeline {
  "_blocks_class_name": "MyBlock",
  "_class_name": "ModularPipeline",
  "_diffusers_version": "0.37.0.dev0",
  "text_encoder": [
    null,
    null,
    {
      "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
      "revision": null,
      "subfolder": "text_encoder",
      "type_hint": [
        "transformers",
        "CLIPTextModel"
      ],
      "variant": null
    }
  ]
}
```

If a component in your block doesn't exist in the repository, it remains `null` and is skipped during [`~ModularPipeline.load_components`].

### from_pretrained

[`~ModularPipeline.from_pretrained`] is a convenient way to create a [`ModularPipeline`] without defining blocks yourself.

It works with three types of repositories.

**A regular diffusers repository.** Pass any supported model repository and it automatically maps to the default pipeline blocks. Currently supported models include SDXL, Wan, Qwen, Z-Image, Flux, and Flux2.

```py
from diffusers import ModularPipeline, ComponentsManager

components = ComponentsManager()
pipeline = ModularPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", components_manager=components
)
```

**A modular repository.** These repositories contain a `modular_model_index.json` that specifies where to load each component from — the components can come from different repositories and the modular repository itself may not contain any model weights. For example, [diffusers/flux2-bnb-4bit-modular](https://huggingface.co/diffusers/flux2-bnb-4bit-modular) loads a quantized transformer from one repository and the remaining components from another. See [Modular repository](#modular-repository) for more details on the format.

```py
from diffusers import ModularPipeline, ComponentsManager

components = ComponentsManager()
pipeline = ModularPipeline.from_pretrained(
    "diffusers/flux2-bnb-4bit-modular", components_manager=components
)
```

**A modular repository with custom code.** Some repositories include custom pipeline blocks alongside the loading configuration. Add `trust_remote_code=True` to load them. See [Custom blocks](./custom_blocks) for how to create your own.

```py
from diffusers import ModularPipeline, ComponentsManager

components = ComponentsManager()
pipeline = ModularPipeline.from_pretrained(
    "diffusers/Florence2-image-Annotator", trust_remote_code=True, components_manager=components
)
```

## Loading components

A [`ModularPipeline`] doesn't automatically instantiate with components. It only loads the configuration and component specifications. You can load components with [`~ModularPipeline.load_components`].

This will load all the components that have a valid loading spec.

```py
import torch

pipeline.load_components(torch_dtype=torch.float16)
```

You can also load specific components by name. The example below only loads the text_encoder.

```py
pipeline.load_components(names=["text_encoder"], torch_dtype=torch.float16)
```

After loading, printing the pipeline shows which components are loaded — the first two fields change from `null` to the component's library and class.

```py
pipeline
```

```
# text_encoder is loaded - shows library and class
"text_encoder": [
  "transformers",
  "CLIPTextModel",
  { ... }
]

# unet is not loaded yet - still null
"unet": [
  null,
  null,
  { ... }
]
```

Loading keyword arguments like `torch_dtype`, `variant`, `revision`, and `quantization_config` are passed through to `from_pretrained()` for each component. You can pass a single value to apply to all components, or a dict to set per-component values.

```py
# apply bfloat16 to all components
pipeline.load_components(torch_dtype=torch.bfloat16)

# different dtypes per component
pipeline.load_components(torch_dtype={"transformer": torch.bfloat16, "default": torch.float32})
```

Note that [`~ModularPipeline.load_components`] only loads components that haven't been loaded yet and have a valid loading spec. This means if you've already set a component on the pipeline, calling [`~ModularPipeline.load_components`] again won't reload it.

## Updating components

[`~ModularPipeline.update_components`] replaces a component on the pipeline with a new one. When a component is updated, the loading specifications are also updated in the pipeline config and [`~ModularPipeline.load_components`] will skip it on subsequent calls.

### From AutoModel

You can pass a model object loaded with `AutoModel.from_pretrained()`. Models loaded this way are automatically tagged with their loading information.

```py
from diffusers import AutoModel

unet = AutoModel.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9", subfolder="unet", variant="fp16", torch_dtype=torch.float16
)
pipeline.update_components(unet=unet)
```

### From ComponentSpec

Use [`~ModularPipeline.get_component_spec`] to get a copy of the current component specification, modify it, and load a new component.

```py
unet_spec = pipeline.get_component_spec("unet")

# modify to load from a different repository
unet_spec.pretrained_model_name_or_path = "RunDiffusion/Juggernaut-XL-v9"

# load and update
unet = unet_spec.load(torch_dtype=torch.float16)
pipeline.update_components(unet=unet)
```

You can also create a [`ComponentSpec`] from scratch.

Not all components are loaded from pretrained weights — some are created from a config (listed under `pipeline.config_component_names`). For these, use [`~ComponentSpec.create`] instead of [`~ComponentSpec.load`].

```py
guider_spec = pipeline.get_component_spec("guider")
guider_spec.config = {"guidance_scale": 5.0}
guider = guider_spec.create()
pipeline.update_components(guider=guider)
```

Or simply pass the object directly.

```py
from diffusers.guiders import ClassifierFreeGuidance

guider = ClassifierFreeGuidance(guidance_scale=5.0)
pipeline.update_components(guider=guider)
```

See the [Guiders](./guiders) guide for more details on available guiders and how to configure them.

## Modular repository

A repository is required if the pipeline blocks use *pretrained components*. The repository supplies loading specifications and metadata.

[`ModularPipeline`] works with regular diffusers repositories out of the box. However, you can also create a *modular repository* for more flexibility. A modular repository contains a `modular_model_index.json` file containing the following 3 elements.

- `library` and `class` shows which library the component was loaded from and its class. If `null`, the component hasn't been loaded yet.
- `loading_specs_dict` contains the information required to load the component such as the repository and subfolder it is loaded from.

The key advantage of a modular repository is that components can be loaded from different repositories. For example, [diffusers/flux2-bnb-4bit-modular](https://huggingface.co/diffusers/flux2-bnb-4bit-modular) loads a quantized transformer from `diffusers/FLUX.2-dev-bnb-4bit` while loading the remaining components from `black-forest-labs/FLUX.2-dev`.

To convert a regular diffusers repository into a modular one, create the pipeline using the regular repository, and then save it using `save_pretrained()`. The saved repository will contain a `modular_model_index.json` with all the loading specifications. Optionnally, you can pass a `repo_id` and `push_to_hub=True` to publish it on Huggingface Hub.

```py
from diffusers import ModularPipeline

# load from a regular repo
pipeline = ModularPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# push as a modular repository
pipeline.save_pretrained("local/path", repo_id="my-username/sdxl-modular", push_to_hub=True)
```

A modular repository can also include custom pipeline blocks as Python code. This allows you to share specialized blocks that aren't native to Diffusers. For example, [diffusers/Florence2-image-Annotator](https://huggingface.co/diffusers/Florence2-image-Annotator) contains custom blocks alongside the loading configuration:

```
Florence2-image-Annotator/
├── block.py                    # Custom pipeline blocks implementation
├── config.json                 # Pipeline configuration and auto_map
├── mellon_config.json          # UI configuration for Mellon
└── modular_model_index.json    # Component loading specifications
```

The `config.json` file contains an `auto_map` key that tells [`ModularPipeline`] where to find the custom blocks:

```json
{
  "_class_name": "Florence2AnnotatorBlocks",
  "auto_map": {
    "ModularPipelineBlocks": "block.Florence2AnnotatorBlocks"
  }
}
```

Load custom code repositories with `trust_remote_code=True` as shown in [from_pretrained](#from_pretrained). See [Custom blocks](./custom_blocks) for how to create and share your own.