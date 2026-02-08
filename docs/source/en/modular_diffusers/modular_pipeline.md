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
image.save("moduar_inpaint_out.png")
```

</hfoption>
</hfoptions>

This guide will show you how to create a [`ModularPipeline`] and manage the components in it, and run it.

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

Call [`~ModularPipelineBlocks.init_pipeline`] to convert it into a pipeline. The `blocks` attribute on the pipeline is the blocks it was created from — it determines the expected inputs, outputs, and computation logic.

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

A [`ModularPipeline`] doesn't automatically instantiate with components. It only loads the configuration and component specifications. You can load all components with [`~ModularPipeline.load_components`] or only load specific components with [`~ModularPipeline.load_components`].

<hfoptions id="load">
<hfoption id="load_components">

```py
import torch

pipeline.load_components(torch_dtype=torch.float16)
t2i_pipeline.to("cuda")
```

</hfoption>
<hfoption id="load_components">

The example below only loads the UNet and VAE.

```py
import torch

pipeline.load_components(names=["unet", "vae"], torch_dtype=torch.float16)
```

</hfoption>
</hfoptions>

Print the pipeline to inspect the loaded pretrained components.

```py
t2i_pipeline
```

This should match the `modular_model_index.json` file from the modular repository a pipeline is initialized from. If a pipeline doesn't need a component, it won't be included even if it exists in the modular repository.

To modify where components are loaded from, edit the `modular_model_index.json` file in the repository and change it to your desired loading path. The example below loads a UNet from a different repository.

```json
# original
"unet": [
  null, null,
  {
    "repo": "stabilityai/stable-diffusion-xl-base-1.0",
    "subfolder": "unet",
    "variant": "fp16"
  }
]

# modified
"unet": [
  null, null,
  {
    "repo": "RunDiffusion/Juggernaut-XL-v9",
    "subfolder": "unet",
    "variant": "fp16"
  }
]
```

### Component loading status

The pipeline properties below provide more information about which components are loaded.

Use `component_names` to return all expected components.

```py
t2i_pipeline.component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'guider', 'scheduler', 'unet', 'vae', 'image_processor']
```

Use `null_component_names` to return components that aren't loaded yet. Load these components with [`~ModularPipeline.from_pretrained`].

```py
t2i_pipeline.null_component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']
```

Use `pretrained_component_names` to return components that will be loaded from pretrained models.

```py
t2i_pipeline.pretrained_component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler', 'unet', 'vae']
```

Use `config_component_names` to return components that are created with the default config (not loaded from a modular repository). Components from a config aren't included because they are already initialized during pipeline creation. This is why they aren't listed in `null_component_names`.

```py
t2i_pipeline.config_component_names
['guider', 'image_processor']
```

## Updating components

Components may be updated depending on whether it is a *pretrained component* or a *config component*.

> [!WARNING]
> A component may change from pretrained to config when updating a component. The component type is initially defined in a block's `expected_components` field.

A pretrained component is updated with [`ComponentSpec`] whereas a config component is updated by eihter passing the object directly or with [`ComponentSpec`].

The [`ComponentSpec`] shows `default_creation_method="from_pretrained"` for a pretrained component shows `default_creation_method="from_config` for a config component.

To update a pretrained component, create a [`ComponentSpec`] with the name of the component and where to load it from. Use the [`~ComponentSpec.load`] method to load the component.

```py
from diffusers import ComponentSpec, UNet2DConditionModel

unet_spec = ComponentSpec(name="unet",type_hint=UNet2DConditionModel, repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", variant="fp16")
unet = unet_spec.load(torch_dtype=torch.float16)
```

The [`~ModularPipeline.update_components`] method replaces the component with a new one.

```py
t2i_pipeline.update_components(unet=unet2)
```

When a component is updated, the loading specifications are also updated in the pipeline config.

### Component extraction and modification

When you use [`~ComponentSpec.load`], the new component maintains its loading specifications. This makes it possible to extract the specification and recreate the component.

```py
spec = ComponentSpec.from_component("unet", unet2)
spec
ComponentSpec(name='unet', type_hint=<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>, description=None, config=None, repo='stabilityai/stable-diffusion-xl-base-1.0', subfolder='unet', variant='fp16', revision=None, default_creation_method='from_pretrained')
unet2_recreated = spec.load(torch_dtype=torch.float16)
```

The [`~ModularPipeline.get_component_spec`] method gets a copy of the current component specification to modify or update.

```py
unet_spec = t2i_pipeline.get_component_spec("unet")
unet_spec
ComponentSpec(
    name='unet',
    type_hint=<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>,
    pretrained_model_name_or_path='RunDiffusion/Juggernaut-XL-v9',
    subfolder='unet',
    variant='fp16',
    default_creation_method='from_pretrained'
)

# modify to load from a different repository
unet_spec.pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

# load component with modified spec
unet = unet_spec.load(torch_dtype=torch.float16)
```

## Modular repository

A repository is required if the pipeline blocks use *pretrained components*. The repository supplies loading specifications and metadata.

[`ModularPipeline`] specifically requires *modular repositories* (see [example repository](https://huggingface.co/YiYiXu/modular-diffdiff)) which are more flexible than a typical repository. It contains a `modular_model_index.json` file containing the following 3 elements.

- `library` and `class` shows which library the component was loaded from and it's class. If `null`, the component hasn't been loaded yet.
- `loading_specs_dict` contains the information required to load the component such as the repository and subfolder it is loaded from.

Unlike standard repositories, a modular repository can fetch components from different repositories based on the `loading_specs_dict`. Components don't need to exist in the same repository.

A modular repository may contain custom code for loading a [`ModularPipeline`]. This allows you to use specialized blocks that aren't native to Diffusers.

```
modular-diffdiff-0704/
├── block.py                    # Custom pipeline blocks implementation
├── config.json                 # Pipeline configuration and auto_map
└── modular_model_index.json    # Component loading specifications
```

The [config.json](https://huggingface.co/YiYiXu/modular-diffdiff-0704/blob/main/config.json) file contains an `auto_map` key that points to where a custom block is defined in `block.py`.

```json
{
  "_class_name": "DiffDiffBlocks",
  "auto_map": {
    "ModularPipelineBlocks": "block.DiffDiffBlocks"
  }
}
```
