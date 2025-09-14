<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ModularPipeline

[`ModularPipeline`] converts [`~modular_pipelines.ModularPipelineBlocks`]'s into an executable pipeline that loads models and performs the computation steps defined in the block. It is the main interface for running a pipeline and it is very similar to the [`DiffusionPipeline`] API.

The main difference is to include an expected `output` argument in the pipeline.

<hfoptions id="example">
<hfoption id="text-to-image">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS

blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

image = pipeline(prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", output="images")[0]
image.save("modular_t2i_out.png")
```

</hfoption>
<hfoption id="image-to-image">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import IMAGE2IMAGE_BLOCKS

blocks = SequentialPipelineBlocks.from_blocks_dict(IMAGE2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
init_image = load_image(url)
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt=prompt, image=init_image, strength=0.8, output="images")[0]
image.save("modular_i2i_out.png")
```

</hfoption>
<hfoption id="inpainting">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import INPAINT_BLOCKS
from diffusers.utils import load_image

blocks = SequentialPipelineBlocks.from_blocks_dict(INPAINT_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "A deep sea diver floating"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, output="images")[0]
image.save("moduar_inpaint_out.png")
```

</hfoption>
</hfoptions>

This guide will show you how to create a [`ModularPipeline`] and manage the components in it.

## Adding blocks

Blocks are [`InsertableDict`] objects that can be inserted at specific positions, providing a flexible way to mix-and-match blocks.

Use [`~modular_pipelines.modular_pipeline_utils.InsertableDict.insert`] on either the block class or `sub_blocks` attribute to add a block.

```py
# BLOCKS is dict of block classes, you need to add class to it
BLOCKS.insert("block_name", BlockClass, index)
# sub_blocks attribute contains instance, add a block instance to the  attribute
t2i_blocks.sub_blocks.insert("block_name", block_instance, index)
```

Use [`~modular_pipelines.modular_pipeline_utils.InsertableDict.pop`] on either the block class or `sub_blocks` attribute to remove a block.

```py
# remove a block class from preset
BLOCKS.pop("text_encoder")
# split out a block instance on its own
text_encoder_block = t2i_blocks.sub_blocks.pop("text_encoder")
```

Swap blocks by setting the existing block to the new block.

```py
# Replace block class in preset
BLOCKS["prepare_latents"] = CustomPrepareLatents
# Replace in sub_blocks attribute using an block instance
t2i_blocks.sub_blocks["prepare_latents"] = CustomPrepareLatents()
```

## Creating a pipeline

There are two ways to create a [`ModularPipeline`]. Assemble and create a pipeline from [`ModularPipelineBlocks`] or load an existing pipeline with [`~ModularPipeline.from_pretrained`].

You should also initialize a [`ComponentsManager`] to handle device placement and memory and component management.

> [!TIP]
> Refer to the [ComponentsManager](./components_manager) doc for more details about how it can help manage components across different workflows.

<hfoptions id="create">
<hfoption id="ModularPipelineBlocks">

Use the [`~ModularPipelineBlocks.init_pipeline`] method to create a [`ModularPipeline`] from the component and configuration specifications. This method loads the *specifications* from a `modular_model_index.json` file, but it doesn't load the *models* yet.

```py
from diffusers import ComponentsManager
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS

t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
components = ComponentsManager()
t2i_pipeline = t2i_blocks.init_pipeline(modular_repo_id, components_manager=components)
```

</hfoption>
<hfoption id="from_pretrained">

The [`~ModularPipeline.from_pretrained`] method creates a [`ModularPipeline`] from a modular repository on the Hub.

```py
from diffusers import ModularPipeline, ComponentsManager

components = ComponentsManager()
pipeline = ModularPipeline.from_pretrained("YiYiXu/modular-loader-t2i-0704", components_manager=components)
```

Add the `trust_remote_code` argument to load a custom [`ModularPipeline`].

```py
from diffusers import ModularPipeline, ComponentsManager

components = ComponentsManager()
modular_repo_id = "YiYiXu/modular-diffdiff-0704"
diffdiff_pipeline = ModularPipeline.from_pretrained(modular_repo_id, trust_remote_code=True, components_manager=components)
```

</hfoption>
</hfoptions>

## Loading components

A [`ModularPipeline`] doesn't automatically instantiate with components. It only loads the configuration and component specifications. You can load all components with [`~ModularPipeline.load_components`] or only load specific components with [`~ModularPipeline.load_components`].

<hfoptions id="load">
<hfoption id="load_components">

```py
import torch

t2i_pipeline.load_components(torch_dtype=torch.float16)
t2i_pipeline.to("cuda")
```

</hfoption>
<hfoption id="load_components">

The example below only loads the UNet and VAE.

```py
import torch

t2i_pipeline.load_components(names=["unet", "vae"], torch_dtype=torch.float16)
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
    repo='RunDiffusion/Juggernaut-XL-v9',
    subfolder='unet',
    variant='fp16',
    default_creation_method='from_pretrained'
)

# modify to load from a different repository
unet_spec.repo = "stabilityai/stable-diffusion-xl-base-1.0"

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
