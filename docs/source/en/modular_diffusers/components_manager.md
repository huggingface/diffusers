<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ComponentsManager

The [`ComponentsManager`] is a model registry and management system for Modular Diffusers. It adds and tracks models, stores useful metadata (model size, device placement, adapters), prevents duplicate model instances, and supports offloading.

This guide will show you how to use [`ComponentsManager`] to manage components and device memory.

## Add a component

The [`ComponentsManager`] should be created alongside a [`ModularPipeline`] in either [`~ModularPipeline.from_pretrained`] or [`~ModularPipelineBlocks.init_pipeline`].

> [!TIP]
> The `collection` parameter is optional but makes it easier to organize and manage components.

<hfoptions id="create">
<hfoption id="from_pretrained">

```py
from diffusers import ModularPipeline, ComponentsManager

comp = ComponentsManager()
pipe = ModularPipeline.from_pretrained("YiYiXu/modular-demo-auto", components_manager=comp, collection="test1")
```

</hfoption>
<hfoption id="init_pipeline">

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
</hfoptions>

Components are only loaded and registered when using [`~ModularPipeline.load_components`] or [`~ModularPipeline.load_components`]. The example below uses [`~ModularPipeline.load_components`] to create a second pipeline that reuses all the components from the first one, and assigns it to a different collection

```py
pipe.load_components()
pipe2 = ModularPipeline.from_pretrained("YiYiXu/modular-demo-auto", components_manager=comp, collection="test2")
```

Use the [`~ModularPipeline.null_component_names`] property to identify any components that need to be loaded, retrieve them with [`~ComponentsManager.get_components_by_names`], and then call [`~ModularPipeline.update_components`] to add the missing components.

```py
pipe2.null_component_names 
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'image_encoder', 'unet', 'vae', 'scheduler', 'controlnet']

comp_dict = comp.get_components_by_names(names=pipe2.null_component_names)
pipe2.update_components(**comp_dict)
```

To add individual components, use the [`~ComponentsManager.add`] method. This registers a component with a unique id.

```py
from diffusers import AutoModel

text_encoder = AutoModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
component_id = comp.add("text_encoder", text_encoder)
comp
```

Use [`~ComponentsManager.remove`] to remove a component using their id.

```py
comp.remove("text_encoder_139917733042864")
```

## Retrieve a component

The [`ComponentsManager`] provides several methods to retrieve registered components.

### get_one

The [`~ComponentsManager.get_one`] method returns a single component and supports pattern matching for the `name` parameter. If multiple components match, [`~ComponentsManager.get_one`] returns an error.

| Pattern     | Example                          | Description                               |
|-------------|----------------------------------|-------------------------------------------|
| exact       | `comp.get_one(name="unet")`      | exact name match                          |
| wildcard    | `comp.get_one(name="unet*")`     | names starting with "unet"                |
| exclusion   | `comp.get_one(name="!unet")`     | exclude components named "unet"           |
| or          | `comp.get_one(name="unet&#124;vae")`  | name is "unet" or "vae"                   |

[`~ComponentsManager.get_one`] also filters components by the `collection` argument or `load_id` argument.

```py
comp.get_one(name="unet", collection="sdxl")
```

### get_components_by_names

The [`~ComponentsManager.get_components_by_names`] method accepts a list of names and returns a dictionary mapping names to components. This is especially useful with [`ModularPipeline`] since they provide lists of required component names and the returned dictionary can be passed directly to [`~ModularPipeline.update_components`].

```py
component_dict = comp.get_components_by_names(names=["text_encoder", "unet", "vae"])
{"text_encoder": component1, "unet": component2, "vae": component3}
```

## Duplicate detection

It is recommended to load model components with [`ComponentSpec`] to assign components with a unique id that encodes their loading parameters. This allows [`ComponentsManager`] to automatically detect and prevent duplicate model instances even when different objects represent the same underlying checkpoint.

```py
from diffusers import ComponentSpec, ComponentsManager
from transformers import CLIPTextModel

comp = ComponentsManager()

# Create ComponentSpec for the first text encoder
spec = ComponentSpec(name="text_encoder", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", type_hint=AutoModel)
# Create ComponentSpec for a duplicate text encoder (it is same checkpoint, from the same repo/subfolder)
spec_duplicated = ComponentSpec(name="text_encoder_duplicated", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", type_hint=CLIPTextModel)

# Load and add both components - the manager will detect they're the same model
comp.add("text_encoder", spec.load())
comp.add("text_encoder_duplicated", spec_duplicated.load())
```

This returns a warning with instructions for removing the duplicate.

```py
ComponentsManager: adding component 'text_encoder_duplicated_139917580682672', but it has duplicate load_id 'stabilityai/stable-diffusion-xl-base-1.0|text_encoder|null|null' with existing components: text_encoder_139918506246832. To remove a duplicate, call `components_manager.remove('<component_id>')`.
'text_encoder_duplicated_139917580682672'
```

You could also add a component without using [`ComponentSpec`] and duplicate detection still works in most cases even if you're adding the same component under a different name.

However, [`ComponentManager`] can't detect duplicates when you load the same component into different objects. In this case, you should load a model with [`ComponentSpec`].

```py
text_encoder_2 = AutoModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
comp.add("text_encoder", text_encoder_2)
'text_encoder_139917732983664'
```

## Collections

Collections are labels assigned to components for better organization and management. Add a component to a collection with the `collection` argument in [`~ComponentsManager.add`].

Only one component per name is allowed in each collection. Adding a second component with the same name automatically removes the first component.

```py
from diffusers import ComponentSpec, ComponentsManager

comp = ComponentsManager()
# Create ComponentSpec for the first UNet
spec = ComponentSpec(name="unet", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", type_hint=AutoModel)
# Create ComponentSpec for a different UNet
spec2 = ComponentSpec(name="unet", repo="RunDiffusion/Juggernaut-XL-v9", subfolder="unet", type_hint=AutoModel, variant="fp16")

# Add both UNets to the same collection - the second one will replace the first
comp.add("unet", spec.load(), collection="sdxl")
comp.add("unet", spec2.load(), collection="sdxl")
```

This makes it convenient to work with node-based systems because you can:

- Mark all models as loaded from one node with the `collection` label.
- Automatically replace models when new checkpoints are loaded under the same name.
- Batch delete all models in a collection when a node is removed.

## Offloading

The [`~ComponentsManager.enable_auto_cpu_offload`] method is a global offloading strategy that works across all models regardless of which pipeline is using them. Once enabled, you don't need to worry about device placement if you add or remove components.

```py
comp.enable_auto_cpu_offload(device="cuda")
```

All models begin on the CPU and [`ComponentsManager`] moves them to the appropriate device right before they're needed, and moves other models back to the CPU when GPU memory is low.

You can set your own rules for which models to offload first.
