<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Overview

The Modular Diffusers Framework consist of three main components

## ModularPipelineBlocks

Pipeline blocks are the fundamental building blocks of the Modular Diffusers system. All pipeline blocks inherit from the base class `ModularPipelineBlocks`, including:
- [`PipelineBlock`](TODO)
- [`SequentialPipelineBlocks`](TODO)
- [`LoopSequentialPipelineBlocks`](TODO)
- [`AutoPipelineBlocks`](TODO)


Each block defines:

**Specifications:**
- Inputs: User-provided parameters that the block expects
- Intermediate inputs: Variables from other blocks that this block needs
- Intermediate outputs: Variables this block produces for other blocks to use
- Components: Models and processors the block requires (e.g., UNet, VAE, scheduler)

**Computation:**
- `__call__` method: Defines the actual computational steps within the block

Pipeline blocks are essentially **"definitions"** - they define the specifications and computational steps for a pipeline, but are not runnable until converted into a `ModularPipeline` object.

All blocks interact with a global `PipelineState` object that maintains the pipeline's state throughout execution.

### Load/save a custom `ModularPipelineBlocks`

You can load a custom pipeline block from a hub repository directly

```py
from diffusers import ModularPipelineBlocks
diffdiff_block = ModularPipelineBlocks.from_pretrained(repo_id, trust_remote_code=True)
```

to save, and publish to a hub repository

```py
diffdiff_block.save(repo_id)
```

## PipelineState & BlockState

`PipelineState` and `BlockState` manage dataflow between pipeline blocks. `PipelineState` acts as the global state container that `ModularPipelineBlocks` operate on - each block gets a local view (`BlockState`) of the relevant variables it needs from `PipelineState`, performs its operations, and then updates PipelineState with any changes.

<Tip>

You typically don't need to manually create or manage these state objects. The `ModularPipeline` automatically creates and manages them for you. However, understanding their roles is important for developing custom pipeline blocks.

</Tip>

## ModularPipeline

`ModularPipeline` is the main interface to create and execute pipelines in the Modular Diffusers system.

### Create a `ModularPipeline`

Each `ModularPipelineBlocks` has an `init_pipeline` method that can initialize a `ModularPipeline` object based on its component and configuration specifications.

```py
>>> pipeline = blocks.init_pipeline(pretrained_model_name_or_path)
```

`ModularPipeline` only works with modular repositories, so make sure `pretrained_model_name_or_path` points to a modular repo (you can see an example [here](https://huggingface.co/YiYiXu/modular-diffdiff)).

The main differences from standard diffusers repositories are:

1. `modular_model_index.json` vs `model_index.json`

In standard `model_index.json`, each component entry is a `(library, class)` tuple:

```py
"text_encoder": [
  "transformers",
  "CLIPTextModel"
],
```

In `modular_model_index.json`, each component entry contains 3 elements: `(library, class, loading_specs {})`

- `library` and `class`: Information about the actual component loaded in the pipeline at the time of saving (can be `None` if not loaded)
- **`loading_specs`**: A dictionary containing all information required to load this component, including `repo`, `revision`, `subfolder`, `variant`, and `type_hint`

```py
"text_encoder": [
  null,  # library (same as model_index.json)
  null,  # class (same as model_index.json)
  {      # loading specs map (unique to modular_model_index.json)
    "repo": "stabilityai/stable-diffusion-xl-base-1.0",  # can be a different repo
    "revision": null,
    "subfolder": "text_encoder",
    "type_hint": [  # (library, class) for the expected component class
      "transformers",  
      "CLIPTextModel"
    ],
    "variant": null
  }
],
```

2. Cross-Repository Component Loading

Unlike standard repositories where components must be in subfolders within the same repo, modular repositories can fetch components from different repositories based on the `loading_specs` dictionary. In our example above, the `text_encoder` component will be fetched from the "text_encoder" folder in `stabilityai/stable-diffusion-xl-base-1.0` while other components come from different repositories.


<Tip>

💡 We recommend using `ModularPipeline` with Component Manager by passing a `components_manager`:

```py
>>> components = ComponentsManager()
>>> pipeline = blocks.init_pipeline(pretrained_model_name_or_path, components_manager=components)
```

This helps you to:
1. Detect and manage duplicated models (warns when trying to register an existing model)
2. Easily reuse components across different pipelines
3. Apply offloading strategies across multiple pipelines

You can read more about Components Manager [here](TODO)

</Tip>


Unlike `DiffusionPipeline`, you need to explicitly load model components using `load_components`:

```py
>>> pipeline.load_components(torch_dtype=torch.float16)
>>> pipeline.to(device)
```

You can partially load specific components using the `component_names` argument, for example to only load unet and vae:

```py
>>> pipeline.load_components(component_names=["unet", "vae"])
```

<Tip>

💡 You can inspect the pipeline's `config` attribute (which contains the same structure as `modular_model_index.json` we just walked through) to check the "loading status" of the pipeline, e.g. what components this pipeline expects to load and their loading specs, what components are already loaded and their actual class & loading specs etc.

</Tip>

### Execute a `ModularPipeline`

The API to run the `ModularPipeline` is very similar to how you would run a regular `DiffusionPipeline`:

```py
>>> image = pipeline(prompt="a cat", num_inference_steps=15, output="images")[0]
```

There are a few key differences though:
1. You can also pass a `PipelineState` object directly to the pipeline instead of individual arguments
2. If you do not specify the `output` argument, it returns the `PipelineState` object
3. You can pass a list as `output`, e.g. `pipeline(... output=["images", "latents"])` will return a dictionary containing both the generated image and the final denoised latents

Under the hood, `ModularPipeline`'s `__call__` method is a wrapper around the pipeline blocks' `__call__` method: it creates a `PipelineState` object and populates it with user inputs, then returns the output to the user based on the `output` argument. It also ensures that all pipeline-level config and components are exposed to all pipeline blocks by preparing and passing a `components` input.

### Load a `ModularPipeline` from hub

You can directly load a `ModularPipeline` from a HuggingFace Hub repository, as long as it's a modular repo

```py
pipeine = ModularPipeline.from_pretrained(repo_id, components_manager=..., collection=...)
```

Loading custom code is also supported, just pass a `trust_remote_code=True` argument:

```py
diffdiff_pipeline = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True, ...)
```

The ModularPipeine created with `from_pretrained` method also would not load any components and you would have to call `load_components` to explicitly load components you need.


### Save a `ModularPipeline`

to save a `ModularPipeline` and publish it to hub

```py
pipeline.save_pretrained("YiYiXu/modular-loader-t2i", push_to_hub=True) 
```

<Tip>

We do not automatically save custom code and share it on hub for you, please read more about how to share your custom pipeline on hub [here](TODO: ModularPipeline/CustomCode)






