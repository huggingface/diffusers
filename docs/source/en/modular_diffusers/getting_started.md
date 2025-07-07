<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Getting Started with Modular Diffusers: A Comprehensive Overview

With Modular Diffusers, we introduce a unified pipeline system that simplifies how you work with diffusion models. Instead of creating separate pipelines for each task, Modular Diffusers lets you:

**Write Only What's New**: You won't need to write an entire pipeline from scratch every time you have a new use case. You can create pipeline blocks just for your new workflow's unique aspects and reuse existing blocks for existing functionalities. 

**Assemble Like LEGO®**: You can mix and match between blocks in flexible ways. This allows you to write dedicated blocks unique to specific workflows, and then assemble different blocks into a pipeline that can be used more conveniently for multiple workflows. 

In this guide, we will focus on how to build end-to-end pipelines using blocks we officially support at diffusers 🧨! We will show you how to write your own pipeline blocks and go into more details on how they work under the hood in this [guide](./write_own_pipeline_block.md). For advanced users who want to build complete workflows from scratch, we provide an end-to-end example in the [Developer Guide](./end_to_end.md) that covers everything from writing custom pipeline blocks to deploying your workflow as a UI node.

Let's get started! The Modular Diffusers Framework consists of three main components:
- ModularPipelineBlocks
- PipelineState & BlockState
- ModularPipeline

## ModularPipelineBlocks

Pipeline blocks are the fundamental building blocks of the Modular Diffusers system. All pipeline blocks inherit from the base class `ModularPipelineBlocks`, including:

- [`PipelineBlock`]: The most granular block - you define the computation logic.
- [`SequentialPipelineBlocks`]: A multi-block composed of multiple blocks that run sequentially, passing outputs as inputs to the next block.
- [`LoopSequentialPipelineBlocks`]: A special type of `SequentialPipelineBlocks` that runs the same sequence of blocks multiple times (loops), typically used for iterative processes like denoising steps in diffusion models.
- [`AutoPipelineBlocks`]: A multi-block composed of multiple blocks that are selected at runtime based on the inputs.

All blocks have a consistent interface defining their requirements (components, configs, inputs, outputs) and computation logic. They can be defined standalone or combined into larger blocks - They are designed to be assembled into workflows for tasks such as image generation, video creation, and inpainting. However, blocks aren't runnable on thier own and they need to be converted into a a ModularPipeline to actually run. 

**Blocks vs Pipelines**: Blocks are just definitions - they define what components, inputs/outputs, and computation logics are needed, but they don't actually run anything. To execute blocks, you need to put them into a `ModularPipeline`. See the [ModularPipeline from ModularPipelineBlocks](#modularpipeline-from-modularpipelineblocks) section for how to create and run pipelines.

It is very easy to use a `ModularPipelineBlocks` officially supported in 🧨 Diffusers

```py
from diffusers.modular_pipelines.stable_diffusion_xl import StableDiffusionXLTextEncoderStep

text_encoder_block = StableDiffusionXLTextEncoderStep()
```

This is a single `PipelineBlock`. You'll see that this text encoder block uses 2 text_encoders, 2 tokenizers as well as a guider component. It takes user inputs such as `prompt` and `negative_prompt`, and return text embeddings outputs such as `prompt_embeds` and `negative_prompt_embeds`.

```py
>>> text_encoder_block
StableDiffusionXLTextEncoderStep(
  Class: PipelineBlock
  Description: Text Encoder step that generate text_embeddings to guide the image generation
    Components:
        text_encoder (`CLIPTextModel`)
        text_encoder_2 (`CLIPTextModelWithProjection`)
        tokenizer (`CLIPTokenizer`)
        tokenizer_2 (`CLIPTokenizer`)
        guider (`ClassifierFreeGuidance`)
    Configs:
        force_zeros_for_empty_prompt (default: True)
  Inputs:
    prompt=None, prompt_2=None, negative_prompt=None, negative_prompt_2=None, cross_attention_kwargs=None, clip_skip=None
  Intermediates:
    - outputs: prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
)
```

More commonly, you can create a `SequentialPipelineBlocks` using a block classes preset from 🧨 Diffusers.

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS
t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)
```

This creates a `SequentialPipelineBlocks`, which is a multi-block composed of other blocks. Unlike single blocks (like the `text_encoder_block` we saw earlier), this multi-block has a `sub_blocks` attribute that contains the sub-blocks (text_encoder, input, set_timesteps, prepare_latents, prepare_added_con, denoise, decode). Its requirements for components, inputs, and intermediate inputs are combined from these blocks that compose it. At runtime, it executes its sub-blocks sequentially and passes the pipeline state from one block to another. 

```py
>>> t2i_blocks
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  Description: 


  Components:
      text_encoder (`CLIPTextModel`)
      text_encoder_2 (`CLIPTextModelWithProjection`)
      tokenizer (`CLIPTokenizer`)
      tokenizer_2 (`CLIPTokenizer`)
      guider (`ClassifierFreeGuidance`)
      scheduler (`EulerDiscreteScheduler`)
      unet (`UNet2DConditionModel`)
      vae (`AutoencoderKL`)
      image_processor (`VaeImageProcessor`)

  Configs:
      force_zeros_for_empty_prompt (default: True)

  Sub-Blocks:
    [0] text_encoder (StableDiffusionXLTextEncoderStep)
       Description: Text Encoder step that generate text_embeddings to guide the image generation

    [1] input (StableDiffusionXLInputStep)
       Description: Input processing step that:
                     1. Determines `batch_size` and `dtype` based on `prompt_embeds`
                     2. Adjusts input tensor shapes based on `batch_size` (number of prompts) and `num_images_per_prompt`
                   
                   All input tensors are expected to have either batch_size=1 or match the batch_size
                   of prompt_embeds. The tensors will be duplicated across the batch dimension to
                   have a final batch_size of batch_size * num_images_per_prompt.

    [2] set_timesteps (StableDiffusionXLSetTimestepsStep)
       Description: Step that sets the scheduler's timesteps for inference

    [3] prepare_latents (StableDiffusionXLPrepareLatentsStep)
       Description: Prepare latents step that prepares the latents for the text-to-image generation process

    [4] prepare_add_cond (StableDiffusionXLPrepareAdditionalConditioningStep)
       Description: Step that prepares the additional conditioning for the text-to-image generation process

    [5] denoise (StableDiffusionXLDenoiseStep)
       Description: Denoise step that iteratively denoise the latents. 
                   Its loop logic is defined in `StableDiffusionXLDenoiseLoopWrapper.__call__` method 
                   At each iteration, it runs blocks defined in `sub_blocks` sequencially:
                    - `StableDiffusionXLLoopBeforeDenoiser`
                    - `StableDiffusionXLLoopDenoiser`
                    - `StableDiffusionXLLoopAfterDenoiser`
                   This block supports both text2img and img2img tasks.

    [6] decode (StableDiffusionXLDecodeStep)
       Description: Step that decodes the denoised latents into images

)
```

The block classes preset (`TEXT2IMAGE_BLOCKS`) we used is just a dictionary that maps names to ModularPipelineBlocks classes

```py
>>> TEXT2IMAGE_BLOCKS
InsertableDict([
  0: ('text_encoder', <class 'diffusers.modular_pipelines.stable_diffusion_xl.encoders.StableDiffusionXLTextEncoderStep'>),
  1: ('input', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLInputStep'>),
  2: ('set_timesteps', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLSetTimestepsStep'>),
  3: ('prepare_latents', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareLatentsStep'>),
  4: ('prepare_add_cond', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareAdditionalConditioningStep'>),
  5: ('denoise', <class 'diffusers.modular_pipelines.stable_diffusion_xl.denoise.StableDiffusionXLDenoiseLoop'>),
  6: ('decode', <class 'diffusers.modular_pipelines.stable_diffusion_xl.decoders.StableDiffusionXLDecodeStep'>)
])
```

When we create a `SequentialPipelineBlocks` from this preset, it instantiates each block class into actual block objects. Its `sub_blocks` attribute now contains these instantiated objects:

```py
>>> t2i_blocks.sub_blocks
InsertableDict([
  0: ('text_encoder', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.encoders.StableDiffusionXLTextEncoderStep'>),
  1: ('input', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLInputStep'>),
  2: ('set_timesteps', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLSetTimestepsStep'>),
  3: ('prepare_latents', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareLatentsStep'>),
  4: ('prepare_add_cond', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareAdditionalConditioningStep'>),
  5: ('denoise', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.denoise.StableDiffusionXLDenoiseStep'>),
  6: ('decode', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.decoders.StableDiffusionXLDecodeStep'>)
])
```

Note that both the block classes preset and the `sub_blocks` attribute are `InsertableDict` objects. This is a custom dictionary that extends `OrderedDict` with the ability to insert items at specific positions. You can perform all standard dictionary operations (get, set, delete) plus insert items at any index, which is particularly useful for reordering or inserting blocks in the middle of a pipeline.

**Add a block:**
```py
# Add a block class to the preset
BLOCKS.insert("block_name", BlockClass, index)
# Add a block instance to the `sub_blocks` attribute
t2i_blocks.sub_blocks.insert("block_name", block_instance, index)
```

**Remove a block:**
```py
# remove a block class from preset
BLOCKS.pop("text_encoder")
# split out a block instance on its own
text_encoder_block = t2i_blocks.sub_blocks.pop("text_encoder")
```

**Swap block:**
```py
# Replace block class in preset
BLOCKS["prepare_latents"] = CustomPrepareLatents
# Replace in sub_blocks attribute
t2i_blocks.sub_blocks["prepare_latents"] = CustomPrepareLatents()
```

This means you can mix-and-match blocks in very flexible ways. Let's see some real examples:

**Example 1: Adding IP-Adapter to the Block Classes Preset**
Let's make a new block classes preset by insert IP-Adapter at index 0 (before the text_encoder block), and create a text-to-image pipeline with IP-Adapter support:

```py
from diffusers.modular_pipelines.stable_diffusion_xl import StableDiffusionXLAutoIPAdapterStep
CUSTOM_BLOCKS = TEXT2IMAGE_BLOCKS.copy()
CUSTOM_BLOCKS.insert("ip_adapter", StableDiffusionXLAutoIPAdapterStep, 0)
custom_blocks = SequentialPipelineBlocks.from_blocks_dict(CUSTOM_BLOCKS)
```

**Example 2: Extracting a block from a multi-block**
You can extract a block instance from the multi-block to use it independently. A common pattern is to use text_encoder to process prompts once, then reuse the text embeddings outputs to generate multiple images with different settings (schedulers, seeds, inference steps). We can do this by simply extracting the text_encoder block from the pipeline.

```py
# this gives you StableDiffusionXLTextEncoderStep()
>>> text_encoder_blocks = t2i_blocks.sub_blocks.pop("text_encoder")
>>> text_encoder_blocks
```

The multi-block now has fewer components and no longer has the `text_encoder` block. If you check its docstring `t2i_blocks.doc`, you will see that it no longer accepts `prompt` as input - you will need to pass the embeddings instead.

```py
>>> t2i_blocks
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  Description: 

  Components:
      scheduler (`EulerDiscreteScheduler`)
      guider (`ClassifierFreeGuidance`)
      unet (`UNet2DConditionModel`)
      vae (`AutoencoderKL`)
      image_processor (`VaeImageProcessor`)

  Blocks:
    [0] input (StableDiffusionXLInputStep)
       Description: Input processing step that:
                     1. Determines `batch_size` and `dtype` based on `prompt_embeds`
                     2. Adjusts input tensor shapes based on `batch_size` (number of prompts) and `num_images_per_prompt`
                   
                   All input tensors are expected to have either batch_size=1 or match the batch_size
                   of prompt_embeds. The tensors will be duplicated across the batch dimension to
                   have a final batch_size of batch_size * num_images_per_prompt.

    [1] set_timesteps (StableDiffusionXLSetTimestepsStep)
       Description: Step that sets the scheduler's timesteps for inference

    [2] prepare_latents (StableDiffusionXLPrepareLatentsStep)
       Description: Prepare latents step that prepares the latents for the text-to-image generation process

    [3] prepare_add_cond (StableDiffusionXLPrepareAdditionalConditioningStep)
       Description: Step that prepares the additional conditioning for the text-to-image generation process

    [4] denoise (StableDiffusionXLDenoiseLoop)
       Description: Denoise step that iteratively denoise the latents. 
                   Its loop logic is defined in `StableDiffusionXLDenoiseLoopWrapper.__call__` method 
                   At each iteration, it runs blocks defined in `blocks` sequencially:
                    - `StableDiffusionXLLoopBeforeDenoiser`
                    - `StableDiffusionXLLoopDenoiser`
                    - `StableDiffusionXLLoopAfterDenoiser`
                   

    [5] decode (StableDiffusionXLDecodeStep)
       Description: Step that decodes the denoised latents into images

)
```

<Tip>

💡 You can find all the block classes presets we support for each model in `ALL_BLOCKS`.

```py
# For Stable Diffusion XL
from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS
ALL_BLOCKS
# For other models...
from diffusers.modular_pipelines.<model_name> import ALL_BLOCKS
```

Each model provides a dictionary that maps all supported tasks/techniques to their corresponding block classes presets. For SDXL, it is 

```py
ALL_BLOCKS = {
    "text2img": TEXT2IMAGE_BLOCKS,
    "img2img": IMAGE2IMAGE_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
    "ip_adapter": IP_ADAPTER_BLOCKS,
    "auto": AUTO_BLOCKS,
}
```

</Tip>

We will not go over how to write your own ModularPipelineBlocks but you can learn more about it [here](./write_own_pipeline_block.md).

This covers the essentials of pipeline blocks! You may have noticed that we haven't discussed how to load or run pipeline blocks - that's because **pipeline blocks are not runnable by themselves**. They are essentially **"definitions"** - they define the specifications and computational steps for a pipeline, but they do not contain any model states. To actually run them, you need to convert them into a `ModularPipeline` object.

## PipelineState & BlockState

`PipelineState` and `BlockState` manage dataflow between pipeline blocks. `PipelineState` acts as the global state container that `ModularPipelineBlocks` operate on - each block gets a local view (`BlockState`) of the relevant variables it needs from `PipelineState`, performs its operations, and then updates `PipelineState` as needed.

<Tip>

You typically don't need to manually create or manage these state objects. The `ModularPipeline` automatically creates and manages them for you. However, understanding their roles is important for developing custom pipeline blocks.

</Tip>

## ModularPipeline

`ModularPipeline` is the main interface to create and execute pipelines in the Modular Diffusers system.

### Modular Repo

`ModularPipeline` only works with modular repositories. You can find an example modular repo [here](https://huggingface.co/YiYiXu/modular-diffdiff).

A `DiffusionPipeline` defines `model_index.json` to configure its components. However, repositories for Modular Diffusers work with `modular_model_index.json`. Let's walk through the differences here.

In standard `model_index.json`, each component entry is a `(library, class)` tuple:

"text_encoder": [
  "transformers",
  "CLIPTextModel"
],
```

In `modular_model_index.json`, each component entry contains 3 elements: `(library, class, loading_specs_dict)`

- `library` and `class`: Information about the actual component loaded in the pipeline at the time of saving (will be `null` if not loaded)
- `loading_specs_dict`: A dictionary containing all information required to load this component, including `repo`, `revision`, `subfolder`, `variant`, and `type_hint`. 

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

Unlike standard repositories where components must be in subfolders within the same repo, modular repositories can fetch components from different repositories based on the `loading_specs_dict`. e.g. the `text_encoder` component will be fetched from the "text_encoder" folder in `stabilityai/stable-diffusion-xl-base-1.0` while other components come from different repositories.


### Creating a `ModularPipeline` from `ModularPipelineBlocks`

Each `ModularPipelineBlocks` has an `init_pipeline` method that can initialize a `ModularPipeline` object based on its component and configuration specifications.

Let's convert our `t2i_blocks` (which we created earlier) into a runnable `ModularPipeline`:

```py
# We already have this from earlier
t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

# Now convert it to a ModularPipeline
modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
t2i_pipeline = t2i_blocks.init_pipeline(modular_repo_id)
```

<Tip>

💡 We recommend using `ModularPipeline` with Component Manager by passing a `components_manager`:

```py
>>> components = ComponentsManager()
>>> pipeline = blocks.init_pipeline(modular_repo_id, components_manager=components)
```

This helps you to:
1. Detect and manage duplicated models (warns when trying to register an existing model)
2. Easily reuse components across different pipelines
3. Apply offloading strategies across multiple pipelines

You can read more about Components Manager [here](TODO)

</Tip>


### Creating a `ModularPipeline` with `from_pretrained`

You can create a `ModularPipeline` from a HuggingFace Hub repository with `from_pretrained` method, as long as it's a modular repo:

```py
from diffusers import ModularPipeline
pipeline = ModularPipeline.from_pretrained( "YiYiXu/modular-loader-t2i-0704")
```

Loading custom code is also supported:

```py
from diffusers import ModularPipeline
modular_repo_id = "YiYiXu/modular-diffdiff-0704"
diffdiff_pipeline = ModularPipeline.from_pretrained(modular_repo_id, trust_remote_code=True)
```

This modular repository contains custom code. The [`config.json`](https://huggingface.co/YiYiXu/modular-diffdiff-0704/blob/main/config.json) file defines a custom `DiffDiffBlocks` class and points to its implementation:

```json
{
  "_class_name": "DiffDiffBlocks",
  "auto_map": {
    "ModularPipelineBlocks": "block.DiffDiffBlocks"
  }
}
```

The `auto_map` tells the pipeline where to find the custom blocks definition - in this case, it's looking for `DiffDiffBlocks` in the `block.py` file. The actual `DiffDiffBlocks` class is defined in [`block.py`](https://huggingface.co/YiYiXu/modular-diffdiff-0704/blob/main/block.py) within the repository.

When `diffdiff_pipeline.blocks` is created, it's based on the `DiffDiffBlocks` definition from the custom code in the repository, allowing you to use specialized blocks that aren't part of the standard diffusers library.

### Loading components into a `ModularPipeline`

Unlike `DiffusionPipeline`, when you create a `ModularPipeline` instance (whether using `from_pretrained` or converting from pipeline blocks), its components aren't loaded automatically. You need to explicitly load model components using `load_default_components` or `load_components(names=..,)`:

```py
# This will load ALL the expected components into pipeline
import torch
t2i_pipeline.load_default_components(torch_dtype=torch.float16)
t2i_pipeline.to("cuda")
```

All expected components are now loaded into the pipeline. You can also partially load specific components using the `names` argument. For example, to only load unet and vae:

```py
>>> t2i_pipeline.load_components(names=["unet", "vae"], torch_dtype=torch.float16)
```

You can inspect the pipeline's loading status by simply printing the pipeline itself. It helps you understand what components are expected to load, which ones are already loaded, how they were loaded, and what loading specs are available. Let's print out the `t2i_pipeline`:

```py
>>> t2i_pipeline
StableDiffusionXLModularPipeline {
  "_blocks_class_name": "SequentialPipelineBlocks",
  "_class_name": "StableDiffusionXLModularPipeline",
  "_diffusers_version": "0.35.0.dev0",
  "force_zeros_for_empty_prompt": true,
  "scheduler": [
    null,
    null,
    {
      "repo": "stabilityai/stable-diffusion-xl-base-1.0",
      "revision": null,
      "subfolder": "scheduler",
      "type_hint": [
        "diffusers",
        "EulerDiscreteScheduler"
      ],
      "variant": null
    }
  ],
  "text_encoder": [
    null,
    null,
    {
      "repo": "stabilityai/stable-diffusion-xl-base-1.0",
      "revision": null,
      "subfolder": "text_encoder",
      "type_hint": [
        "transformers",
        "CLIPTextModel"
      ],
      "variant": null
    }
  ],
  "text_encoder_2": [
    null,
    null,
    {
      "repo": "stabilityai/stable-diffusion-xl-base-1.0",
      "revision": null,
      "subfolder": "text_encoder_2",
      "type_hint": [
        "transformers",
        "CLIPTextModelWithProjection"
      ],
      "variant": null
    }
  ],
  "tokenizer": [
    null,
    null,
    {
      "repo": "stabilityai/stable-diffusion-xl-base-1.0",
      "revision": null,
      "subfolder": "tokenizer",
      "type_hint": [
        "transformers",
        "CLIPTokenizer"
      ],
      "variant": null
    }
  ],
  "tokenizer_2": [
    null,
    null,
    {
      "repo": "stabilityai/stable-diffusion-xl-base-1.0",
      "revision": null,
      "subfolder": "tokenizer_2",
      "type_hint": [
        "transformers",
        "CLIPTokenizer"
      ],
      "variant": null
    }
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel",
    {
      "repo": "RunDiffusion/Juggernaut-XL-v9",
      "revision": null,
      "subfolder": "unet",
      "type_hint": [
        "diffusers",
        "UNet2DConditionModel"
      ],
      "variant": "fp16"
    }
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL",
    {
      "repo": "madebyollin/sdxl-vae-fp16-fix",
      "revision": null,
      "subfolder": null,
      "type_hint": [
        "diffusers",
        "AutoencoderKL"
      ],
      "variant": null
    }
  ]
}
```

You can see all the components that will be loaded using `from_pretrained` method are listed as entries. Each entry contains 3 elements: `(library, class, loading_specs_dict)`:

- **`library` and `class`**: Show the actual loaded component info. If `null`, the component is not loaded yet.
- **`loading_specs_dict`**: Contains all the information needed to load the component (repo, subfolder, variant, etc.)

In this example:
- **Loaded components**: `vae` and `unet` (their `library` and `class` fields show the actual loaded models)
- **Not loaded yet**: `scheduler`, `text_encoder`, `text_encoder_2`, `tokenizer`, `tokenizer_2` (their `library` and `class` fields are `null`, but you can see their loading specs to know where they'll be loaded from when you call `load_components()`)

You're looking at essentailly the pipeline's config dict that's synced with the `modular_model_index.json` from the repository you used during `init_pipeline()` - it takes the loading specs that match the pipeline's component requirements.

For example, if your pipeline needs a `text_encoder` component, it will include the loading spec for `text_encoder` from the modular repo during the `init_pipeline`. If the pipeline doesn't need a component (like `controlnet` in a basic text-to-image pipeline), that component won't be included even if it exists in the modular repo.

There are also a few properties that can provide a quick summary of component loading status: 

```py
# All components expected by the pipeline
>>> t2i_pipeline.component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'guider', 'scheduler', 'unet', 'vae', 'image_processor']

# Components that are not loaded yet (will be loaded with from_pretrained)
>>> t2i_pipeline.null_component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']

# Components that will be loaded from pretrained models
>>> t2i_pipeline.pretrained_component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler', 'unet', 'vae']

# Components that are created with default config (no repo needed)
>>> t2i_pipeline.config_component_names
['guider', 'image_processor']
```

### Modifying Loading Specs

When you call `pipeline.load_components(names=)` or `pipeline.load_default_components()`, it uses the loading specs from the modular repository's `modular_model_index.json`. You can change where components are loaded from by default by modifying the `modular_model_index.json` in the repository. You can change any field in the loading specs: `repo`, `subfolder`, `variant`, `revision`, etc.

```py
# Original spec in modular_model_index.json
"unet": [
  null, null,
  {
    "repo": "stabilityai/stable-diffusion-xl-base-1.0",
    "subfolder": "unet",
    "variant": "fp16"
  }
]

# Modified spec - changed repo, subfolder, and variant
"unet": [
  null, null,
  {
    "repo": "RunDiffusion/Juggernaut-XL-v9",
    "subfolder": "unet", 
    "variant": "fp16"
  }
]
```

When you call `pipeline.load_components(...)`/`pipeline.load_default_components()`, it will now load from the new repository by default.


### Updating components in a `ModularPipeline`

Similar to `DiffusionPipeline`, you can load components separately to replace the default ones in the pipeline. In Modular Diffusers, the approach depends on the component type:

- **Pretrained components** (`default_creation_method='from_pretrained'`): Must use `ComponentSpec` to load them, as they get tagged with a unique ID that encodes their loading parameters
- **Config components** (`default_creation_method='from_config'`): These are components that don't need loading specs - they're created during pipeline initialization with default config. To update them, you can either pass the object directly or pass a ComponentSpec directly (which will call `create()` under the hood).

`ComponentSpec` defines how to create or load components and can actually create them using its `create()` method (for ConfigMixin objects) or `load()` method (wrapper around `from_pretrained()`). When a component is loaded with a ComponentSpec, it gets tagged with a unique ID that encodes its creation parameters, allowing you to always extract the original specification using `ComponentSpec.from_component()`.

So instead of 

```py
from diffusers import UNet2DConditionModel
import torch
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", variant="fp16", torch_dtype=torch.float16)
```
You should do

```py
from diffusers import ComponentSpec, UNet2DConditionModel
unet_spec = ComponentSpec(name="unet",type_hint=UNet2DConditionModel, repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", variant="fp16")
unet2 = unet_spec.load(torch_dtype=torch.float16)
```

The key difference is that the second unet (the one we load with `ComponentSpec`) retains its loading specs, so you can extract and recreate it:

```py
# to extract spec, you can do spec.load() to recreate it
>>> spec = ComponentSpec.from_component("unet", unet2)
>>> spec
ComponentSpec(name='unet', type_hint=<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>, description=None, config=None, repo='stabilityai/stable-diffusion-xl-base-1.0', subfolder='unet', variant='fp16', revision=None, default_creation_method='from_pretrained')
```

To replace the unet in the pipeline

```
t2i_pipeline.update_components(unet=unet2)
```

Not only is the `unet` component swapped, but its loading specs are also updated from "RunDiffusion/Juggernaut-XL-v9" to "stabilityai/stable-diffusion-xl-base-1.0". This means that if you save the pipeline now and load it back with `from_pretrained`, the new pipeline will by default load the SDXL original unet.

```
>>> t2i_pipeline
StableDiffusionXLModularPipeline {
  ...
  "unet": [
    "diffusers",
    "UNet2DConditionModel",
    {
      "repo": "stabilityai/stable-diffusion-xl-base-1.0",
      "revision": null,
      "subfolder": "unet",
      "type_hint": [
        "diffusers",
        "UNet2DConditionModel"
      ],
      "variant": "fp16"
    }
  ],
  ...
}  
```
<Tip>

💡 **Modifying Component Specs**: You can get a copy of the current component spec from the pipeline using `get_component_spec()`. This makes it easy to modify the spec and updating components.

```py
>>> unet_spec = t2i_pipeline.get_component_spec("unet")
>>> unet_spec
ComponentSpec(
    name='unet', 
    type_hint=<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>, 
    repo='RunDiffusion/Juggernaut-XL-v9', 
    subfolder='unet', 
    variant='fp16', 
    default_creation_method='from_pretrained'
)

# Modify the spec to load from a different repository
>>> unet_spec.repo = "stabilityai/stable-diffusion-xl-base-1.0"

# Load the component with the modified spec
>>> unet = unet_spec.load(torch_dtype=torch.float16)
```

</Tip>

### Customizing Guidance Techniques

Guiders are guidance techniques that can be applied during the denoising process to improve generation quality, control, and adherence to prompts. They work by modifying the noise predictions or model behavior to steer the generation process in desired directions. In diffusers, guiders are implemented as subclasses of `BaseGuidance` and can be easily integrated into modular pipelines, providing a flexible way to enhance generation quality without modifying the underlying diffusion models.

**ClassifierFreeGuidance (CFG)** is the first and most common guidance technique, used in all our standard pipelines. But we offer many more guidance techniques beyond CFG, including **PerturbedAttentionGuidance (PAG)**, **SkipLayerGuidance (SLG)**, **SmoothedEnergyGuidance (SEG)**, and others that can provide even better results for specific use cases.

This section demonstrates how to use guiders using the component updating methods we just learned. Since `BaseGuidance` components are stateless (similar to schedulers), they are typically created with default configurations during pipeline initialization using `default_creation_method='from_config'`. This means they don't require loading specs from the repository - you won't see guider listed in `modular_model_index.json` files.

Let's take a look at the default guider configuration:

```py
>>> t2i_pipeline.get_component_spec("guider")
ComponentSpec(name='guider', type_hint=<class 'diffusers.guiders.classifier_free_guidance.ClassifierFreeGuidance'>, description=None, config=FrozenDict([('guidance_scale', 7.5), ('guidance_rescale', 0.0), ('use_original_formulation', False), ('start', 0.0), ('stop', 1.0), ('_use_default_values', ['start', 'guidance_rescale', 'stop', 'use_original_formulation'])]), repo=None, subfolder=None, variant=None, revision=None, default_creation_method='from_config')
```

As you can see, the guider is configured to use `ClassifierFreeGuidance` with default parameters and `default_creation_method='from_config'`, meaning it's created during pipeline initialization rather than loaded from a repository. Let's verify this, here we run `init_pipeline()` without a modular repo, and there it is, a guider with the default configuration we just saw


```py
>>> pipeline = t2i_blocks.init_pipeline()
>>> pipeline.guider
ClassifierFreeGuidance {
  "_class_name": "ClassifierFreeGuidance",
  "_diffusers_version": "0.35.0.dev0",
  "guidance_rescale": 0.0,
  "guidance_scale": 7.5,
  "start": 0.0,
  "stop": 1.0,
  "use_original_formulation": false
}
```

#### Modify Parameters of the Same Guider Type

To change parameters of the same guider type (e.g., adjusting the `guidance_scale` for CFG), you have two options:

**Option 1: Use ComponentSpec.create() method**
```python
>>> guider_spec = t2i_pipeline.get_component_spec("guider")
>>> guider = guider_spec.create(guidance_scale=10)
>>> t2i_pipeline.update_components(guider=guider)
```

**Option 2: Pass ComponentSpec directly**
```python
>>> guider_spec = t2i_pipeline.get_component_spec("guider")
>>> guider_spec.config["guidance_scale"] = 10
>>> t2i_pipeline.update_components(guider=guider_spec)
```

Both approaches produce the same result:
```python
>>> t2i_pipeline.guider
ClassifierFreeGuidance {
  "_class_name": "ClassifierFreeGuidance",
  "_diffusers_version": "0.35.0.dev0",
  "guidance_rescale": 0.0,
  "guidance_scale": 10,
  "start": 0.0,
  "stop": 1.0,
  "use_original_formulation": false
}
```

#### Switch to a Different Guider Type

Since guiders are `from_config` components (ConfigMixin objects), you can pass guider objects directly to switch between different guidance techniques:

```py
from diffusers import LayerSkipConfig, PerturbedAttentionGuidance
config = LayerSkipConfig(indices=[2, 9], fqn="mid_block.attentions.0.transformer_blocks", skip_attention=False, skip_attention_scores=True, skip_ff=False)
guider = PerturbedAttentionGuidance(
    guidance_scale=5.0, perturbed_guidance_scale=2.5, perturbed_guidance_config=config
)
t2i_pipeline.update_components(guider=guider)
```

Note that you will get a warning about changing the guider type, which is expected:

```
ModularPipeline.update_components: adding guider with new type: PerturbedAttentionGuidance, previous type: ClassifierFreeGuidance
```

<Tip>

💡 **Component Loading Methods**: 
- For `from_config` components (like guiders, schedulers): You can pass the object directly OR pass a ComponentSpec directly (which calls `create()` under the hood)
- For `from_pretrained` components (like models): You must use ComponentSpec to ensure proper tagging and loading

</Tip>

Let's verify that the guider has been updated:

```py
>>> t2i_pipeline.guider
PerturbedAttentionGuidance {
  "_class_name": "PerturbedAttentionGuidance",
  "_diffusers_version": "0.35.0.dev0",
  "guidance_rescale": 0.0,
  "guidance_scale": 5.0,
  "perturbed_guidance_config": {
    "dropout": 1.0,
    "fqn": "mid_block.attentions.0.transformer_blocks",
    "indices": [
      2,
      9
    ],
    "skip_attention": false,
    "skip_attention_scores": true,
    "skip_ff": false
  },
  "perturbed_guidance_layers": null,
  "perturbed_guidance_scale": 2.5,
  "perturbed_guidance_start": 0.01,
  "perturbed_guidance_stop": 0.2,
  "skip_layer_config": [
    {
      "dropout": 1.0,
      "fqn": "mid_block.attentions.0.transformer_blocks",
      "indices": [
        2,
        9
      ],
      "skip_attention": false,
      "skip_attention_scores": true,
      "skip_ff": false
    }
  ],
  "skip_layer_guidance_layers": null,
  "skip_layer_guidance_scale": 2.5,
  "skip_layer_guidance_start": 0.01,
  "skip_layer_guidance_stop": 0.2,
  "start": 0.0,
  "stop": 1.0,
  "use_original_formulation": false
}

```

The component spec has also been updated to reflect the new guider type:

```py
>>> t2i_pipeline.get_component_spec("guider")
ComponentSpec(name='guider', type_hint=<class 'diffusers.guiders.perturbed_attention_guidance.PerturbedAttentionGuidance'>, description=None, config=FrozenDict([('guidance_scale', 5.0), ('perturbed_guidance_scale', 2.5), ('perturbed_guidance_start', 0.01), ('perturbed_guidance_stop', 0.2), ('perturbed_guidance_layers', None), ('perturbed_guidance_config', LayerSkipConfig(indices=[2, 9], fqn='mid_block.attentions.0.transformer_blocks', skip_attention=False, skip_attention_scores=True, skip_ff=False, dropout=1.0)), ('guidance_rescale', 0.0), ('use_original_formulation', False), ('start', 0.0), ('stop', 1.0), ('_use_default_values', ['use_original_formulation', 'perturbed_guidance_stop', 'stop', 'guidance_rescale', 'start', 'perturbed_guidance_layers', 'perturbed_guidance_start']), ('skip_layer_guidance_scale', 2.5), ('skip_layer_guidance_start', 0.01), ('skip_layer_guidance_stop', 0.2), ('skip_layer_guidance_layers', None), ('skip_layer_config', [LayerSkipConfig(indices=[2, 9], fqn='mid_block.attentions.0.transformer_blocks', skip_attention=False, skip_attention_scores=True, skip_ff=False, dropout=1.0)]), ('_class_name', 'PerturbedAttentionGuidance'), ('_diffusers_version', '0.35.0.dev0')]), repo=None, subfolder=None, variant=None, revision=None, default_creation_method='from_config')
```

However, the "guider" is still not included in the pipeline config and will not be saved into the `modular_model_index.json` since it remains a `from_config` component: 

```py
>>> assert "guider" not in  t2i_pipeline.config
```

#### Upload Custom Guider to Hub for Easy Loading & Sharing

You can upload your customized guider to the Hub so that it can be loaded more easily:

```py
guider.push_to_hub("YiYiXu/modular-loader-t2i-guider", subfolder="pag_guider")
```

Voilà! Now you have a subfolder called `pag_guider` on that repository. Let's change our guider_spec to use `from_pretrained` as the default creation method and update the loading spec to use this subfolder we just created:

```python
guider_spec = t2i_pipeline.get_component_spec("guider")
guider_spec.default_creation_method="from_pretrained"
guider_spec.repo="YiYiXu/modular-loader-t2i-guider"
guider_spec.subfolder="pag_guider"
pag_guider = guider_spec.load()
t2i_pipeline.update_components(guider=pag_guider)
```

You will get a warning about changing the creation method:

```
ModularPipeline.update_components: changing the default_creation_method of guider from from_config to from_pretrained.
```

Now not only the `guider` component and its component_spec are updated, but so is the pipeline config. Let's push it to a new repository:

```py
t2i_pipeline.push_to_hub("YiYiXu/modular-doc-guider")
```

If you check the `modular_model_index.json`, you'll see the guider is now included:

```json
{
  "guider": [
    "diffusers",
    "PerturbedAttentionGuidance",
    {
      "repo": "YiYiXu/modular-loader-t2i-guider",
      "revision": null,
      "subfolder": "pag_guider",
      "type_hint": [
        "diffusers",
        "PerturbedAttentionGuidance"
      ],
      "variant": null
    }
  ]
}
```

Now when you create the pipeline from that repo directly, the `guider` is not automatically loaded anymore (since it's now a `from_pretrained` component), but when you run `load_default_components()`, the PAG guider will be loaded by default:

```py
t2i_pipeline = t2i_blocks.init_pipeline("YiYiXu/modular-doc-guider")
assert t2i_pipeline.guider is None
t2i_pipeline.load_default_components()
t2i_pipeline.guider
```

Of course, you can also directly modify the `modular_model_index.json` to add a loading spec for the guider by pointing to a folder containing the desired guider config.


<Tip>

💡 **Guidance Techniques Summary**: 
- **ClassifierFreeGuidance (CFG)**: The standard choice, best for general use and prompt adherence
- **PerturbedAttentionGuidance (PAG)**: Enhances attention-based features by perturbing attention mechanisms
- **SkipLayerGuidance (SLG)**: Improves structure and anatomy coherence by skipping specific layers
- **SmoothedEnergyGuidance (SEG)**: Helps with energy distribution smoothing
- **AdaptiveProjectedGuidance (APG)**: Adaptive guidance that projects predictions for better quality

Experiment with different techniques and parameters to find what works best for your specific use case!

</Tip>

### Running a `ModularPipeline`

The API to run the `ModularPipeline` is very similar to how you would run a regular `DiffusionPipeline`:

```py
>>> image = pipeline(prompt="a cat", num_inference_steps=15, output="images")[0]
```

There are a few key differences though:
1. You can also pass a `PipelineState` object directly to the pipeline instead of individual arguments
2. If you do not specify the `output` argument, it returns the `PipelineState` object
3. You can pass a list as `output`, e.g. `pipeline(... output=["images", "latents"])` will return a dictionary containing both the generated image and the final denoised latents

Under the hood, `ModularPipeline`'s `__call__` method is a wrapper around the pipeline blocks' `__call__` method: it creates a `PipelineState` object and populates it with user inputs, then returns the output to the user based on the `output` argument. It also ensures that all pipeline-level config and components are exposed to all pipeline blocks by preparing and passing a `components` input.

<Tip>

You can inspect the docstring of a `ModularPipeline` to check what arguments the pipeline accepts and how to specify the `output` you want. It will list all available outputs (basically everything in the intermediate pipeline state) so you can choose from the list.

**Important**: It is important to always check the docstring because arguments can be different from standard pipelines that you're familar with. For example, in Modular Diffusers we standardized controlnet image input as `control_image`, but regular pipelines have inconsistencies over the names, e.g. controlnet text-to-image uses `image` while SDXL controlnet img2img uses `control_image`.

**Note**: The `output` list might be longer than you expected - it includes everything in the intermediate state that you can choose to return. Most of the time, you'll just want `output="images"` or `output="latents"`.

```py
t2i_pipeline.doc
```

</Tip>

#### Text-to-Image, Image-to-Image, and Inpainting

These are minimum inference examples for basic tasks: text-to-image, image-to-image, and inpainting. The process to create different pipelines is the same - only difference is the block classes presets. The inference is also more or less same to standard pipelines, but please always check `.doc` for correct input names and remember to pass `output="images"`.


<hfoptions id="basic-tasks">
<hfoption id="text-to-image">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS

# create pipeline from official blocks preset
blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_default_components(torch_dtype=torch.float16)
pipeline.to("cuda")

# run pipeline, need to pass a "output=images" argument
image = pipeline(prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", output="images")[0]
image.save("modular_t2i_out.png")
```

</hfoption>
<hfoption id="image-to-image">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import IMAGE2IMAGE_BLOCKS

# create pipeline from blocks preset
blocks = SequentialPipelineBlocks.from_blocks_dict(IMAGE2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_default_components(torch_dtype=torch.float16)
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

# create pipeline from blocks preset
blocks = SequentialPipelineBlocks.from_blocks_dict(INPAINT_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_default_components(torch_dtype=torch.float16)
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

#### ControlNet

For ControlNet, we provide one auto block you can place at the `denoise` step. Let's create it and inspect it to see what it tells us. 

<Tip>

💡 **How to explore new tasks**: When you want to figure out how to do a specific task in Modular Diffusers, it is a good idea to start by checking what block classes presets we offer in `ALL_BLOCKS`. Then create the block instance and inspect it - it will show you the required components, description, and sub-blocks. This is crucial for understanding what each block does and what it needs.

</Tip>

```py
>>> from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS
>>> ALL_BLOCKS["controlnet"]
InsertableDict([
  0: ('denoise', <class 'diffusers.modular_pipelines.stable_diffusion_xl.modular_blocks.StableDiffusionXLAutoControlnetStep'>)
])
>>> controlnet_blocks = ALL_BLOCKS["controlnet"]["denoise"]()
>>> controlnet_blocks
StableDiffusionXLAutoControlnetStep(
  Class: SequentialPipelineBlocks

  ====================================================================================================
  This pipeline contains blocks that are selected at runtime based on inputs.
  Trigger Inputs: {'mask', 'control_mode', 'control_image', 'controlnet_cond'}
  Use `get_execution_blocks()` with input names to see selected blocks (e.g. `get_execution_blocks('mask')`).
  ====================================================================================================


  Description: Controlnet auto step that prepare the controlnet input and denoise the latents. It works for both controlnet and controlnet_union and supports text2img, img2img and inpainting tasks. (it should be replace at 'denoise' step)


  Components:
      controlnet (`ControlNetUnionModel`)
      control_image_processor (`VaeImageProcessor`)
      scheduler (`EulerDiscreteScheduler`)
      unet (`UNet2DConditionModel`)
      guider (`ClassifierFreeGuidance`)

  Sub-Blocks:
    [0] controlnet_input (StableDiffusionXLAutoControlNetInputStep)
       Description: Controlnet Input step that prepare the controlnet input.
                   This is an auto pipeline block that works for both controlnet and controlnet_union.
                    (it should be called right before the denoise step) - `StableDiffusionXLControlNetUnionInputStep` is called to prepare the controlnet input when `control_mode` and `control_image` are provided.
                    - `StableDiffusionXLControlNetInputStep` is called to prepare the controlnet input when `control_image` is provided. - if neither `control_mode` nor `control_image` is provided, step will be skipped.

    [1] controlnet_denoise (StableDiffusionXLAutoControlNetDenoiseStep)
       Description: Denoise step that iteratively denoise the latents with controlnet. This is a auto pipeline block that using controlnet for text2img, img2img and inpainting tasks.This block should not be used without a controlnet_cond input - `StableDiffusionXLInpaintControlNetDenoiseStep` (inpaint_controlnet_denoise) is used when mask is provided. - `StableDiffusionXLControlNetDenoiseStep` (controlnet_denoise) is used when mask is not provided but controlnet_cond is provided. - If neither mask nor controlnet_cond are provided, step will be skipped.

)
```

<Tip>

💡 **Auto Blocks**: This is first time we meet a Auto Blocks! `AutoPipelineBlocks` automatically adapt to your inputs by combining multiple workflows with conditional logic. This is why one convenient block can work for all tasks and controlnet types. See the [Auto Blocks Guide](https://huggingface.co/docs/diffusers/modular_diffusers/write_own_pipeline_block#autopipelineblocks) for more details.

</Tip>

The block shows us it has two steps (prepare inputs + denoise) and supports all tasks with both controlnet and controlnet union. Most importantly, it tells us to place it at the 'denoise' step. Let's do exactly that:

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS, StableDiffusionXLAutoControlnetStep
from diffusers.utils import load_image

# create pipeline from blocks preset
blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

# these two lines applies controlnet
controlnet_blocks = StableDiffusionXLAutoControlnetStep()
blocks.sub_blocks["denoise"] = controlnet_blocks 
```

Before we convert the blocks into a pipeline and load its components, let's inspect the blocks and its docs again to make sure it was assembled correctly. You should be able to see that `controlnet` and `control_image_processor` are now listed as `Components`, so we should initialize the pipeline with a repo that contains desired loading specs for these 2 components.

```py
# make sure to a modular_repo including controlnet
modular_repo_id = "YiYiXu/modular-demo-auto"
pipeline = blocks.init_pipeline(modular_repo_id)
pipeline.load_default_components(torch_dtype=torch.float16)
pipeline.to("cuda")

# generate
canny_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
)
image = pipeline(
    prompt="a bird", controlnet_conditioning_scale=0.5, control_image=canny_image, output="images"
)[0]
image.save("modular_control_out.png")
```

#### IP-Adapter

**Challenge time!** Before we show you how to apply IP-adapter, try doing it yourself! Use the same process we just walked you through with ControlNet: check the official blocks preset, inspect the block instance and docstring `.doc`, and adapt a regular IP-adapter example to modular.

Let's walk through the steps:

1. Check blocks preset

```py
>>> from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS
>>> ALL_BLOCKS["ip_adapter"]
InsertableDict([
  0: ('ip_adapter', <class 'diffusers.modular_pipelines.stable_diffusion_xl.modular_blocks.StableDiffusionXLAutoIPAdapterStep'>)
])
```

2. inspect the block & doc

```
>>> from diffusers.modular_pipelines.stable_diffusion_xl import StableDiffusionXLAutoIPAdapterStep
>>> ip_adapter_blocks = StableDiffusionXLAutoIPAdapterStep()
>>> ip_adapter_blocks
StableDiffusionXLAutoIPAdapterStep(
  Class: AutoPipelineBlocks

  ====================================================================================================
  This pipeline contains blocks that are selected at runtime based on inputs.
  Trigger Inputs: {'ip_adapter_image'}
  Use `get_execution_blocks()` with input names to see selected blocks (e.g. `get_execution_blocks('ip_adapter_image')`).
  ====================================================================================================


  Description: Run IP Adapter step if `ip_adapter_image` is provided. This step should be placed before the 'input' step.
      


  Components:
      image_encoder (`CLIPVisionModelWithProjection`)
      feature_extractor (`CLIPImageProcessor`)
      unet (`UNet2DConditionModel`)
      guider (`ClassifierFreeGuidance`)

  Sub-Blocks:
    • ip_adapter [trigger: ip_adapter_image] (StableDiffusionXLIPAdapterStep)
       Description: IP Adapter step that prepares ip adapter image embeddings.
                   Note that this step only prepares the embeddings - in order for it to work correctly, you need to load ip adapter weights into unet via ModularPipeline.load_ip_adapter() and pipeline.set_ip_adapter_scale().
                   See [ModularIPAdapterMixin](https://huggingface.co/docs/diffusers/api/loaders/ip_adapter#diffusers.loaders.ModularIPAdapterMixin) for more details

)
```
3. follow the instruction to build

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS

# create pipeline from official blocks preset
blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

# insert ip_adapter_blocks before the input step as instructed
blocks.sub_blocks.insert("ip_adapter", ip_adapter_blocks, 1)

# inspec the blocks before you convert it into pipelines,
# and make sure to use a repo that contains the loading spec for all components
# for ip-adapter, you need image_encoder & feature_extractor
modular_repo_id = "YiYiXu/modular-demo-auto"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_default_components(torch_dtype=torch.float16)
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter_sdxl.bin"
)
pipeline.set_ip_adapter_scale(0.8)
pipeline.to("cuda")
```

4. adapt an example to modular

We are using [this one](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter?ipadapter-variants=IP-Adapter+Plus#ip-adapter) from our IP-Adapter doc!


```py
from diffusers.utils import load_image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png")
image = pipeline(
    prompt="a polar bear sitting in a chair drinking a milkshake",
    ip_adapter_image=image,
    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    output="images"
)[0]
image.save("modular_ipa_out.png")
```

## Building Advanced Workflows: The Modular Way

We've learned the basic components of the Modular Diffusers System. Now let's tie everything together with more practical example that demonstrates the true power of Modular Diffusers: working between with multiple pipelines that can share components. 

In this example, we'll generate latents from a text-to-image pipeline, then refine them with an image-to-image pipeline. We will use IP-adapter, LoRA, and ControlNet.

### Base Text-to-Image

Let's setup the text-to-image workflow. Instead of putting all blocks into one complete pipeline, we'll create separate `text_blocks` for encoding prompts, `t2i_blocks` for generating latents, and `decoder_blocks` for creating final images.


```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS

# create t2i blocks and then pop out the text_encoder step and decoder step so that we can use them in standalone manner
t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["text2img"])
text_blocks = t2i_blocks.sub_blocks.pop("text_encoder")
decoder_blocks = t2i_blocks.sub_blocks.pop("decode")
```

Next, convert them into runnable pipelines. We'll use a Components Manager with auto offloading strategy.

**Components Manager**: Create one manager and pass it to `init_pipeline` along with a collection name. All models loaded by that pipeline will be added to the manager under that collection.

**Auto Offloading**: All components are placed on CPU and only moved to device right before their forward pass. The manager monitors device memory and may move components off-device to make space for new ones. Unlike `DiffusionPipeline.enable_model_cpu_offload()`, this works across all components in the manager and all your workflows.


```py
from diffusers import ComponentsManager
# Set up component manager and turn on the offloading
components = ComponentsManager()
components.enable_auto_cpu_offload(device="cuda")
```

Since we have a modular setup where different pipelines may share components, we recommend using a seperate `ModularPipeline` to load components all at once and add them to each pipeline with `update_components()`.


```py
from diffusers import ModularPipeline
t2i_repo = "YiYiXu/modular-demo-auto"
t2i_loader_pipe = ModularPipeline.from_pretrained(t2i_repo, components_manager=components, collection="t2i")

text_node = text_blocks.init_pipeline(t2i_repo, components_manager=components)
decoder_node = decoder_blocks.init_pipeline(t2i_repo, components_manager=components)
t2i_pipe = t2i_blocks.init_pipeline(t2i_repo, components_manager=components)
```

We'll load components in `t2i_loader_pipe`. You can get the list of all loadable components from loader's `pretrained_component_names` property.

```py
>>> t2i_loader_pipe.pretrained_component_names
['controlnet', 'image_encoder', 'scheduler', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'vae']
```

It include controlnet and image_encoder for ip-adapter that we don't need now. But I'll load them anyway since they'll stay on CPU and I might use them later. But you can choose what to load in the `names` argument.

```py
import torch
# inspect before you load
# t2i_loader
t2i_loader_pipe.load_components(names=t2i_loader_pipe.pretrained_component_names, torch_dtype=torch.float16)
```
All the models are registered to components manager under the collection "t2i".

```py
>>> components
Components:
============================================================================================================================================================
Models:
------------------------------------------------------------------------------------------------------------------------------------------------------------
Name           | Class                        | Device: act(exec)| Dtype        | Size (GB)| Load ID                                            | Collection
------------------------------------------------------------------------------------------------------------------------------------------------------------
vae            | AutoencoderKL                | cpu(cuda:0)      | torch.float16| 0.16     | SG161222/RealVisXL_V4.0|vae|null|null              | t2i
image_encoder  | CLIPVisionModelWithProjection| cpu(cuda:0)      | torch.float16| 3.44     | h94/IP-Adapter|sdxl_models/image_encoder|null|null | t2i
text_encoder   | CLIPTextModel                | cpu(cuda:0)      | torch.float16| 0.23     | SG161222/RealVisXL_V4.0|text_encoder|null|null     | t2i
unet           | UNet2DConditionModel         | cpu(cuda:0)      | torch.float16| 4.78     | SG161222/RealVisXL_V4.0|unet|null|null             | t2i
text_encoder_2 | CLIPTextModelWithProjection  | cpu(cuda:0)      | torch.float16| 1.29     | SG161222/RealVisXL_V4.0|text_encoder_2|null|null   | t2i
controlnet     | ControlNetModel              | cpu(cuda:0)      | torch.float16| 2.33     | diffusers/controlnet-canny-sdxl-1.0|null|null|null | t2i
------------------------------------------------------------------------------------------------------------------------------------------------------------

Other Components:
------------------------------------------------------------------------------------------------------------------------------------------------------------
Name           | Class                        | Collection
------------------------------------------------------------------------------------------------------------------------------------------------------------
tokenizer_2    | CLIPTokenizer                | t2i
tokenizer      | CLIPTokenizer                | t2i
scheduler      | EulerDiscreteScheduler       | t2i
------------------------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```

Let's add the loaded components to each pipeline. We'll follow this pattern for each pipeline:
1. Check what components the pipeline needs: inspect `pipeline` or use `pipeline.null_component_names`
2. Get them from the components manager: use its `search_models()`/`get_one`/`get_components_from_names` method
3. Update the pipeline: `pipeline.update_components()`
4. Verify the components are loaded correctly: inspect `pipeline` as well as components manager

We will start with `decoder_node`. First, check what components it needs:

```py
>>> decoder_node.null_component_names
['vae']
```
The pipeline only needs a `vae`. Looking at the components manager table, there's only one VAE available:

```
Name | Class        | Device: act(exec)| Dtype        | Size (GB)| Load ID                               | Collection
----------------------------------------------------------------------------------------------------------------------
vae  | AutoencoderKL| cpu(cuda:0)      | torch.float16| 0.16     | SG161222/RealVisXL_V4.0|vae|null|null | t2i
```
Since there's only one VAE, we can get it using its unique Load ID:

```py
vae = components.get_one(load_id="SG161222/RealVisXL_V4.0|vae|null|null")
decoder_node.update_components(vae=vae)
```

Verify it's correctly loaded:

```py
decoder_node
```
Now let's do the same for `text_node`. Get the list of components the pipeline needs to load:

```py
>>> text_node.null_component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']
```
Pass the list directly to the components manager to get the components and add it to the pipeline

```py
text_components = components.get_components_by_names(text_node.null_component_names)
# Add components to pipeline
text_node.update_components(**text_components)

# Verify components are loaded
assert not text_node.null_component_names
text_node
```

Finally, let's set up `t2i_pipe`:

```py

# Get unet & scheduler from components manager and add to pipeline
comps = components.get_components_by_names(t2i_pipe.null_component_names)
t2i_pipe.update_components(**comps)

# Verify everything is loaded
assert not t2i_pipe.null_component_names
t2i_pipe

# Verify components manager hasn't changed (we only reused existing components)
components
```

We can start to generate an image with the t2i pipeline.

First to run the prompt through text_node to get prompt embeddings

<Tip>

💡 don't forget to `text_node.doc` to find out what outputs are available and set the `output` argument accordingly

</Tip>

```py
prompt = "an astronaut"
text_embeddings = text_node(prompt=prompt, output=["prompt_embeds","negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"])
```

Now generate latents with t2i pipeline and then decode with decoder.


```py
generator = torch.Generator(device="cuda").manual_seed(0)
latents_t2i = t2i_pipe(**text_embeddings, num_inference_steps=25, generator=generator, output="latents")
image = decoder_node(latents=latents_t2i, output="images")[0]
image.save("modular_part2_t2i.png")

```

### Lora

Now let's add a LoRA to our pipeline. With the modular approach we will be able to reuse intermediate outputs from blocks that otherwise needs to be re-run. Let's load the LoRA weights and see what happens:

```py
t2i_loader_pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy_face")
components
```
Notice that the "Additional Component Info" section shows that only the `unet` component has the LoRA adapter loaded. This means we can skip the text encoding step and reuse the existing embeddings, making the generation much faster.

```out
Components:
============================================================================================================================================================
...
Additional Component Info:
==================================================

unet:
  Adapters: ['toy_face']
```


<Tip>

🔍 Alternatively, you can find a component's ID and then use `get_model_info` to get detailed metadata about that component:

```py
id = components.get_ids("unet")[0]
components.get_model_info(id)
# {'model_id': 'unet_6c2b839d-ec39-4ce9-8741-333ba6d25932', 'added_time': 1751101289.203884, 'collection': 't2i', 'class_name': 'UNet2DConditionModel', 'size_gb': 4.940812595188618, 'adapters': ['toy_face'], 'has_hook': True, 'execution_device': device(type='cuda', index=0)}
```
</Tip>


```py
generator = torch.Generator(device="cuda").manual_seed(0)
latents_lora = t2i_pipe(**text_embeddings, num_inference_steps=25, generator=generator, output="latents")
image = decoder_node(latents=latents_lora, output="images")[0]
image.save("modular_part2_lora.png")
```

### IP-adapter 

IP-adapter can also be used as a standalone pipeline. We can generate the embeddings once and reuse them for different workflows.

```py
from diffusers.utils import load_image

ipa_blocks = ALL_BLOCKS["ip_adapter"]["ip_adapter"]()
ipa_node = ipa_blocks.init_pipeline(t2i_repo, components_manager=components)
comps = components.get_components_by_names(ipa_node.loader.null_component_names)
ipa_node.update_components(**comps)

t2i_loader_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
t2i_loader_pipe.set_ip_adapter_scale(0.6)

# check it's correctly loaded
assert not ipa_node.null_component_names
ipa_node
# find out inputs/outputs 
print(ipa_node.doc)

ip_adapter_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy/img5.png")
ipa_embeddings = ipa_node(ip_adapter_image=ip_adapter_image, output=["ip_adapter_embeds","negative_ip_adapter_embeds"])

generator = torch.Generator(device="cuda").manual_seed(0)
latents_ipa = t2i_pipe(**text_embeddings, **ipa_embeddings, num_inference_steps=25, generator=generator, output="latents")

image = decoder_node(latents=latents_ipa, output="images")[0]
image.save("modular_part2_lora_ipa.png")
```

### ControlNet

We can create a new ControlNet workflow by modifying the pipeline blocks, reusing components as much as possible, and see how it affects the generation.

We want to use a different ControlNet from the one that's already loaded.

```py
from diffusers import ComponentSpec, ControlNetModel
control_blocks = ALL_BLOCKS["controlnet"]["denoise"]()
# update the t2i_blocks and create pipeline
t2i_blocks.sub_blocks["denoise"] = control_blocks
t2i_control_pipe = t2i_blocks.init_pipeline(t2i_repo, components_manager=components)

# fetch the controlnet_pose seperately since we need to change name when adding it to the pipeline
controlnet_spec = ComponentSpec(name="controlnet_pose", type_hint=ControlNetModel, repo="thibaud/controlnet-openpose-sdxl-1.0")
controlnet = controlnet_spec.load(torch_dtype=torch.float16)
t2i_control_pipe.update_components(controlnet=controlnet)

# fetch the rest of the components from the components manager
comps = components.get_components_by_names(t2i_control_pipe.loader.null_component_names)
t2i_control_pipe.update_components(**comps)

control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/person_pose.png")
generator = torch.Generator(device="cuda").manual_seed(0)
latents_control = t2i_control_pipe(**text_embeddings, **ipa_embeddings, control_image=control_image, num_inference_steps=25, generator=generator, output="latents")

image = decoder_node(latents=latents_control, output="images")[0]
image.save("modular_part2_lora_ipa_control.png")
```


Now set up refiner workflow. For refiner blocks, we removed `image_encoder` since the refiner works with latents directly, and `decoder` since we already have a dedicated one. We keep `text_encoder` because SDXL refiner encodes text prompts differently from the text-to-image pipeline, so we cannot share it.

```py
# Create a refiner blocks
# - removing image_encoder a since we'll use latents from t2i
# - removing decode since we already created a seperate decoder_block
refiner_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["img2img"])
refiner_blocks.sub_blocks.pop("image_encoder")
refiner_blocks.sub_blocks.pop("decode")
```

### Refiner

Create refiner pipeline. refiner has a different unet and use only one text_encoder so it is hosted in a different repo. We pass the same components manager to refiner pipeline, along with a unique "refiner" collection.

```py
refiner_repo = "YiYiXu/modular_refiner"
refiner_pipe = refiner_blocks.init_pipeline(refiner_repo, components_manager=components, collection="refiner")
```


We want to reuse components from the t2i pipeline in the refiner as much as possible. First, let's check the loading status of the refiner pipeline to understand what components are needed:

```py
>>> refiner_pipe
```

Looking at the loader output, you can see that `text_encoder` and `tokenizer` have empty loading spec maps (their `repo` fields are `null`), this is because refiner pipeline does not use these two components so they are not listed in the `modular_model_index.json` in `refiner_repo`. The `unet` is different from the one we loaded for text-to-image. The remaining components: `vae`, `text_encoder_2`, `tokenizer_2`, and `scheduler` are already available in the t2i collection, we can reuse them instead of loading duplicates.

```py
refiner_pipe.load_components(names="unet", torch_dtype=torch.float16)

# verify loaded correctly
refiner_pipe

# veryfiy registered to components manager under refiner
components
```

Now let's reuse the components from the t2i pipeline in the refiner. We use the`|` to select multiple components from components manager at once:

```py
# Reuse components from t2i pipeline (select everything at once)
reuse_components = components.search_components("text_encoder_2|scheduler|vae|tokenizer_2")
refiner_pipe.update_components(**reuse_components)
```

You'll see warnings indicating that these components already exist in the components manager:

```out
component 'text_encoder_2' already exists as 'text_encoder_2_238ae9a7-c864-4837-a8a2-f58ed753b2d0'
component 'tokenizer_2' already exists as 'tokenizer_2_b795af3d-f048-4b07-a770-9e8237a2be2d'
component 'scheduler' already exists as 'scheduler_e3435f63-266a-4427-9383-eb812e830fe8'
component 'vae' already exists as 'vae_357eee6a-4a06-46f1-be83-494f7d60ca69'
```

These warnings are expected and indicate that the components manager is correctly identifying that these components are already loaded. The system will reuse the existing components rather than creating duplicates.

Let's check the components manager again to see the updated state. You should see `text_encoder_2`, `vae`, `tokenizer_2`, and `scheduler` now appear under both "t2i" and "refiner" collections.

Now let's refine! 

```py
# refine the latents from base text-to-image workflow
refined_latents = refiner_pipe(image_latents=latents_t2i, prompt=prompt, num_inference_steps=10, output="latents")
refined_image = decoder_node(latents=refined_latents, output="images")[0]
refined_image.save("modular_part2_t2i_refine_out.png")

# refine the latents from the text-to-image lora workflow
refined_latents = refiner_pipe(image_latents=latents_lora, prompt=prompt, num_inference_steps=10, output="latents")
refined_image = decoder_node(latents=refined_latents, output="images")[0]
refined_image.save("modular_part2_lora_refine_out.png")

# refine the latents from the text-to-image + lora + ip-adapter workflow
refined_latents = refiner_pipe(image_latents=latents_ipa, prompt=prompt, num_inference_steps=10, output="latents")
refined_image = decoder_node(latents=refined_latents, output="images")[0]
refined_image.save("modular_part2_ipa_refine_out.png")

# refine the latents from the text-to-image + lora + ip-adapter + controlnet workflow
refined_latents = refiner_pipe(image_latents=latents_control, prompt=prompt, num_inference_steps=10, output="latents")
refined_image = decoder_node(latents=refined_latents, output="images")[0]
refined_image.save("modular_part2_control_refine_out.png")
```


### Results

Here are the results from our modular pipeline examples.

#### Base Text-to-Image Generation
| Base Text-to-Image | Base Text-to-Image (Refined) |
|-------------------|------------------------------|
| ![Base T2I](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_t2i.png) | ![Base T2I Refined](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_t2i_refine_out.png) |

#### LoRA
| LoRA              | LoRA               (Refined) |
|-------------------|------------------------------|
| ![LoRA](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_lora.png) | ![LoRA Refined](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_lora_refine_out.png) |

#### LoRA + IP-Adapter
| LoRA + IP-Adapter | LoRA + IP-Adapter (Refined) |
|-------------------|------------------------------|
| ![LoRA + IP-Adapter](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_ipa.png) | ![LoRA + IP-Adapter Refined](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_ipa_refine_out.png) |

#### ControlNet + LoRA + IP-Adapter
| ControlNet + LoRA + IP-Adapter | ControlNet + LoRA + IP-Adapter (Refined) |
|-------------------|------------------------------|
| ![ControlNet + LoRA + IP-Adapter](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_control.png) | ![ControlNet + LoRA + IP-Adapter Refined](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_control_refine_out.png) |


