<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Getting Started with Modular Diffusers

With Modular Diffusers, we introduce a unified pipeline system that simplifies how you work with diffusion models. Instead of creating separate pipelines for each task, Modular Diffusers let you:

**Write Only What's New**: You won't need to rewrite the entire pipeline from scratch. You can create pipeline blocks just for your new workflow's unique aspects and reuse existing blocks for existing functionalities. 

**Assemble Like LEGO®**: You can mix and match blocks in flexible ways. This allows you to write dedicated blocks for specific workflows, and then assemble different blocks into a pipeline that that can be used more conveniently for multiple workflows. 

In this guide, we will focus on how to use pipeline like this we built with Modular diffusers 🧨! We will also go over the basics of pipeline blocks, how they work under the hood, and how to assemble SequentialPipelineBlocks and AutoPipelineBlocks in this [guide](TODO). For advanced users who want to build complete workflows from scratch, we provide an end-to-end example in the [Developer Guide](developer_guide.md) that covers everything from writing custom pipeline blocks to deploying your workflow as a UI node.

Let's get started! The Modular Diffusers Framework consists of three main components:

## ModularPipelineBlocks

Pipeline blocks are the fundamental building blocks of the Modular Diffusers system. All pipeline blocks inherit from the base class `ModularPipelineBlocks`, including:
- [`PipelineBlock`](TODO)
- [`SequentialPipelineBlocks`](TODO)
- [`LoopSequentialPipelineBlocks`](TODO)
- [`AutoPipelineBlocks`](TODO)


To use a `ModularPipelineBlocks` officially supported in 🧨 Diffusers
```py
>>> from diffusers.modular_pipelines.stable_diffusion_xl import StableDiffusionXLTextEncoderStep
>>> text_encoder_block = StableDiffusionXLTextEncoderStep()
```

Each [`ModularPipelineBlocks`] defines its requirement for components, configs, inputs, intermediate inputs, and outputs. You'll see that this text encoder block uses 2 text_encoders, 2 tokenizers as well as a guider component. It takes user inputs such as `prompt` and `negative_prompt`, and return text embeddings such as `prompt_embeds` and `negative_prompt_embeds`.

```
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

More commonly, you can create a `SequentialPipelineBlocks` using a modular blocks preset officially supported in 🧨 Diffusers.


```py
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS
t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)
```

This creates a text-to-image pipeline. 

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

  Blocks:
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

    [5] denoise (StableDiffusionXLDenoiseLoop)
       Description: Denoise step that iteratively denoise the latents. 
                   Its loop logic is defined in `StableDiffusionXLDenoiseLoopWrapper.__call__` method 
                   At each iteration, it runs blocks defined in `blocks` sequencially:
                    - `StableDiffusionXLLoopBeforeDenoiser`
                    - `StableDiffusionXLLoopDenoiser`
                    - `StableDiffusionXLLoopAfterDenoiser`
                   

    [6] decode (StableDiffusionXLDecodeStep)
       Description: Step that decodes the denoised latents into images

)
```

The blocks preset we used (`TEXT2IMAGE_BLOCKS`) is just a dictionary that maps names to ModularPipelineBlocks classes

```py
>>> TEXT2IMAGE_BLOCKS
InsertableOrderedDict([
  0: ('text_encoder', <class 'diffusers.modular_pipelines.stable_diffusion_xl.encoders.StableDiffusionXLTextEncoderStep'>),
  1: ('input', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLInputStep'>),
  2: ('set_timesteps', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLSetTimestepsStep'>),
  3: ('prepare_latents', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareLatentsStep'>),
  4: ('prepare_add_cond', <class 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareAdditionalConditioningStep'>),
  5: ('denoise', <class 'diffusers.modular_pipelines.stable_diffusion_xl.denoise.StableDiffusionXLDenoiseLoop'>),
  6: ('decode', <class 'diffusers.modular_pipelines.stable_diffusion_xl.decoders.StableDiffusionXLDecodeStep'>)
])
```

When we create a `SequentialPipelineBlocks` from this preset, it instantiates each class into actual block objects. Its `blocks` attribute contains these instantiated objects:

```py
>>> t2i_blocks.blocks
InsertableOrderedDict([
  0: ('text_encoder', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.encoders.StableDiffusionXLTextEncoderStep'>),
  1: ('input', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLInputStep'>),
  2: ('set_timesteps', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLSetTimestepsStep'>),
  3: ('prepare_latents', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareLatentsStep'>),
  4: ('prepare_add_cond', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.before_denoise.StableDiffusionXLPrepareAdditionalConditioningStep'>),
  5: ('denoise', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.denoise.StableDiffusionXLDenoiseLoop'>),
  6: ('decode', <obj 'diffusers.modular_pipelines.stable_diffusion_xl.decoders.StableDiffusionXLDecodeStep'>)
])
```

Note that both the preset and the `blocks` attribute are `InsertableOrderedDict` objects, which allows you to modify them in several ways:

**Add a block/block_class at specific positions:**
```py
# Add to preset (class)
BLOCKS.insert("block_name", BlockClass, index)
# Add to blocks attribute (instance)
t2i_blocks.blocks.insert("block_name", block_instance, index)
```

**Remove blocks:**
```py
# remove a block class from preset
BLOCKS.pop("text_encoder")
# split out a block instance on its own
text_encoder_block = t2i_blocks.blocks.pop("text_encoder")
```

**Swap/replace blocks:**
```py
# Replace in preset (class)
BLOCKS["prepare_latents"] = CustomPrepareLatents
# Replace in blocks attribute (instance)
t2i_blocks.blocks["prepare_latents"] = CustomPrepareLatents()
```

This means you can mix-and-match blocks in very flexible ways. Let's see some real examples:

**Example 1: Adding IP-Adapter to the preset**
Let's insert IP-Adapter at index 0 (before the text_encoder block) to create a text-to-image pipeline with IP-Adapter support:

```py
from diffusers.modular_pipelines.stable_diffusion_xl import StableDiffusionXLAutoIPAdapterStep
CUSTOM_BLOCKS = TEXT2IMAGE_BLOCKS.copy()
CUSTOM_BLOCKS.insert("ip_adapter", StableDiffusionXLAutoIPAdapterStep, 0)
custom_blocks = SequentialPipelineBlocks.from_blocks_dict(CUSTOM_BLOCKS)
```

**Example 2: Extracting a block from the pipeline**
You can extract a block instance from the pipeline to use it independently. A common pattern is to extract the text_encoder to process prompts once, then reuse the text embeddings to generate multiple images with different settings (schedulers, seeds, inference steps).

```py
>>> text_encoder_blocks = t2i_blocks.blocks.pop("text_encoder")
>>> text_encoder_blocks
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

the pipeline now has fewer components and no longer has the `text_encoder` block:

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

We will not go over how to write your own ModularPipelineBlocks but you can learn more about it [here](TODO).

This covers the essentials of pipeline blocks! You may have noticed that we haven't discussed how to load or run pipeline blocks - that's because **pipeline blocks are not runnable by themselves**. They are essentially **"definitions"** - they define the specifications and computational steps for a pipeline, but they do not contain any model states. To actually run them, you need to convert them into a `ModularPipeline` object.

## PipelineState & BlockState

`PipelineState` and `BlockState` manage dataflow between pipeline blocks. `PipelineState` acts as the global state container that `ModularPipelineBlocks` operate on - each block gets a local view (`BlockState`) of the relevant variables it needs from `PipelineState`, performs its operations, and then updates PipelineState with any changes.

<Tip>

You typically don't need to manually create or manage these state objects. The `ModularPipeline` automatically creates and manages them for you. However, understanding their roles is important for developing custom pipeline blocks.

</Tip>

## ModularPipeline

`ModularPipeline` is the main interface to create and execute pipelines in the Modular Diffusers system.

### Modular Repo

`ModularPipeline` only works with modular repositories. You can find an example modular repo [here](https://huggingface.co/YiYiXu/modular-diffdiff).

Instead of using a `model_index.json` to configure components loading in `DiffusionPipeline`. Modular repositories work with `modular_model_index.json`. Let's walk through the difference here.

In standard `model_index.json`, each component entry is a `(library, class)` tuple:

```py
"text_encoder": [
  "transformers",
  "CLIPTextModel"
],
```

In `modular_model_index.json`, each component entry contains 3 elements: `(library, class, loading_specs {})`

- `library` and `class`: Information about the actual component loaded in the pipeline at the time of saving (can be `None` if not loaded)
- `loading_specs`: A dictionary containing all information required to load this component, including `repo`, `revision`, `subfolder`, `variant`, and `type_hint`

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

Unlike standard repositories where components must be in subfolders within the same repo, modular repositories can fetch components from different repositories based on the `loading_specs` dictionary. e.g. the `text_encoder` component will be fetched from the "text_encoder" folder in `stabilityai/stable-diffusion-xl-base-1.0` while other components come from different repositories.


### Creating a `ModularPipeline` from `ModularPipelineBlocks`

Each `ModularPipelineBlocks` has an `init_pipeline` method that can initialize a `ModularPipeline` object based on its component and configuration specifications.

Let's convert our `t2i_blocks` (which we created earlier) into a runnable `ModularPipeline`:

```py
# We already have this from earlier
t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

# Now convert it to a ModularPipeline
modular_repo_id = "YiYiXu/modular-loader-t2i"
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
# YiYi TODO: this is not yet supported actually 😢, need to add support
from diffusers import ModularPipeline
pipeline = ModularPipeline.from_pretrained(repo_id, components_manager=..., collection=...)
```

Loading custom code is also supported:

```py
from diffusers import ModularPipeline
modular_repo_id = "YiYiXu/modular-diffdiff"
diffdiff_pipeline = ModularPipeline.from_pretrained(modular_repo_id, trust_remote_code=True)
```

### Loading components into a `ModularPipeline`

Unlike `DiffusionPipeline`, when you create a `ModularPipeline` instance (whether using `from_pretrained` or converting from pipeline blocks), its components aren't loaded automatically. You need to explicitly load model components using `load_components`:

```py
# This will load ALL the expected components into pipeline
t2i_pipeline.load_components(torch_dtype=torch.float16)
t2i_pipeline.to(device)
```

All expected components are now loaded into the pipeline. You can also partially load specific components using the `component_names` argument. For example, to only load unet and vae:

```py
>>> t2i_pipeline.load_components(component_names=["unet", "vae"])
```

You can inspect the pipeline's loading status through its `loader` attribute to understand what components are expected to load, which ones are already loaded, how they were loaded, and what loading specs are available. It has the same structure as the `modular_model_index.json` we discussed earlier - each component entry contains the `(library, class, loading_specs)` format. You'll need to understand that structure to properly read the loading status below.

Let's inspect the `t2i_pipeline`, you can see all the components expected to load are listed as entries in the loader. The `guider` and `image_processor` components were created using default config (their `library` and `class` field are populated, this means they are initialized, but `loading_spec["repo"]` is null). The `vae` and `unet` components were loaded using their respective loading specs. The rest of the components (scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2) are not loaded yet (their `library`, `class` fields are `null`), but you can examine their loading specs to see where they would be loaded from when you call `load_components()`.

```py
>>> t2i_pipeline.loader
StableDiffusionXLModularLoader {
  "_class_name": "StableDiffusionXLModularLoader",
  "_diffusers_version": "0.34.0.dev0",
  "force_zeros_for_empty_prompt": true,
  "guider": [
    "diffusers",
    "ClassifierFreeGuidance",
    {
      "repo": null,
      "revision": null,
      "subfolder": null,
      "type_hint": [
        "diffusers",
        "ClassifierFreeGuidance"
      ],
      "variant": null
    }
  ],
  "image_processor": [
    "diffusers",
    "VaeImageProcessor",
    {
      "repo": null,
      "revision": null,
      "subfolder": null,
      "type_hint": [
        "diffusers",
        "VaeImageProcessor"
      ],
      "variant": null
    }
  ],
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
### Updating components in a `ModularPipeline`

Similar to `DiffusionPipeline`, You could load an components separately to replace the default one in the pipeline. But in Modular Diffusers system, you need to use `ComponentSpec` to load/create them.

`ComponentSpec` defines how to create or load components and can actually create them using its `create()` method (for ConfigMixin objects) or `load()` method (wrapper around `from_pretrained()`). When a component is loaded with a ComponentSpec, it gets tagged with a unique ID that encodes its creation parameters, allowing you to always extract the original specification using `ComponentSpec.from_component()`. In Modular Diffusers, all pretrained models should be loaded using `ComponentSpec` objects.

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
>>> t2i_pipeline.loader
StableDiffusionXLModularLoader {
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


### Run a `ModularPipeline`

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

```py
t2i_pipeline.doc
```

</Tip>
```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS

t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i"
t2i_pipeline = t2i_blocks.init_pipeline(modular_repo_id)

t2i_pipeline.load_components(torch_dtype=torch.float16)
t2i_pipeline.to("cuda")

image = t2i_pipeline(prompt="a cat", output="images")[0]
image.save("modular_t2i_out.png")
```


## An slightly advanced Workflow

We've learned the basic components of the Modular Diffusers System. Now let's tie everything together with more practical example that demonstrates the true power of Modular Diffusers: working between with multiple pipelines that can share components.


```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks, ComponentsManager
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS, IMAGE2IMAGE_BLOCKS

# create t2i blocks and then pop out the text_encoder step and decoder step so that we can use them in standalone manner
t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS.copy())
text_blocks = t2i_blocks.blocks.pop("text_encoder")
decoder_blocks = t2i_blocks.blocks.pop("decode")

# Create a refiner blocks
# - removing image_encoder a since we'll use latents from t2i
# - removing decode since we already created a seperate decoder_block
i2i_blocks_dict = IMAGE2IMAGE_BLOCKS.copy()
i2i_blocks_dict.pop("image_encoder")
i2i_blocks_dict.pop("decode")
refiner_blocks = SequentialPipelineBlocks.from_blocks_dict(i2i_blocks_dict)

# Set up component manager and turn on the offloading
components = ComponentsManager()
components.enable_auto_cpu_offload(device="cuda")

# convert all blocks into runnable pipelines: text_node, decoder_node, t2i_pipe, refiner_pipe
t2i_repo = "YiYiXu/modular-loader-t2i"
refiner_repo = "YiYiXu/modular_refiner"
dtype = torch.float16

text_node = text_blocks.init_pipeline(t2i_repo, component_manager=components, collection="t2i")
text_node.load_components(torch_dtype=dtype)

decoder_node = decoder_blocks.init_pipeline(t2i_repo, component_manager=components, collection="t2i")
decoder_node.load_components(torch_dtype=dtype)

t2i_pipe = t2i_blocks.init_pipeline(t2i_repo, component_manager=components, collection="t2i")
t2i_pipe.load_components(torch_dtype=dtype)

# for refiner pipeline, only unet is unique so we only load unet here, and we will reuse other components
refiner_pipe = refiner_blocks.init_pipeline(refiner_repo, component_manager=components, collection="refiner")
refiner_pipe.load_components(component_names="unet", torch_dtype=dtype)
```

let's inspect components manager here, you can see that 5 models are automatically registered: two text encoders, two UNets, and one VAE. The models are organized by collection - 4 models under "t2i" and one UNet under "refiner". This happens because we passed a `collection` parameter when initializing each pipeline. For example, when we created the refiner pipeline, we did `refiner_pipe = refiner_blocks.init_pipeline(refiner_repo, component_manager=components, collection="refiner")`. All models loaded by `refiner_pipe.load_components(...)` are automatically placed under the "refiner" collection. 

Notice that all models are currently on CPU with execution device "cuda:0" - this is due to the auto CPU offloading strategy we enabled with `components.enable_auto_cpu_offload(device="cuda")`. 

The manager also displays useful info like dtype and memory size for each model.

```py
>>> components
Components:
======================================================================================================================================================================================
Models:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name            | Class                       | Device: act(exec)    | Dtype           | Size (GB)  | Load ID                                                           | Collection
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
text_encoder_2  | CLIPTextModelWithProjection | cpu(cuda:0)          | torch.float16   | 1.29       | stabilityai/stable-diffusion-xl-base-1.0|text_encoder_2|null|null | t2i
text_encoder    | CLIPTextModel               | cpu(cuda:0)          | torch.float16   | 0.23       | stabilityai/stable-diffusion-xl-base-1.0|text_encoder|null|null   | t2i
unet            | UNet2DConditionModel        | cpu(cuda:0)          | torch.float16   | 4.78       | RunDiffusion/Juggernaut-XL-v9|unet|fp16|null                      | t2i
unet            | UNet2DConditionModel        | cpu(cuda:0)          | torch.float16   | 4.21       | stabilityai/stable-diffusion-xl-refiner-1.0|unet|null|null        | refiner
vae             | AutoencoderKL               | cpu(cuda:0)          | torch.float16   | 0.16       | madebyollin/sdxl-vae-fp16-fix|null|null|null                      | t2i
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Other Components:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name            | Class                       | Collection
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tokenizer       | CLIPTokenizer               | t2i
tokenizer_2     | CLIPTokenizer               | t2i
scheduler       | EulerDiscreteScheduler      | t2i
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```


Now let's reuse components from the t2i pipeline in the refiner. First, let's check the loading status of the refiner pipeline to understand what components are needed:

```py
>>> refiner_pipe.loader
```

Looking at the loader output, you can see that `text_encoder` and `tokenizer` have empty loading spec maps (their `repo` fields are `null`), this is because refiner pipeline does not use these two components so they are not listed in the `modular_model_index.json` in `refiner_repo`. The `unet` is already correctly loaded from the refiner repository. We need to load the remaining components: `vae`, `text_encoder_2`, `tokenizer_2`, and `scheduler`. Since these components are already available in the t2i collection, we can reuse them instead of loading duplicates.

Now let's reuse the components from the t2i pipeline in the refiner. We use the`|` to select multiple components from components manager at once:

```py
# Reuse components from t2i pipeline (select everything at once)
reuse_components = components.get("text_encoder_2|scheduler|vae|tokenizer_2", as_name_component_tuples=True)
refiner_pipe.update_components(**dict(reuse_components))
```

You'll see warnings indicating that these components already exist in the components manager:

```out
component 'text_encoder_2' already exists as 'text_encoder_2_238ae9a7-c864-4837-a8a2-f58ed753b2d0'
component 'tokenizer_2' already exists as 'tokenizer_2_b795af3d-f048-4b07-a770-9e8237a2be2d'
component 'scheduler' already exists as 'scheduler_e3435f63-266a-4427-9383-eb812e830fe8'
component 'vae' already exists as 'vae_357eee6a-4a06-46f1-be83-494f7d60ca69'
```

These warnings are expected and indicate that the components manager is correctly identifying that these components are already loaded. The system will reuse the existing components rather than creating duplicates.

Let's check the components manager again to see the updated state:

```py
>>> components
Components:
======================================================================================================================================================================================
Models:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name            | Class                       | Device: act(exec)    | Dtype           | Size (GB)  | Load ID                                                           | Collection
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
text_encoder    | CLIPTextModel               | cpu(cuda:0)          | torch.float16   | 0.23       | stabilityai/stable-diffusion-xl-base-1.0|text_encoder|null|null   | t2i
text_encoder_2  | CLIPTextModelWithProjection | cpu(cuda:0)          | torch.float16   | 1.29       | stabilityai/stable-diffusion-xl-base-1.0|text_encoder_2|null|null | t2i
                |                             |                      |                 |            |                                                                   | refiner
vae             | AutoencoderKL               | cpu(cuda:0)          | torch.float16   | 0.16       | madebyollin/sdxl-vae-fp16-fix|null|null|null                      | t2i
                |                             |                      |                 |            |                                                                   | refiner
unet            | UNet2DConditionModel        | cpu(cuda:0)          | torch.float16   | 4.78       | RunDiffusion/Juggernaut-XL-v9|unet|fp16|null                      | t2i
unet            | UNet2DConditionModel        | cpu(cuda:0)          | torch.float16   | 4.21       | stabilityai/stable-diffusion-xl-refiner-1.0|unet|null|null        | refiner
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Other Components:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name            | Class                       | Collection
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tokenizer_2     | CLIPTokenizer               | t2i
                |                             | refiner
tokenizer       | CLIPTokenizer               | t2i
scheduler       | EulerDiscreteScheduler      | t2i
                |                             | refiner
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```

Notice how `text_encoder_2`, `vae`, `tokenizer_2`, and `scheduler` now appear under both "t2i" and "refiner" collections.

We can start to generate an image with the t2i pipeline and refine it.

First to run the prompt through text_node to get prompt embeddings

<Tip>

💡 don't forget to `text_node.doc` to find out what outputs are available and set the `output` argument accordingly

</Tip>

```py
prompt = "A crystal orb resting on a wooden table with a yellow rubber duck, surrounded by aged scrolls and alchemy tools, illuminated by candlelight, detailed texture, high resolution image"

text_embeddings = text_node(prompt=prompt, output=["prompt_embeds","negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"])
```

Now generate latents with t2i pipeline and then refine with refiner. Note that both our `t2i_pipe` and `refiner_pipe` do not have decoder steps since we separated them out earlier, so we need to use `output="latents"` instead of `output="images"`.

<Tip>

💡 `t2i_pipe.blocks` shows you what steps this pipeline takes. You can see that our `t2i_pipe` no longer includes the `text_encoder` and `decode` steps since we removed them earlier when we popped them out to create separate nodes.

```py
>>> t2i_pipe.blocks
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  Description: 


  Components:
      scheduler (`EulerDiscreteScheduler`)
      guider (`ClassifierFreeGuidance`)
      unet (`UNet2DConditionModel`)

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
                   

)
```

</Tip>

```py
latents = t2i_pipe(**text_embeddings, num_inference_steps=25, output="latents")
refined_latents = refiner_pipe(image_latents=latents, prompt=prompt, num_inference_steps=10, output="latents")
```

To get the final images, we need to pass the latents through our separate decoder node:

```py
image = decoder_node(latents=latents, output="images")[0]
refined_image = decoder_node(latents=refined_latents, output="images")[0]
```

# YiYi TODO: maybe more on controlnet/lora/ip-adapter






