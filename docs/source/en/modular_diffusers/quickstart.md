<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quickstart

Modular Diffusers is a framework for quickly building flexible and customizable pipelines. These pipelines can go beyond what standard `DiffusionPipeline`s can do. At the core of Modular Diffusers are [`ModularPipelineBlocks`] that can be combined with other blocks to adapt to new workflows. The blocks are converted into a [`ModularPipeline`], a friendly user-facing interface for running generation tasks.

This guide shows you how to run a modular pipeline, understand its structure, and customize it by modifying the blocks that compose it.

## Run a pipeline

[`ModularPipeline`] is the main interface for loading, running, and managing modular pipelines.
```py
import torch
from diffusers import ModularPipeline, ComponentsManager

# Use ComponentsManager to enable auto CPU offloading for memory efficiency
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda:0")

pipe = ModularPipeline.from_pretrained("Qwen/Qwen-Image", components_manager=manager)
pipe.load_components(torch_dtype=torch.bfloat16)

image = pipe(
    prompt="cat wizard with red hat, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney",
).images[0]
image
```

[`~ModularPipeline.from_pretrained`] uses lazy loading - it reads the configuration to learn where to load each component from, but doesn't actually load the model weights until you call [`~ModularPipeline.load_components`]. This gives you control over when and how components are loaded.

> [!TIP]
> `ComponentsManager` with `enable_auto_cpu_offload` automatically moves models between CPU and GPU as needed, reducing memory usage for large models like Qwen-Image. Learn more in the [ComponentsManager](./components_manager) guide.
>
> If you don't need offloading, remove the `components_manager` argument and move the pipeline to your device manually with `to("cuda")`.

Learn more about creating and loading pipelines in the [Creating a pipeline](https://huggingface.co/docs/diffusers/modular_diffusers/modular_pipeline#creating-a-pipeline) and [Loading components](https://huggingface.co/docs/diffusers/modular_diffusers/modular_pipeline#loading-components) guides.

## Understand the structure

A [`ModularPipeline`] has two parts: a **definition** (the blocks) and a **state** (the loaded components and configs).

Print the pipeline to see its state — the components and their loading status and configuration.
```py
print(pipe)
```
```
QwenImageModularPipeline {
  "_blocks_class_name": "QwenImageAutoBlocks",
  "_class_name": "QwenImageModularPipeline",
  "_diffusers_version": "0.37.0.dev0",
  "transformer": [
    "diffusers",
    "QwenImageTransformer2DModel",
    {
      "pretrained_model_name_or_path": "Qwen/Qwen-Image",
      "revision": null,
      "subfolder": "transformer",
      "type_hint": [
        "diffusers",
        "QwenImageTransformer2DModel"
      ],
      "variant": null
    }
  ],
  ...
}
```

Access the definition through `pipe.blocks` — this is the [`~modular_pipelines.ModularPipelineBlocks`] that defines the pipeline's workflows, inputs, outputs, and computation logic.
```py
print(pipe.blocks)
```
```
QwenImageAutoBlocks(
  Class: SequentialPipelineBlocks

  Description: Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using QwenImage.
      
      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `prompt`, `image`
        - `inpainting`: requires `prompt`, `mask_image`, `image`
        - `controlnet_text2image`: requires `prompt`, `control_image`
        ...

  Components:
      text_encoder (`Qwen2_5_VLForConditionalGeneration`)
      vae (`AutoencoderKLQwenImage`)
      transformer (`QwenImageTransformer2DModel`)
      ...

  Sub-Blocks:
    [0] text_encoder (QwenImageAutoTextEncoderStep)
    [1] vae_encoder (QwenImageAutoVaeEncoderStep)
    [2] controlnet_vae_encoder (QwenImageOptionalControlNetVaeEncoderStep)
    [3] denoise (QwenImageAutoCoreDenoiseStep)
    [4] decode (QwenImageAutoDecodeStep)
)
```

The output returns:
- The supported workflows (text2image, image2image, inpainting, etc.)
- The Sub-Blocks it's composed of (text_encoder, vae_encoder, denoise, decode)

### Workflows

This pipeline supports multiple workflows and adapts its behavior based on the inputs you provide. For example, if you pass `image` to the pipeline, it runs an image-to-image workflow instead of text-to-image. Learn more about how this works under the hood in the [AutoPipelineBlocks](https://huggingface.co/docs/diffusers/modular_diffusers/auto_pipeline_blocks) guide.

```py
from diffusers.utils import load_image

input_image = load_image("https://github.com/Trgtuan10/Image_storage/blob/main/cute_cat.png?raw=true")

image = pipe(
    prompt="cat wizard with red hat, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney",
    image=input_image,
).images[0]
```

Use `get_workflow()` to extract the blocks for a specific workflow. Pass the workflow name (e.g., `"image2image"`, `"inpainting"`, `"controlnet_text2image"`) to get only the blocks relevant to that workflow. This is useful when you want to customize or debug a specific workflow. You can check `pipe.blocks.available_workflows` to see all available workflows.
```py
img2img_blocks = pipe.blocks.get_workflow("image2image")
```


### Sub-blocks

Blocks can contain other blocks. `pipe.blocks` gives you the top-level block definition (here, `QwenImageAutoBlocks`), while `sub_blocks` lets you access the smaller blocks inside it.

`QwenImageAutoBlocks` is composed of: `text_encoder`, `vae_encoder`, `controlnet_vae_encoder`, `denoise`, and `decode`.

These sub-blocks run one after another and data flows linearly from one block to the next — each block's `intermediate_outputs` become available as `inputs` to the next block. This is how [`SequentialPipelineBlocks`](./sequential_pipeline_blocks) work.

You can access them through the `sub_blocks` property. The `doc` property is useful for seeing the full documentation of any block, including its inputs, outputs, and components.
```py
vae_encoder_block = pipe.blocks.sub_blocks["vae_encoder"]
print(vae_encoder_block.doc)
```

This block can be converted to a pipeline so that it can run on its own with [`~ModularPipelineBlocks.init_pipeline`].
```py
vae_encoder_pipe = vae_encoder_block.init_pipeline()

# Reuse the VAE we already loaded, we can reuse it with update_components() method
vae_encoder_pipe.update_components(vae=pipe.vae)

# Run just this block
image_latents = vae_encoder_pipe(image=input_image).image_latents
print(image_latents.shape)
```

It reuses the VAE from our original pipeline instead of reloading it, keeping memory usage efficient. Learn more in the [Loading components](https://huggingface.co/docs/diffusers/modular_diffusers/modular_pipeline#loading-components) guide.

Since blocks are composable, you can modify the pipeline's definition by adding, removing, or swapping blocks to create new workflows. In the next section, we'll add a canny edge detection block to a ControlNet pipeline, so you can pass a regular image instead of a pre-processed canny edge map.

## Compose new workflows

Let's add a canny edge detection block to a ControlNet pipeline. First, load a pre-built canny block from the Hub (see [Building Custom Blocks](https://huggingface.co/docs/diffusers/modular_diffusers/custom_blocks) to create your own).
```py
from diffusers.modular_pipelines import ModularPipelineBlocks

# Load a canny block from the Hub
canny_block = ModularPipelineBlocks.from_pretrained(
    "diffusers-internal-dev/canny-filtering",
    trust_remote_code=True,
)

print(canny_block.doc)
```
```
class CannyBlock

  Inputs:
      image (`Union[Image, ndarray]`):
          Image to compute canny filter on
      low_threshold (`int`, *optional*, defaults to 50):
          Low threshold for the canny filter.
      high_threshold (`int`, *optional*, defaults to 200):
          High threshold for the canny filter.
      ...

  Outputs:
      control_image (`PIL.Image`):
          Canny map for input image
```

Use `get_workflow` to extract the ControlNet workflow from [`QwenImageAutoBlocks`].
```py
# Get the controlnet workflow that we want to work with
blocks = pipe.blocks.get_workflow("controlnet_text2image")
print(blocks.doc)
```
```
class SequentialPipelineBlocks

  Inputs:
      prompt (`str`):
          The prompt or prompts to guide image generation.
      control_image (`Image`):
          Control image for ControlNet conditioning.
      ...
```


The extracted workflow is a [`SequentialPipelineBlocks`](./sequential_pipeline_blocks) and it currently requires `control_image` as input. Insert the canny block at the beginning so the pipeline accepts a regular image instead.
```py
# Insert canny at the beginning
blocks.sub_blocks.insert("canny", canny_block, 0)

# Check the updated structure: CannyBlock is now listed as first sub-block
print(blocks)
# Check the updated doc
print(blocks.doc)
```
```
class SequentialPipelineBlocks

  Inputs:
      image (`Union[Image, ndarray]`):
          Image to compute canny filter on
      low_threshold (`int`, *optional*, defaults to 50):
          Low threshold for the canny filter.
      high_threshold (`int`, *optional*, defaults to 200):
          High threshold for the canny filter.
      prompt (`str`):
          The prompt or prompts to guide image generation.
      ...
```

Now the pipeline takes `image` as input instead of `control_image`. Because blocks in a sequence share data automatically, the canny block's output (`control_image`) flows to the denoise block that needs it, and the canny block's input (`image`) becomes a pipeline input since no earlier block provides it.

Create a pipeline from the modified blocks and load a ControlNet model. The ControlNet isn't part of the original model repository, so load it separately and add it with [`~ModularPipeline.update_components`].
```py
pipeline = blocks.init_pipeline("Qwen/Qwen-Image", components_manager=manager)

pipeline.load_components(torch_dtype=torch.bfloat16)

# Load the ControlNet model
controlnet_spec = pipeline.get_component_spec("controlnet")
controlnet_spec.pretrained_model_name_or_path = "InstantX/Qwen-Image-ControlNet-Union"
controlnet = controlnet_spec.load(torch_dtype=torch.bfloat16)
pipeline.update_components(controlnet=controlnet)
```

Now run the pipeline - the canny block preprocesses the image for ControlNet.
```py
from diffusers.utils import load_image

prompt = "cat wizard with red hat, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney"
image = load_image("https://github.com/Trgtuan10/Image_storage/blob/main/cute_cat.png?raw=true")

output = pipeline(
    prompt=prompt,
    image=image,
).images[0]
output
```

## Next steps

<hfoptions id="next">
<hfoption id="Learn the basics">

Understand the core building blocks of Modular Diffusers:

- [ModularPipelineBlocks](./pipeline_block): The basic unit for defining a step in a pipeline.
- [SequentialPipelineBlocks](./sequential_pipeline_blocks): Chain blocks to run in sequence.
- [AutoPipelineBlocks](./auto_pipeline_blocks): Create pipelines that support multiple workflows.
- [States](./modular_diffusers_states): How data is shared between blocks.

</hfoption>
<hfoption id="Build custom blocks">

Learn how to create your own blocks with custom logic in the [Building Custom Blocks](./custom_blocks) guide.

</hfoption>
<hfoption id="Share components">

Use [`ComponentsManager`](./components_manager) to share models across multiple pipelines and manage memory efficiently.

</hfoption>
<hfoption id="Visual interface">

Connect modular pipelines to [Mellon](https://github.com/cubiq/Mellon), a visual node-based interface for building workflows. Custom blocks built with Modular Diffusers work out of the box with Mellon - no UI code required. Read more in the Mellon guide.

</hfoption>
</hfoptions>