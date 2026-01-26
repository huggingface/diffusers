<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quickstart

Modular Diffusers is a framework for quickly building flexible and customizable pipelines. At the core of Modular Diffusers are [`ModularPipelineBlocks`] that can be combined with other blocks to adapt to new workflows. The blocks are converted into a [`ModularPipeline`], a friendly user-facing interface for running generation tasks.

This guide shows you how to run a modular pipeline, understand its structure, and customize it by modifying the blocks that compose it.

## Run a pipeline

[`ModularPipeline`] is the main interface for loading, running, and managing modular pipelines.

```py
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("Qwen/Qwen-Image")
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    prompt="cat wizard with red hat, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney",
).images[0]
image
```

[`~ModularPipeline.from_pretrained`] uses lazy loading - it reads the configuration to learn where to load each component from, but doesn't actually load the model weights until you call [`~ModularPipeline.load_components`]. This gives you control over when and how components are loaded.

Learn more about creating and loading pipelines in the [Creating a pipeline](https://huggingface.co/docs/diffusers/modular_diffusers/modular_pipeline#creating-a-pipeline) and [Loading components](https://huggingface.co/docs/diffusers/modular_diffusers/modular_pipeline#loading-components) guides.

## Understand the structure

The pipeline is built from [`ModularPipelineBlocks`] specific to the model. For example, [`QwenImage`] is built from `QwenImageAutoBlocks`. Print it to see its structure.

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

This pipeline supports multiple workflows and adapts its behavior based on the inputs you provide. For example, if you pass `image` to the pipeline, it runs an image-to-image workflow instead of text-to-image.

```py
from diffusers.utils import load_image

input_image = load_image("https://github.com/Trgtuan10/Image_storage/blob/main/cute_cat.png?raw=true")

image = pipe(
    prompt="cat wizard with red hat, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney",
    image=input_image,
).images[0]
```

Learn more about conditional blocks in the [AutoPipelineBlocks](https://huggingface.co/docs/diffusers/modular_diffusers/auto_pipeline_blocks) guide.

Use `get_workflow()` to extract the blocks for a specific workflow.

```py
img2img_blocks = pipe.blocks.get_workflow("image2image")
```

### Sub-blocks

Blocks are the building blocks of the modular system. They are *definitions* that specify the inputs, outputs, and computation logic for a step - and they can be composed together in different ways.

`QwenImageAutoBlocks` is itself composed of smaller blocks: `text_encoder`, `vae_encoder`, `controlnet_vae_encoder`, `denoise`, and `decode`. Access them through the `sub_blocks` property.

The `doc` property is useful for seeing the full documentation of any block, including its inputs, outputs, and components.

```py
vae_encoder_block = pipe.blocks.sub_blocks["vae_encoder"]
print(vae_encoder_block.doc)
```

This block can be converted to a pipeline and run on its own with [`~ModularPipelineBlocks.init_pipeline`].
```py
vae_encoder_pipe = vae_encoder_block.init_pipeline()

# Reuse the VAE we already loaded, we can reuse it with update_componenets() method
vae_encoder_pipe.update_components(vae=pipe.vae)

# Run just this block
image_latents = vae_encoder_pipe(image=input_image).image_latents
print(image_latents.shape)
```

It reuses the VAE from our original pipeline instead of reloading it, keeping memory usage efficient. Learn more in the [Loading components](https://huggingface.co/docs/diffusers/modular_diffusers/modular_pipeline#loading-components) guide.

You can also add new blocks to compose new workflows. Let's add a canny edge detection block to create a ControlNet pipeline.

1. Load the canny block from the Hub and insert it into the ControlNet workflow. If you want to learn how to create your own custom blocks and share them on the Hub, check out the [Building Custom Blocks](https://huggingface.co/docs/diffusers/modular_diffusers/custom_blocks) guide.

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

Use `get_workflow` to extract the ControlNet workflow.

```py
# Get the controlnet workflow 
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
It requires control_image as input. After inserting the canny block, the pipeline will accept a regular image instead.

```py
# and insert canny at the beginning
blocks.sub_blocks.insert("canny", canny_block, 0)

# Check the updated structure - notice the pipeline now takes "image" as input
# even though it's a controlnet pipeline, because canny preprocesses it into control_image
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

Now the pipeline takes `image` as input - the canny block will preprocess it into `control_image` automatically.

Create a pipeline from the modified blocks and load a ControlNet model.

```py
pipeline = blocks.init_pipeline("Qwen/Qwen-Image")
pipeline.load_components(torch_dtype=torch.bfloat16)

# Load the ControlNet model
controlnet_spec = pipeline.get_component_spec("controlnet")
controlnet_spec.pretrained_model_name_or_path = "InstantX/Qwen-Image-ControlNet-Union"
controlnet = controlnet_spec.load(torch_dtype=torch.bfloat16)
pipeline.update_components(controlnet=controlnet)
pipeline.to("cuda")
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
<hfoption id="Build custom blocks">

Learn how to create your own blocks with custom logic in the [Building Custom Blocks](./custom_blocks) guide.

</hfoption>
<hfoption id="Share components">

Use [`ComponentsManager`](./components_manager) to share models across multiple pipelines and manage memory efficiently.

</hfoption>
<hfoption id="Visual interface">

Connect modular pipelines to [Mellon](https://github.com/cubiq/Mellon), a visual node-based interface for building workflows. Custom blocks built with Modular Diffusers work out of the box with Mellon - no UI code required. Read more in Mellon guide

</hfoption>
</hfoptions>
