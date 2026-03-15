<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipelineBlocks

[`~modular_pipelines.AutoPipelineBlocks`] are a multi-block type containing blocks that support different workflows. It automatically selects which sub-blocks to run based on the input provided at runtime. This is typically used to package multiple workflows - text-to-image, image-to-image, inpaint - into a single pipeline for convenience.

This guide shows how to create [`~modular_pipelines.AutoPipelineBlocks`].

Create three [`~modular_pipelines.ModularPipelineBlocks`] for text-to-image, image-to-image, and inpainting. These represent the different workflows available in the pipeline.

<hfoptions id="auto">
<hfoption id="text-to-image">

```py
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam

class TextToImageBlock(ModularPipelineBlocks):
    model_name = "text2img"

    @property
    def inputs(self):
        return [InputParam(name="prompt")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "I'm a text-to-image workflow!"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        print("running the text-to-image workflow")
        # Add your text-to-image logic here
        # For example: generate image from prompt
        self.set_block_state(state, block_state)
        return components, state
```


</hfoption>
<hfoption id="image-to-image">

```py
class ImageToImageBlock(ModularPipelineBlocks):
    model_name = "img2img"

    @property
    def inputs(self):
        return [InputParam(name="prompt"), InputParam(name="image")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "I'm an image-to-image workflow!"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        print("running the image-to-image workflow")
        # Add your image-to-image logic here
        # For example: transform input image based on prompt
        self.set_block_state(state, block_state)
        return components, state
```


</hfoption>
<hfoption id="inpaint">

```py
class InpaintBlock(ModularPipelineBlocks):
    model_name = "inpaint"

    @property
    def inputs(self):
        return [InputParam(name="prompt"), InputParam(name="image"), InputParam(name="mask")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "I'm an inpaint workflow!"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        print("running the inpaint workflow")
        # Add your inpainting logic here
        # For example: fill masked areas based on prompt
        self.set_block_state(state, block_state)
        return components, state
```

</hfoption>
</hfoptions>

Create an [`~modular_pipelines.AutoPipelineBlocks`] class that includes a list of the sub-block classes and their corresponding block names.

You also need to include `block_trigger_inputs`, a list of input names that trigger the corresponding block. If a trigger input is provided at runtime, then that block is selected to run. Use `None` to specify the default block to run if no trigger inputs are detected.

Lastly, it is important to include a `description` that clearly explains which inputs trigger which workflow. This helps users understand how to run specific workflows.

```py
from diffusers.modular_pipelines import AutoPipelineBlocks

class AutoImageBlocks(AutoPipelineBlocks):
    # List of sub-block classes to choose from
    block_classes = [InpaintBlock, ImageToImageBlock, TextToImageBlock]
    # Names for each block in the same order
    block_names = ["inpaint", "img2img", "text2img"]
    # Trigger inputs that determine which block to run
    # - "mask" triggers inpaint workflow
    # - "image" triggers img2img workflow (but only if mask is not provided)
    # - if none of above, runs the text2img workflow (default)
    block_trigger_inputs = ["mask", "image", None]

    @property
    def description(self):
        return (
            "Pipeline generates images given different types of conditions!\n"
            + "This is an auto pipeline block that works for text2img, img2img and inpainting tasks.\n"
            + " - inpaint workflow is run when `mask` is provided.\n"
            + " - img2img workflow is run when `image` is provided (but only when `mask` is not provided).\n"
            + " - text2img workflow is run when neither `image` nor `mask` is provided.\n"
        )
```

It is **very** important to include a `description` to avoid any confusion over how to run a block and what inputs are required. While [`~modular_pipelines.AutoPipelineBlocks`] are convenient, its conditional logic may be difficult to figure out if it isn't properly explained.

Create an instance of `AutoImageBlocks`.

```py
auto_blocks = AutoImageBlocks()
```

For more complex compositions, such as nested [`~modular_pipelines.AutoPipelineBlocks`] blocks when they're used as sub-blocks in larger pipelines, use the [`~modular_pipelines.SequentialPipelineBlocks.get_execution_blocks`] method to extract the a block that is actually run based on your input.

```py
auto_blocks.get_execution_blocks(mask=True)
```

## ConditionalPipelineBlocks

[`~modular_pipelines.AutoPipelineBlocks`] is a special case of [`~modular_pipelines.ConditionalPipelineBlocks`]. While [`~modular_pipelines.AutoPipelineBlocks`] selects blocks based on whether a trigger input is provided or not, [`~modular_pipelines.ConditionalPipelineBlocks`] is able to select a block based on custom selection logic provided in the `select_block` method.

Here is the same example written using [`~modular_pipelines.ConditionalPipelineBlocks`] directly:

```py
from diffusers.modular_pipelines import ConditionalPipelineBlocks

class AutoImageBlocks(ConditionalPipelineBlocks):
    block_classes = [InpaintBlock, ImageToImageBlock, TextToImageBlock]
    block_names = ["inpaint", "img2img", "text2img"]
    block_trigger_inputs = ["mask", "image"]
    default_block_name = "text2img"

    @property
    def description(self):
        return (
            "Pipeline generates images given different types of conditions!\n"
            + "This is an auto pipeline block that works for text2img, img2img and inpainting tasks.\n"
            + " - inpaint workflow is run when `mask` is provided.\n"
            + " - img2img workflow is run when `image` is provided (but only when `mask` is not provided).\n"
            + " - text2img workflow is run when neither `image` nor `mask` is provided.\n"
        )

    def select_block(self, mask=None, image=None) -> str | None:
        if mask is not None:
            return "inpaint"
        if image is not None:
            return "img2img"
        return None  # falls back to default_block_name ("text2img")
```

The inputs listed in `block_trigger_inputs` are passed as keyword arguments to `select_block()`. When `select_block` returns `None`, it falls back to `default_block_name`. If `default_block_name` is also `None`, the entire conditional block is skipped — this is useful for optional processing steps that should only run when specific inputs are provided.

## Workflows

Pipelines that contain conditional blocks ([`~modular_pipelines.AutoPipelineBlocks`] or [`~modular_pipelines.ConditionalPipelineBlocks]`) can support multiple workflows — for example, our SDXL modular pipeline supports a dozen workflows all in one pipeline. But this also means it can be confusing for users to know what workflows are supported and how to run them. For pipeline builders, it's useful to be able to extract only the blocks relevant to a specific workflow.

We recommend defining a `_workflow_map` to give each workflow a name and explicitly list the inputs it requires.

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks

class MyPipelineBlocks(SequentialPipelineBlocks):
    block_classes = [TextEncoderBlock, AutoImageBlocks, DecodeBlock]
    block_names = ["text_encoder", "auto_image", "decode"]

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"image": True, "prompt": True},
        "inpaint": {"mask": True, "image": True, "prompt": True},
    }
```

All of our built-in modular pipelines come with pre-defined workflows. The `available_workflows` property lists all supported workflows:

```py
pipeline_blocks = MyPipelineBlocks()
pipeline_blocks.available_workflows
# ['text2image', 'image2image', 'inpaint']
```

Retrieve a specific workflow with `get_workflow` to inspect and debug a specific block that executes the workflow.

```py
pipeline_blocks.get_workflow("inpaint")
```