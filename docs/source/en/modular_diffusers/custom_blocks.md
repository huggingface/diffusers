<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


# Building Custom Blocks

[ModularPipelineBlocks](./pipeline_block) are the fundamental building blocks of a [`ModularPipeline`]. You can create custom blocks by defining their inputs, outputs, and computation logic. This guide demonstrates how to create and use a custom block.

> [!TIP]
> Explore the [Modular Diffusers Custom Blocks](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks) collection for official custom blocks.

## Project Structure

Your custom block project should use the following structure:

```shell
.
├── block.py
└── modular_config.json
```

- `block.py` contains the custom block implementation
- `modular_config.json` contains the metadata needed to load the block

## Quick Start with Template

The fastest way to create a custom block is to start from our template. The template provides a pre-configured project structure with `block.py` and `modular_config.json` files, plus commented examples showing how to define components, inputs, outputs, and the `__call__` method—so you can focus on your custom logic instead of boilerplate setup.

### Download the template

```python
from diffusers import ModularPipelineBlocks

model_id = "diffusers/custom-block-template"
local_dir = model_id.split("/")[-1]

blocks = ModularPipelineBlocks.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    local_dir=local_dir
)
```

This saves the template files to `custom-block-template/` locally or you could use `local_dir` to save to a specific location.

### Edit locally

Open `block.py` and implement your custom block. The template includes commented examples showing how to define each property. See the [Florence-2 example](#example-florence-2-image-annotator) below for a complete implementation.

### Test your block

```python
from diffusers import ModularPipelineBlocks

blocks = ModularPipelineBlocks.from_pretrained(local_dir, trust_remote_code=True)
pipeline = blocks.init_pipeline()
output = pipeline(...)  # your inputs here
```

### Upload to the Hub

```python
pipeline.save_pretrained(local_dir, repo_id="your-username/your-block-name", push_to_hub=True)
```

## Example: Florence-2 Image Annotator

This example creates a custom block with [Florence-2](https://huggingface.co/docs/transformers/model_doc/florence2) to process an input image and generate a mask for inpainting.

### Define components

Define the components the block needs, `Florence2ForConditionalGeneration` and its processor. When defining components, specify the `name` (how you'll access it in code), `type_hint` (the model class), and `pretrained_model_name_or_path` (where to load weights from).

```python
# Inside block.py
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec
from transformers import AutoProcessor, Florence2ForConditionalGeneration


class Florence2ImageAnnotatorBlock(ModularPipelineBlocks):

    @property
    def expected_components(self):
        return [
            ComponentSpec(
                name="image_annotator",
                type_hint=Florence2ForConditionalGeneration,
                pretrained_model_name_or_path="florence-community/Florence-2-base-ft",
            ),
            ComponentSpec(
                name="image_annotator_processor",
                type_hint=AutoProcessor,
                pretrained_model_name_or_path="florence-community/Florence-2-base-ft",
            ),
        ]
```

### Define inputs and outputs

Inputs include the image, annotation task, and prompt. Outputs include the generated mask and annotations.

```python
from typing import List, Union
from PIL import Image
from diffusers.modular_pipelines import InputParam, OutputParam


class Florence2ImageAnnotatorBlock(ModularPipelineBlocks):

    # ... expected_components from above ...

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "image",
                type_hint=Union[Image.Image, List[Image.Image]],
                required=True,
                description="Image(s) to annotate",
            ),
            InputParam(
                "annotation_task",
                type_hint=str,
                default="<REFERRING_EXPRESSION_SEGMENTATION>",
                description="Annotation task to perform (e.g., <OD>, <CAPTION>, <REFERRING_EXPRESSION_SEGMENTATION>)",
            ),
            InputParam(
                "annotation_prompt",
                type_hint=str,
                required=True,
                description="Prompt to provide context for the annotation task",
            ),
            InputParam(
                "annotation_output_type",
                type_hint=str,
                default="mask_image",
                description="Output type: 'mask_image', 'mask_overlay', or 'bounding_box'",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "mask_image",
                type_hint=Image.Image,
                description="Inpainting mask for the input image",
            ),
            OutputParam(
                "annotations",
                type_hint=dict,
                description="Raw annotation predictions",
            ),
            OutputParam(
                "image",
                type_hint=Image.Image,
                description="Annotated image",
            ),
        ]
```

### Implement the `__call__` method

The `__call__` method contains the block's logic. Access inputs via `block_state`, run your computation, and set outputs back to `block_state`.

```python
import torch
from diffusers.modular_pipelines import PipelineState


class Florence2ImageAnnotatorBlock(ModularPipelineBlocks):

    # ... expected_components, inputs, intermediate_outputs from above ...

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        
        images, annotation_task_prompt = self.prepare_inputs(
            block_state.image, block_state.annotation_prompt
        )
        task = block_state.annotation_task
        fill = block_state.fill
        
        annotations = self.get_annotations(
            components, images, annotation_task_prompt, task
        )
        block_state.annotations = annotations
        if block_state.annotation_output_type == "mask_image":
            block_state.mask_image = self.prepare_mask(images, annotations)
        else:
            block_state.mask_image = None

        if block_state.annotation_output_type == "mask_overlay":
            block_state.image = self.prepare_mask(images, annotations, overlay=True, fill=fill)

        elif block_state.annotation_output_type == "bounding_box":
            block_state.image = self.prepare_bounding_boxes(images, annotations)

        self.set_block_state(state, block_state)

        return components, state
    
    # Helper methods for mask/bounding box generation...
```

> [!TIP]
> See the complete implementation at [diffusers/Florence2-image-Annotator](https://huggingface.co/diffusers/Florence2-image-Annotator).

## Using Custom Blocks

Load a custom block with [`~ModularPipeline.from_pretrained`] and set `trust_remote_code=True`.

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

# Load the Florence-2 annotator pipeline
image_annotator = ModularPipeline.from_pretrained(
    "diffusers/Florence2-image-Annotator",
    trust_remote_code=True
)

# Check the docstring to see inputs/outputs
print(image_annotator.blocks.doc)
```

Use the block to generate a mask:

```python
image_annotator.load_components(torch_dtype=torch.bfloat16)
image_annotator.to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg")
image = image.resize((1024, 1024))
prompt = ["A red car"]
annotation_task = "<REFERRING_EXPRESSION_SEGMENTATION>"
annotation_prompt = ["the car"]

mask_image = image_annotator_node(
    prompt=prompt,
    image=image,
    annotation_task=annotation_task,
    annotation_prompt=annotation_prompt,
    annotation_output_type="mask_image",
).images
mask_image[0].save("car-mask.png")
```

Compose it with other blocks to create a new pipeline:

```python
# Get the annotator block
annotator_block = image_annotator.blocks

# Get an inpainting workflow and insert the annotator at the beginning
inpaint_blocks = ModularPipeline.from_pretrained("Qwen/Qwen-Image").blocks.get_workflow("inpainting")
inpaint_blocks.sub_blocks.insert("image_annotator", annotator_block, 0)

# Initialize the combined pipeline
pipe = inpaint_blocks.init_pipeline()
pipe.load_components(torch_dtype=torch.float16, device="cuda")

# Now the pipeline automatically generates masks from prompts
output = pipe(
    prompt=prompt,
    image=image,
    annotation_task=annotation_task,
    annotation_prompt=annotation_prompt,
    annotation_output_type="mask_image",
    num_inference_steps=35,
    guidance_scale=7.5,
    strength=0.95,
    output="images"
)
output[0].save("florence-inpainting.png")
```

## Editing custom blocks

Edit custom blocks by downloading it locally. This is the same workflow as the [Quick Start with Template](#quick-start-with-template), but starting from an existing block instead of the template.

Use the `local_dir` argument to download a custom block to a specific folder:

```python
from diffusers import ModularPipelineBlocks

# Download to a local folder for editing
annotator_block = ModularPipelineBlocks.from_pretrained(
    "diffusers/Florence2-image-Annotator",
    trust_remote_code=True,
    local_dir="./my-florence-block"
)
```

Any changes made to the block files in this folder will be reflected when you load the block again. When you're ready to share your changes, upload to a new repository:

```python
pipeline = annotator_block.init_pipeline()
pipeline.save_pretrained("./my-florence-block", repo_id="your-username/my-custom-florence", push_to_hub=True)
```

## Next Steps

<hfoptions id="next">
<hfoption id="Learn block types">

This guide covered creating a single custom block. Learn how to compose multiple blocks together:

- [SequentialPipelineBlocks](./sequential_pipeline_blocks): Chain blocks to execute in sequence
- [ConditionalPipelineBlocks](./auto_pipeline_blocks): Create conditional blocks that select different execution paths
- [LoopSequentialPipelineBlocks](./loop_sequential_pipeline_blocks): Define an iterative workflows like the denoising loop

</hfoption>
<hfoption id="Use in Mellon">

Make your custom block work with Mellon's visual interface. See the [Mellon Custom Blocks](./mellon) guide.

</hfoption>
<hfoption id="Explore existing blocks">

Browse the [Modular Diffusers Custom Blocks](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks) collection for inspiration and ready-to-use blocks.

</hfoption>
</hfoptions>