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
> Explore the [Modular Diffusers Custom Blocks](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks) collection for official custom modular blocks like Nano Banana.

## Project Structure

Your custom block project should use the following structure:

```shell
.
├── block.py
└── modular_config.json
```

- `block.py` contains the custom block implementation
- `modular_config.json` contains the metadata needed to load the block

## Example: Florence 2 Inpainting Block

In this example we will create a custom block that uses the [Florence 2](https://huggingface.co/docs/transformers/model_doc/florence2) model to process an input image and generate a mask for inpainting.

The first step is to define the components that the block will use. In this case, we will need to use the `Florence2ForConditionalGeneration` model and its corresponding processor `AutoProcessor`. When defining components, we must specify the name of the component within our pipeline, model class via `type_hint`, and provide a `pretrained_model_name_or_path` for the component if we intend to load the model weights from a specific repository on the Hub.

```py
# Inside block.py
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    ComponentSpec,
)
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

Next, we define the inputs and outputs of the block. The inputs include the image to be annotated, the annotation task, and the annotation prompt. The outputs include the generated mask image and annotations.

```py
from typing import List, Union
from PIL import Image, ImageDraw
import torch
import numpy as np

from diffusers.modular_pipelines import (
    PipelineState,
    ModularPipelineBlocks,
    InputParam,
    ComponentSpec,
    OutputParam,
)
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
                type_hint=Union[str, List[str]],
                required=True,
                default="<REFERRING_EXPRESSION_SEGMENTATION>",
                description="""Annotation Task to perform on the image.
                Supported Tasks:

                <OD>
                <REFERRING_EXPRESSION_SEGMENTATION>
                <CAPTION>
                <DETAILED_CAPTION>
                <MORE_DETAILED_CAPTION>
                <DENSE_REGION_CAPTION>
                <CAPTION_TO_PHRASE_GROUNDING>
                <OPEN_VOCABULARY_DETECTION>

                """,
            ),
            InputParam(
                "annotation_prompt",
                type_hint=Union[str, List[str]],
                required=True,
                description="""Annotation Prompt to provide more context to the task.
                Can be used to detect or segment out specific elements in the image
                """,
            ),
            InputParam(
                "annotation_output_type",
                type_hint=str,
                required=True,
                default="mask_image",
                description="""Output type from annotation predictions. Availabe options are
                mask_image:
                    -black and white mask image for the given image based on the task type
                mask_overlay:
                    - mask overlayed on the original image
                bounding_box:
                    - bounding boxes drawn on the original image
                """,
            ),
            InputParam(
                "annotation_overlay",
                type_hint=bool,
                required=True,
                default=False,
                description="",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "mask_image",
                type_hint=Image,
                description="Inpainting Mask for input Image(s)",
            ),
            OutputParam(
                "annotations",
                type_hint=dict,
                description="Annotations Predictions for input Image(s)",
            ),
            OutputParam(
                "image",
                type_hint=Image,
                description="Annotated input Image(s)",
            ),
        ]

```

Now we implement the `__call__` method, which contains the logic for processing the input image and generating the mask.

```py
from typing import List, Union
from PIL import Image, ImageDraw
import torch
import numpy as np

from diffusers.modular_pipelines import (
    PipelineState,
    ModularPipelineBlocks,
    InputParam,
    ComponentSpec,
    OutputParam,
)
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
                type_hint=Union[str, List[str]],
                required=True,
                default="<REFERRING_EXPRESSION_SEGMENTATION>",
                description="""Annotation Task to perform on the image.
                Supported Tasks:

                <OD>
                <REFERRING_EXPRESSION_SEGMENTATION>
                <CAPTION>
                <DETAILED_CAPTION>
                <MORE_DETAILED_CAPTION>
                <DENSE_REGION_CAPTION>
                <CAPTION_TO_PHRASE_GROUNDING>
                <OPEN_VOCABULARY_DETECTION>

                """,
            ),
            InputParam(
                "annotation_prompt",
                type_hint=Union[str, List[str]],
                required=True,
                description="""Annotation Prompt to provide more context to the task.
                Can be used to detect or segment out specific elements in the image
                """,
            ),
            InputParam(
                "annotation_output_type",
                type_hint=str,
                required=True,
                default="mask_image",
                description="""Output type from annotation predictions. Availabe options are
                mask_image:
                    -black and white mask image for the given image based on the task type
                mask_overlay:
                    - mask overlayed on the original image
                bounding_box:
                    - bounding boxes drawn on the original image
                """,
            ),
            InputParam(
                "annotation_overlay",
                type_hint=bool,
                required=True,
                default=False,
                description="",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "mask_image",
                type_hint=Image,
                description="Inpainting Mask for input Image(s)",
            ),
            OutputParam(
                "annotations",
                type_hint=dict,
                description="Annotations Predictions for input Image(s)",
            ),
            OutputParam(
                "image",
                type_hint=Image,
                description="Annotated input Image(s)",
            ),
        ]

    def get_annotations(self, components, images, prompts, task):
        task_prompts = [task + prompt for prompt in prompts]

        inputs = components.image_annotator_processor(
            text=task_prompts, images=images, return_tensors="pt"
        ).to(components.image_annotator.device, components.image_annotator.dtype)

        generated_ids = components.image_annotator.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        annotations = components.image_annotator_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        outputs = []
        for image, annotation in zip(images, annotations):
            outputs.append(
                components.image_annotator_processor.post_process_generation(
                    annotation, task=task, image_size=(image.width, image.height)
                )
            )
        return outputs

    def prepare_mask(self, images, annotations, overlay=False, fill="white"):
        masks = []
        for image, annotation in zip(images, annotations):
            mask_image = image.copy() if overlay else Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask_image)

            for _, _annotation in annotation.items():
                if "polygons" in _annotation:
                    for polygon in _annotation["polygons"]:
                        polygon = np.array(polygon).reshape(-1, 2)
                        if len(polygon) < 3:
                            continue
                        polygon = polygon.reshape(-1).tolist()
                        draw.polygon(polygon, fill=fill)

                elif "bbox" in _annotation:
                    bbox = _annotation["bbox"]
                    draw.rectangle(bbox, fill="white")

            masks.append(mask_image)

        return masks

    def prepare_bounding_boxes(self, images, annotations):
        outputs = []
        for image, annotation in zip(images, annotations):
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)
            for _, _annotation in annotation.items():
                bbox = _annotation["bbox"]
                label = _annotation["label"]

                draw.rectangle(bbox, outline="red", width=3)
                draw.text((bbox[0], bbox[1] - 20), label, fill="red")

            outputs.append(image_copy)

        return outputs

    def prepare_inputs(self, images, prompts):
        prompts = prompts or ""

        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]

        if len(images) != len(prompts):
            raise ValueError("Number of images and annotation prompts must match.")

        return images, prompts

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

```

Once we have defined our custom block, we can save it to the Hub, using either the CLI or the [`push_to_hub`] method. This will make it easy to share and reuse our custom block with other pipelines.

<hfoptions id="share">
<hfoption id="hf CLI">

```shell
# In the folder with the `block.py` file, run:
diffusers-cli custom_block
```

Then upload the block to the Hub:

```shell
hf upload <your repo id> . .
```
</hfoption>
<hfoption id="push_to_hub">

```py
from block import Florence2ImageAnnotatorBlock
block = Florence2ImageAnnotatorBlock()
block.push_to_hub("<your repo id>")
```

</hfoption>
</hfoptions>

## Using Custom Blocks

Load the custom block with [`~ModularPipelineBlocks.from_pretrained`] and set `trust_remote_code=True`.

```py
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import INPAINT_BLOCKS
from diffusers.utils import load_image

# Fetch the Florence2 image annotator block that will create our mask
image_annotator_block = ModularPipelineBlocks.from_pretrained("diffusers/florence-2-custom-block", trust_remote_code=True)

my_blocks = INPAINT_BLOCKS.copy()
# insert the annotation block before the image encoding step
my_blocks.insert("image_annotator", image_annotator_block, 1)

# Create our initial set of inpainting blocks
blocks = SequentialPipelineBlocks.from_blocks_dict(my_blocks)

repo_id = "diffusers/modular-stable-diffusion-xl-base-1.0"
pipe = blocks.init_pipeline(repo_id)
pipe.load_components(torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true")
image = image.resize((1024, 1024))

prompt = ["A red car"]
annotation_task = "<REFERRING_EXPRESSION_SEGMENTATION>"
annotation_prompt = ["the car"]

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

## Editing Custom Blocks

By default, custom blocks are saved in your cache directory. Use the `local_dir` argument to download and edit a custom block in a specific folder.

```py
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import INPAINT_BLOCKS
from diffusers.utils import load_image

# Fetch the Florence2 image annotator block that will create our mask
image_annotator_block = ModularPipelineBlocks.from_pretrained("diffusers/florence-2-custom-block", trust_remote_code=True, local_dir="/my-local-folder")
```

Any changes made to the block files in this folder will be reflected when you load the block again.
