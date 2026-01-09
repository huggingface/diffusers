<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SequentialPipelineBlocks

[`~modular_pipelines.SequentialPipelineBlocks`] are a multi-block type that composes other [`~modular_pipelines.ModularPipelineBlocks`] together in a sequence. Data flows linearly from one block to the next using `inputs` and `intermediate_outputs`. Each block in [`~modular_pipelines.SequentialPipelineBlocks`] usually represents a step in the pipeline, and by combining them, you gradually build a pipeline.

This guide shows you how to connect two blocks into a [`~modular_pipelines.SequentialPipelineBlocks`].

Create two [`~modular_pipelines.ModularPipelineBlocks`]. The first block, `InputBlock`, outputs a `batch_size` value and the second block, `ImageEncoderBlock` uses `batch_size` as `inputs`.

<hfoptions id="sequential">
<hfoption id="InputBlock">

```py
from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam

class InputBlock(ModularPipelineBlocks):

    @property
    def inputs(self):
        return [
            InputParam(name="prompt", type_hint=list, description="list of text prompts"),
            InputParam(name="num_images_per_prompt", type_hint=int, description="number of images per prompt"),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name="batch_size", description="calculated batch size"),
        ]

    @property
    def description(self):
        return "A block that determines batch_size based on the number of prompts and num_images_per_prompt argument."

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        batch_size = len(block_state.prompt)
        block_state.batch_size = batch_size * block_state.num_images_per_prompt
        self.set_block_state(state, block_state)
        return components, state
```

</hfoption>
<hfoption id="ImageEncoderBlock">

```py
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam

class ImageEncoderBlock(ModularPipelineBlocks):

    @property
    def inputs(self):
        return [
            InputParam(name="image", type_hint="PIL.Image", description="raw input image to process"),
            InputParam(name="batch_size", type_hint=int),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name="image_latents", description="latents representing the image"),
        ]

    @property
    def description(self):
        return "Encode raw image into its latent presentation"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        # Simulate processing the image
        # This will change the state of the image from a PIL image to a tensor for all blocks
        block_state.image = torch.randn(1, 3, 512, 512)
        block_state.batch_size = block_state.batch_size * 2
        block_state.image_latents = torch.randn(1, 4, 64, 64)
        self.set_block_state(state, block_state)
        return components, state
```

</hfoption>
</hfoptions>

Connect the two blocks by defining an [`InsertableDict`] to map the block names to the block instances. Blocks are executed in the order they're registered in `blocks_dict`.

Use [`~modular_pipelines.SequentialPipelineBlocks.from_blocks_dict`] to create a [`~modular_pipelines.SequentialPipelineBlocks`].

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks, InsertableDict

blocks_dict = InsertableDict()
blocks_dict["input"] = input_block
blocks_dict["image_encoder"] = image_encoder_block

blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)
```

Inspect the sub-blocks in [`~modular_pipelines.SequentialPipelineBlocks`] by calling `blocks`, and for more details about the inputs and outputs, access the `docs` attribute.

```py
print(blocks)
print(blocks.doc)
```
