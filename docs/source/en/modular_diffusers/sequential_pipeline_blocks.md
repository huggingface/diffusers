<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SequentialPipelineBlocks

<Tip warning={true}>

🧪 **Experimental Feature**: Modular Diffusers is an experimental feature we are actively developing. The API may be subject to breaking changes.

</Tip>

`SequentialPipelineBlocks` is a subclass of `ModularPipelineBlocks`. Unlike `PipelineBlock`, it is a multi-block that composes other blocks together in sequence, creating modular workflows where data flows from one block to the next. It's one of the most common ways to build complex pipelines by combining simpler building blocks.

<Tip>

Other types of multi-blocks include [AutoPipelineBlocks](auto_pipeline_blocks.md) (for conditional block selection) and [LoopSequentialPipelineBlocks](loop_sequential_pipeline_blocks.md) (for iterative workflows). For information on creating individual blocks, see the [PipelineBlock guide](pipeline_block.md).

Additionally, like all `ModularPipelineBlocks`, `SequentialPipelineBlocks` are definitions/specifications, not runnable pipelines. You need to convert them into a `ModularPipeline` to actually execute them. For information on creating and running pipelines, see the [Modular Pipeline guide](modular_pipeline.md).

</Tip>

In this tutorial, we will focus on how to create `SequentialPipelineBlocks` and how blocks connect and work together.

The key insight is that blocks connect through their intermediate inputs and outputs - the "studs and anti-studs" we discussed in the [PipelineBlock guide](pipeline_block.md). When one block produces an intermediate output, it becomes available as an intermediate input for subsequent blocks.

Let's explore this through an example. We will use the same helper function from the PipelineBlock guide to create blocks.

```py
from diffusers.modular_pipelines import PipelineBlock, InputParam, OutputParam
import torch

def make_block(inputs=[], intermediate_inputs=[], intermediate_outputs=[], block_fn=None, description=None):
    class TestBlock(PipelineBlock):
        model_name = "test"
        
        @property
        def inputs(self):
            return inputs
            
        @property
        def intermediate_inputs(self):
            return intermediate_inputs
            
        @property
        def intermediate_outputs(self):
            return intermediate_outputs
            
        @property
        def description(self):
            return description if description is not None else ""
            
        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            if block_fn is not None:
                block_state = block_fn(block_state, state)
            self.set_block_state(state, block_state)
            return components, state
    
    return TestBlock
```

Let's create a block that produces `batch_size`, which we'll call "input_block":

```py
def input_block_fn(block_state, pipeline_state):
    
    batch_size = len(block_state.prompt)
    block_state.batch_size = batch_size * block_state.num_images_per_prompt
    
    return block_state

input_block_cls = make_block(
    inputs=[
        InputParam(name="prompt", type_hint=list, description="list of text prompts"),
        InputParam(name="num_images_per_prompt", type_hint=int, description="number of images per prompt")
    ],
    intermediate_outputs=[
        OutputParam(name="batch_size", description="calculated batch size")
    ],
    block_fn=input_block_fn,
    description="A block that determines batch_size based on the number of prompts and num_images_per_prompt argument."
)
input_block = input_block_cls()
```

Now let's create a second block that uses the `batch_size` from the first block:

```py
def image_encoder_block_fn(block_state, pipeline_state):
    # Simulate processing the image
    block_state.image = torch.randn(1, 3, 512, 512)
    block_state.batch_size = block_state.batch_size * 2
    block_state.image_latents = torch.randn(1, 4, 64, 64)
    return block_state

image_encoder_block_cls = make_block(
    inputs=[
        InputParam(name="image", type_hint="PIL.Image", description="raw input image to process")
    ],
    intermediate_inputs=[
        InputParam(name="batch_size", type_hint=int)
    ],
    intermediate_outputs=[
        OutputParam(name="image_latents", description="latents representing the image")
    ],
    block_fn=image_encoder_block_fn,
    description="Encode raw image into its latent presentation"
)
image_encoder_block = image_encoder_block_cls()
```

Now let's connect these blocks to create a `SequentialPipelineBlocks`:

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks, InsertableDict

# Define a dict mapping block names to block instances
blocks_dict = InsertableDict()
blocks_dict["input"] = input_block
blocks_dict["image_encoder"] = image_encoder_block

# Create the SequentialPipelineBlocks
blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)
```

Now you have a `SequentialPipelineBlocks` with 2 blocks:

```py
>>> blocks
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  Description: 


  Sub-Blocks:
    [0] input (TestBlock)
       Description: A block that determines batch_size based on the number of prompts and num_images_per_prompt argument.

    [1] image_encoder (TestBlock)
       Description: Encode raw image into its latent presentation

)
```

When you inspect `blocks.doc`, you can see that `batch_size` is not listed as an input. The pipeline automatically detects that the `input_block` can produce `batch_size` for the `image_encoder_block`, so it doesn't ask the user to provide it.

```py
>>> print(blocks.doc)
class SequentialPipelineBlocks

  Inputs:

      prompt (`None`, *optional*):

      num_images_per_prompt (`None`, *optional*):

      image (`PIL.Image`, *optional*):
          raw input image to process

  Outputs:

      batch_size (`None`):

      image_latents (`None`):
          latents representing the image
```

At runtime, you have data flow like this:

![Data Flow Diagram](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/Editor%20_%20Mermaid%20Chart-2025-06-30-092631.png)

**How SequentialPipelineBlocks Works:**

1. Blocks are executed in the order they're registered in the `blocks_dict`
2. Outputs from one block become available as intermediate inputs to all subsequent blocks
3. The pipeline automatically figures out which values need to be provided by the user and which will be generated by previous blocks
4. Each block maintains its own behavior and operates through its defined interface, while collectively these interfaces determine what the entire pipeline accepts and produces

What happens within each block follows the same pattern we described earlier: each block gets its own `block_state` with the relevant inputs and intermediate inputs, performs its computation, and updates the pipeline state with its intermediate outputs.