<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PipelineBlock

<Tip warning={true}>

🧪 **Experimental Feature**: Modular Diffusers is an experimental feature we are actively developing. The API may be subject to breaking changes.

</Tip>

In Modular Diffusers, you build your workflow using `ModularPipelineBlocks`. We support 4 different types of blocks: `PipelineBlock`, `SequentialPipelineBlocks`, `LoopSequentialPipelineBlocks`, and `AutoPipelineBlocks`. Among them, `PipelineBlock` is the most fundamental building block of the whole system - it's like a brick in a Lego system. These blocks are designed to easily connect with each other, allowing for modular construction of creative and potentially very complex workflows.

<Tip>

**Important**: `PipelineBlock`s are definitions/specifications, not runnable pipelines. They define what a block should do and what data it needs, but you need to convert them into a `ModularPipeline` to actually execute them. For information on creating and running pipelines, see the [Modular Pipeline guide](./modular_pipeline.md).

</Tip>

In this tutorial, we will focus on how to write a basic `PipelineBlock` and how it interacts with the pipeline state.

## PipelineState

Before we dive into creating `PipelineBlock`s, make sure you have a basic understanding of `PipelineState`. It acts as the global state container that all blocks operate on - each block gets a local view (`BlockState`) of the relevant variables it needs from `PipelineState`, performs its operations, and then updates `PipelineState` with any changes. See the [PipelineState and BlockState guide](./modular_diffusers_states.md) for more details.

## Define a `PipelineBlock`

To write a `PipelineBlock` class, you need to define a few properties that determine how your block interacts with the pipeline state. Understanding these properties is crucial - they define what data your block can access and what it can produce.

The three main properties you need to define are:
- `inputs`: Immutable values from the user that cannot be modified
- `intermediate_inputs`: Mutable values from previous blocks that can be read and modified  
- `intermediate_outputs`: New values your block creates for subsequent blocks and user access

Let's explore each one and understand how they work with the pipeline state.

**Inputs: Immutable User Values**

Inputs are variables your block needs from the immutable pipeline state - these are user-provided values that cannot be modified by any block. You define them using `InputParam`:

```py
user_inputs = [
    InputParam(name="image", type_hint="PIL.Image", description="raw input image to process")
]
```

When you list something as an input, you're saying "I need this value directly from the end user, and I will talk to them directly, telling them what I need in the 'description' field. They will provide it and it will come to me unchanged."

This is especially useful for raw values that serve as the "source of truth" in your workflow. For example, with a raw image, many workflows require preprocessing steps like resizing that a previous block might have performed. But in many cases, you also want the raw PIL image. In some inpainting workflows, you need the original image to overlay with the generated result for better control and consistency.

**Intermediate Inputs: Mutable Values from Previous Blocks, or Users**

Intermediate inputs are variables your block needs from the mutable pipeline state - these are values that can be read and modified. They're typically created by previous blocks, but could also be directly provided by the user if not the case:

```py
user_intermediate_inputs = [
    InputParam(name="processed_image", type_hint="torch.Tensor", description="image that has been preprocessed and normalized"),
]
```

When you list something as an intermediate input, you're saying "I need this value, but I want to work with a different block that has already created it. I already know for sure that I can get it from this other block, but it's okay if other developers want use something different."

**Intermediate Outputs: New Values for Subsequent Blocks and User Access**

Intermediate outputs are new variables your block creates and adds to the mutable pipeline state. They serve two purposes:

1. **For subsequent blocks**: They can be used as intermediate inputs by other blocks in the pipeline
2. **For users**: They become available as final outputs that users can access when running the pipeline

```py
user_intermediate_outputs = [
    OutputParam(name="image_latents", description="latents representing the image")
]
```

Intermediate inputs and intermediate outputs work together like Lego studs and anti-studs - they're the connection points that make blocks modular. When one block produces an intermediate output, it becomes available as an intermediate input for subsequent blocks. This is where the "modular" nature of the system really shines - blocks can be connected and reconnected in different ways as long as their inputs and outputs match.

Additionally, all intermediate outputs are accessible to users when they run the pipeline, typically you would only need the final images, but they are also able to access intermediate results like latents, embeddings, or other processing steps.

**The `__call__` Method Structure**

Your `PipelineBlock`'s `__call__` method should follow this structure:

```py
def __call__(self, components, state):
    # Get a local view of the state variables this block needs
    block_state = self.get_block_state(state)
    
    # Your computation logic here
    # block_state contains all your inputs and intermediate_inputs
    # You can access them like: block_state.image, block_state.processed_image
    
    # Update the pipeline state with your updated block_states
    self.set_block_state(state, block_state)
    return components, state
```

The `block_state` object contains all the variables you defined in `inputs` and `intermediate_inputs`, making them easily accessible for your computation.

**Components and Configs**

You can define the components and pipeline-level configs your block needs using `ComponentSpec` and `ConfigSpec`:

```py
from diffusers import ComponentSpec, ConfigSpec

# Define components your block needs
expected_components = [
    ComponentSpec(name="unet", type_hint=UNet2DConditionModel),
    ComponentSpec(name="scheduler", type_hint=EulerDiscreteScheduler)
]

# Define pipeline-level configs
expected_config = [
    ConfigSpec("force_zeros_for_empty_prompt", True)
]
```

**Components**: In the `ComponentSpec`, you must provide a `name` and ideally a `type_hint`. You can also specify a `default_creation_method` to indicate whether the component should be loaded from a pretrained model or created with default configurations. The actual loading details (`repo`, `subfolder`, `variant` and `revision` fields) are typically specified when creating the pipeline, as we covered in the [Modular Pipeline Guide](./modular_pipeline.md).

**Configs**: Pipeline-level settings that control behavior across all blocks.

When you convert your blocks into a pipeline using `blocks.init_pipeline()`, the pipeline collects all component requirements from the blocks and fetches the loading specs from the modular repository. The components are then made available to your block as the first argument of the `__call__` method. You can access any component you need using dot notation:

```py
def __call__(self, components, state):
    # Access components using dot notation
    unet = components.unet
    vae = components.vae
    scheduler = components.scheduler
```

That's all you need to define in order to create a `PipelineBlock`. There is no hidden complexity. In fact we are going to create a helper function that take exactly these variables as input and return a pipeline block. We will use this helper function through out the tutorial to create test blocks

Note that for `__call__` method, the only part you should implement differently is the part between `self.get_block_state()` and `self.set_block_state()`, which can be abstracted into a simple function that takes `block_state` and returns the updated state. Our helper function accepts a `block_fn` that does exactly that.

**Helper Function**

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

## Example: Creating a Simple Pipeline Block

Let's create a simple block to see how these definitions interact with the pipeline state. To better understand what's happening, we'll print out the states before and after updates to inspect them:

```py
inputs = [
    InputParam(name="image", type_hint="PIL.Image", description="raw input image to process")
]

intermediate_inputs = [InputParam(name="batch_size", type_hint=int)]

intermediate_outputs = [
    OutputParam(name="image_latents", description="latents representing the image")
]

def image_encoder_block_fn(block_state, pipeline_state):
    print(f"pipeline_state (before update): {pipeline_state}")
    print(f"block_state (before update): {block_state}")
    
    # Simulate processing the image
    block_state.image = torch.randn(1, 3, 512, 512)
    block_state.batch_size = block_state.batch_size * 2
    block_state.processed_image = [torch.randn(1, 3, 512, 512)] * block_state.batch_size
    block_state.image_latents = torch.randn(1, 4, 64, 64)
    
    print(f"block_state (after update): {block_state}")
    return block_state

# Create a block with our definitions
image_encoder_block_cls = make_block(
    inputs=inputs, 
    intermediate_inputs=intermediate_inputs,
    intermediate_outputs=intermediate_outputs, 
    block_fn=image_encoder_block_fn,
    description="Encode raw image into its latent presentation"
)
image_encoder_block = image_encoder_block_cls()
pipe = image_encoder_block.init_pipeline()
```

Let's check the pipeline's docstring to see what inputs it expects:
```py
>>> print(pipe.doc)
class TestBlock

  Encode raw image into its latent presentation

  Inputs:

      image (`PIL.Image`, *optional*):
          raw input image to process

      batch_size (`int`, *optional*):

  Outputs:

      image_latents (`None`):
          latents representing the image
```

Notice that `batch_size` appears as an input even though we defined it as an intermediate input. This happens because no previous block provided it, so the pipeline makes it available as a user input. However, unlike regular inputs, this value goes directly into the mutable intermediate state.

Now let's run the pipeline:

```py
from diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png")
state = pipe(image=image, batch_size=2)
print(f"pipeline_state (after update): {state}")
```
```out
pipeline_state (before update): PipelineState(
  inputs={
    image: <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3ECC494550>
  },
  intermediates={
    batch_size: 2
  },
)
block_state (before update): BlockState(
    image: <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3ECC494640>
    batch_size: 2
)

block_state (after update): BlockState(
    image: Tensor(dtype=torch.float32, shape=torch.Size([1, 3, 512, 512]))
    batch_size: 4
    processed_image: List[4] of Tensors with shapes [torch.Size([1, 3, 512, 512]), torch.Size([1, 3, 512, 512]), torch.Size([1, 3, 512, 512]), torch.Size([1, 3, 512, 512])]
    image_latents: Tensor(dtype=torch.float32, shape=torch.Size([1, 4, 64, 64]))
)
pipeline_state (after update): PipelineState(
  inputs={
    image: <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3ECC494550>
  },
  intermediates={
    batch_size: 4
    image_latents: Tensor(dtype=torch.float32, shape=torch.Size([1, 4, 64, 64]))
  },
)
```

**Key Observations:**

1. **Before the update**: `image` (the input) goes to the immutable inputs dict, while `batch_size` (the intermediate_input) goes to the mutable intermediates dict, and both are available in `block_state`.

2. **After the update**:
   - **`image` (inputs)** changed in `block_state` but not in `pipeline_state` - this change is local to the block only. 
   - **`batch_size (intermediate_inputs)`** was updated in both `block_state` and `pipeline_state` - this change affects subsequent blocks (we didn't need to declare it as an intermediate output since it was already in the intermediates dict)
   - **`image_latents (intermediate_outputs)`** was added to `pipeline_state` because it was declared as an intermediate output
   - **`processed_image`** was not added to `pipeline_state` because it wasn't declared as an intermediate output