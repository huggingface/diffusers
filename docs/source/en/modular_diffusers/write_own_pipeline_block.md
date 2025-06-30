<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# `ModularPipelineBlocks`

In Modular Diffusers, you build your workflow using `ModularPipelineBlocks`. We support 4 different types of blocks: `PipelineBlock`, `SequentialPipelineBlocks`, `LoopSequentialPipelineBlocks`, and `AutoPipelineBlocks`. Among them, `PipelineBlock` is the most fundamental building block of the whole system - it's like a brick in a Lego system. These blocks are designed to easily connect with each other, allowing for modular construction of creative and potentially very complex workflows.

In this tutorial, we will focus on how to write a basic `PipelineBlock` and how it interacts with other components in the system. We will also cover how to connect them together using the multi-blocks: `SequentialPipelineBlocks`, `LoopSequentialPipelineBlocks`, and `AutoPipelineBlocks`.


## Understanding the Foundation: `PipelineState`

Before we dive into creating `PipelineBlock`s, we need to have a basic understanding of `PipelineState` - the core data structure that all blocks operate on. This concept is fundamental to understanding how blocks interact with each other and the pipeline system.

In the modular diffusers system, `PipelineState` acts as the global state container that `PipelineBlock`s operate on - each block gets a local view (`BlockState`) of the relevant variables it needs from `PipelineState`, performs its operations, and then updates `PipelineState` with any changes.

While `PipelineState` maintains the complete runtime state of the pipeline, `PipelineBlock`s define what parts of that state they can read from and write to through their `input`s, `intermediates_inputs`, and `intermediates_outputs` properties.

A `PipelineState` consists of two distinct states:
- The **immutable state** (i.e. the `inputs` dict) contains a copy of values provided by users. Once a value is added to the immutable state, it cannot be changed. Blocks can read from the immutable state but cannot write to it.
- The **mutable state** (i.e. the `intermediates` dict) contains variables that are passed between blocks and can be modified by them.

Here's an example of what a `PipelineState` looks like:

```
PipelineState(
  inputs={
    prompt: 'a cat'
    guidance_scale: 7.0
    num_inference_steps: 25
  },
  intermediates={
    prompt_embeds: Tensor(dtype=torch.float32, shape=torch.Size([1, 1, 1, 1]))
    negative_prompt_embeds: None
  },
```

## Creating a `PipelineBlock`

To write a `PipelineBlock` class, you need to define a few properties that determine how your block interacts with the pipeline state. Understanding these properties is crucial - they define what data your block can access and what it can produce.

The three main properties you need to define are:
- `inputs`: Immutable values from the user that cannot be modified
- `intermediate_inputs`: Mutable values from previous blocks that can be read and modified  
- `intermediate_outputs`: New values your block creates for subsequent blocks

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

**Intermediate Inputs: Mutable Values from Previous Blocks**

Intermediate inputs are variables your block needs from the mutable pipeline state - these are values that can be read and modified. They're typically created by previous blocks, but could also be directly provided by the user if not the case:

```py
user_intermediate_inputs = [
    InputParam(name="processed_image", type_hint="torch.Tensor", description="image that has been preprocessed and normalized"),
]
```

When you list something as an intermediate input, you're saying "I need this value, but I want to work with a different block that has already created it. I already know for sure that I can get it from this other block, but it's okay if other developers want use something different."

**Intermediate Outputs: New Values for Subsequent Blocks**

Intermediate outputs are new variables your block creates and adds to the mutable pipeline state so they can be used by subsequent blocks:

```py
user_intermediate_outputs = [
    OutputParam(name="image_latents", description="latents representing the image")
]
```

Intermediate inputs and intermediate outputs work together like Lego studs and anti-studs - they're the connection points that make blocks modular. When one block produces an intermediate output, it becomes available as an intermediate input for subsequent blocks. This is where the "modular" nature of the system really shines - blocks can be connected and reconnected in different ways as long as their inputs and outputs match. We will see more how they connect when we talk about multi-blocks.

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
    self.add_block_state(state, block_state)
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

**Components**: You must provide a `name` and ideally a `type_hint`. The actual loading details (`repo`, `subfolder`, `variant` and `revision` fields) are typically specified when creating the pipeline, as we covered in the [quicktour](quicktour.md#loading-components-into-a-modularpipeline).

**Configs**: Simple pipeline-level settings that control behavior across all blocks.

When you convert your blocks into a pipeline using `blocks.init_pipeline()`, the pipeline collects all component requirements from the blocks and fetches the loading specs from the modular repository. The components are then made available to your block in the `components` argument of the `__call__` method.

That's all you need to define in order to create a `PipelineBlock`. There is no hidden complexity. In fact we are going to create a helper function that take exactly these variables as input and return a pipeline block. We will use this helper function through out the tutorial to create test blocks

Note that for `__call__` method, the only part you should implement differently is the part between `self.get_block_state()` and `self.add_block_state()`, which can be abstracted into a simple function that takes `block_state` and returns the updated state. Our helper function accepts a `block_fn` that does exactly that.

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
            self.add_block_state(state, block_state)
            return components, state
    
    return TestBlock
```


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
    description=" Encode raw image into its latent presentation"
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

I hope by now you have a basic idea about how `PipelineBlock` manages state through inputs, intermediate inputs, and intermediate outputs. The real power comes when we connect multiple blocks together - their intermediate outputs become intermediate inputs for subsequent blocks, creating modular workflows. Let's explore how to build these connections using multi-blocks like `SequentialPipelineBlocks`.

## Create a `SequentialPipelineBlocks`

I assume that you're already familiar with `SequentialPipelineBlocks` and how to create them with the `from_blocks_dict` API. It's one of the most common ways to use Modular Diffusers, and we've covered it pretty well in the [Getting Started Guide](https://moon-ci-docs.huggingface.co/docs/diffusers/pr_9672/en/modular_diffusers/quicktour#modularpipelineblocks).

But how do blocks actually connect and work together? Understanding this is crucial for building effective modular workflows. Let's explore this through an example.

**How Blocks Connect in SequentialPipelineBlocks:**

The key insight is that blocks connect through their intermediate inputs and outputs - the "studs and anti-studs" we discussed earlier. Let's expand on our example to create a new block that produces `batch_size`, which we'll call "input_block":

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

Now let's connect these blocks to create a pipeline:

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks, InsertableDict
# define a dict map block names to block class
blocks_dict = InsertableDict()
blocks_dict["input"] = input_block
blocks_dict["image_encoder"] = image_encoder_block
# create the multi-block
blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)
# convert it to a runnable pipeline
pipeline = blocks.init_pipeline()
```

Now you have a pipeline with 2 blocks. 

```py
>>> pipeline.blocks
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  Description: 


  Sub-Blocks:
    [0] input (TestBlock)
       Description: A block that determines batch_size based on the number of prompts and num_images_per_prompt argument.

    [1] image_encoder (TestBlock)
       Description:  Encode raw image into its latent presentation

)
```

When you inspect `pipeline.doc`, you can see that `batch_size` is not listed as an input. The pipeline automatically detects that the `input_block` can produce `batch_size` for the `image_encoder_block`, so it doesn't ask the user to provide it.

```py
>>> print(pipeline.doc)
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

## `LoopSequentialPipelineBlocks`

To create a loop in Modular Diffusers, you could use a single `PipelineBlock` like this:

```python
class DenoiseLoop(PipelineBlock):
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        for t in range(block_state.num_inference_steps):
            # ... loop logic here
            pass
        self.add_block_state(state, block_state)
        return components, state
```

Or you could create a `LoopSequentialPipelineBlocks`. The key difference is that with `LoopSequentialPipelineBlocks`, the loop itself is modular: you can add or remove blocks within the loop or reuse the same loop structure with different block combinations.

It involves two parts: a **loop wrapper** and **loop blocks**

* The **loop wrapper** (`LoopSequentialPipelineBlocks`) defines the loop structure, e.g. it defines the iteration variables, and loop configurations such as progress bar.

* The **loop blocks** are basically standard pipeline blocks you add to the loop wrapper.
  - they run sequentially for each iteration of the loop
  - they receive the current iteration index as an additional parameter
  - they share the same block_state throughout the entire loop

Unlike regular `SequentialPipelineBlocks` where each block gets its own state, loop blocks share a single state that persists and evolves across iterations.

We will build a simple loop block to demonstrate these concepts. Creating a loop block involves three steps:
1. defining the loop wrapper class
2. creating the loop blocks
3. adding the loop blocks to the loop wrapper class to create the loop wrapper instance

**Step 1: Define the Loop Wrapper**

To create a `LoopSequentialPipelineBlocks` class, you need to define:

* `loop_inputs`: User input variables (equivalent to `PipelineBlock.inputs`)
* `loop_intermediate_inputs`: Intermediate variables needed from the mutable pipeline state (equivalent to `PipelineBlock.intermediates_inputs`)
* `loop_intermediate_outputs`: New intermediate variables this block will add to the mutable pipeline state (equivalent to `PipelineBlock.intermediates_outputs`)
* `__call__` method: Defines the loop structure and iteration logic

Here is an example of a loop wrapper:

```py
import torch
from diffusers.modular_pipelines import LoopSequentialPipelineBlocks, PipelineBlock, InputParam, OutputParam

class LoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "test"
    @property
    def description(self):
        return "I'm a loop!!"
    @property
    def loop_inputs(self):
        return [InputParam(name="num_steps")]
    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        # Loop structure - can be customized to your needs
        for i in range(block_state.num_steps):
            # loop_step executes all registered blocks in sequence
            components, block_state = self.loop_step(components, block_state, i=i)
        self.add_block_state(state, block_state)
        return components, state
```

**Step 2: Create Loop Blocks**

Loop blocks are standard `PipelineBlock`s, but their `__call__` method works differently:
* It receives the iteration variable (e.g., `i`) passed by the loop wrapper
* It works directly with `block_state` instead of pipeline state
* No need to call `self.get_block_state()` or `self.add_block_state()`

```py
class LoopBlock(PipelineBlock):
    # this is used to identify the model family, we won't worry about it in this example
    model_name = "test"
    @property
    def inputs(self):
        return [InputParam(name="x")]
    @property
    def intermediate_outputs(self):
        # outputs produced by this block
        return [OutputParam(name="x")]
    @property
    def description(self):
        return "I'm a block used inside the `LoopWrapper` class"
    def __call__(self, components, block_state, i: int):
        block_state.x += 1
        return components, block_state
```

**Step 3: Combine Everything**

Finally, assemble your loop by adding the block(s) to the wrapper:

```py
loop = LoopWrapper.from_blocks_dict({"block1": LoopBlock})
```

Now you've created a loop with one step:

```py
>>> loop
LoopWrapper(
  Class: LoopSequentialPipelineBlocks

  Description: I'm a loop!!

  Sub-Blocks:
    [0] block1 (LoopBlock)
       Description: I'm a block used inside the `LoopWrapper` class

)
```

It has two inputs: `x` (used at each step within the loop) and `num_steps` used to define the loop.

```py
>>> print(loop.doc)
class LoopWrapper

  I'm a loop!!

  Inputs:

      x (`None`, *optional*):

      num_steps (`None`, *optional*):

  Outputs:

      x (`None`):
```

**Running the Loop:**

```py
# run the loop
loop_pipeline = loop.init_pipeline()
x = loop_pipeline(num_steps=10, x=0, output="x")
assert x == 10
```

**Adding Multiple Blocks:**

We can add multiple blocks to run within each iteration. Let's run the loop block twice within each iteration:

```py
loop = LoopWrapper.from_blocks_dict({"block1": LoopBlock(), "block2": LoopBlock})
loop_pipeline = loop.init_pipeline()
x = loop_pipeline(num_steps=10, x=0, output="x")
assert x == 20  # Each iteration runs 2 blocks, so 10 iterations * 2 = 20
```

**Key Differences from SequentialPipelineBlocks:**

The main difference is that loop blocks share the same `block_state` across all iterations, allowing values to accumulate and evolve throughout the loop. Loop blocks could receive additional arguments (like the current iteration index) depending on the loop wrapper's implementation, since the wrapper defines how loop blocks are called. You can easily add, remove, or reorder blocks within the loop without changing the loop logic itself.

The officially supported denoising loops in Modular Diffusers are implemented using `LoopSequentialPipelineBlocks`. You can explore the actual implementation to see how these concepts work in practice:

```py
from diffusers.modular_pipelines.stable_diffusion_xl.denoise import StableDiffusionXLDenoiseStep
StableDiffusionXLDenoiseStep()
```

## `AutoPipelineBlocks`

`AutoPipelineBlocks` allows you to pack different pipelines into one and automatically select which one to run at runtime based on the inputs. The main purpose is convenience and portability - for developers, you can package everything into one workflow, making it easier to share and use.

For example, you might want to support text-to-image and image-to-image tasks. Instead of creating two separate pipelines, you can create an `AutoPipelineBlocks` that automatically chooses the workflow based on whether an `image` input is provided.

Let's see an example. Here we'll create a dummy `AutoPipelineBlocks` that includes dummy text-to-image, image-to-image, and inpaint pipelines.


```py
from diffusers.modular_pipelines import AutoPipelineBlocks 

# These are dummy blocks and we only focus on "inputs" for our purpose
inputs = [InputParam(name="prompt")]
# block_fn prints out which workflow is running so we can see the execution order at runtime
block_fn = lambda x, y: print("running the text-to-image workflow")
block_t2i_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a text-to-image workflow!")

inputs = [InputParam(name="prompt"), InputParam(name="image")]
block_fn = lambda x, y: print("running the image-to-image workflow")
block_i2i_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a image-to-image workflow!")

inputs = [InputParam(name="prompt"), InputParam(name="image"), InputParam(name="mask")]
block_fn = lambda x, y: print("running the inpaint workflow")
block_inpaint_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a inpaint workflow!")

class AutoImageBlocks(AutoPipelineBlocks):
    # List of sub-block classes to choose from
    block_classes = [block_inpaint_cls, block_i2i_cls, block_t2i_cls]
    # Names for each block in the same order
    block_names = ["inpaint", "img2img", "text2img"]
    # Trigger inputs that determine which block to run
    # - "mask" triggers inpaint workflow
    # - "image" triggers img2img workflow (but only if mask is not provided) 
    # - if none of above, runs the text2img workflow (default)
    block_trigger_inputs = ["mask", "image", None]
    # Description is extremely important for AutoPipelineBlocks
    @property
    def description(self):
        return (
            "Pipeline generates images given different types of conditions!\n"
            + "This is an auto pipeline block that works for text2img, img2img and inpainting tasks.\n"
            + " - inpaint workflow is run when `mask` is provided.\n"
            + " - img2img workflow is run when `image` is provided (but only when `mask` is not provided).\n"
            + " - text2img workflow is run when neither `image` nor `mask` is provided.\n"
        )

# Create the blocks
auto_blocks = AutoImageBlocks()
# convert to pipeline
auto_pipeline = auto_blocks.init_pipeline()
```

Now we have created an `AutoPipelineBlocks` that contains 3 sub-blocks. Notice the warning message at the top - this automatically appears in every `ModularPipelineBlocks` that contains `AutoPipelineBlocks` to remind end users that dynamic block selection happens at runtime. 

```py
AutoImageBlocks(
  Class: AutoPipelineBlocks

  ====================================================================================================
  This pipeline contains blocks that are selected at runtime based on inputs.
  Trigger Inputs: ['mask', 'image']
  ====================================================================================================


  Description: Pipeline generates images given different types of conditions!
      This is an auto pipeline block that works for text2img, img2img and inpainting tasks.
       - inpaint workflow is run when `mask` is provided.
       - img2img workflow is run when `image` is provided (but only when `mask` is not provided).
       - text2img workflow is run when neither `image` nor `mask` is provided.
      


  Sub-Blocks:
    • inpaint [trigger: mask] (TestBlock)
       Description: I'm a inpaint workflow!

    • img2img [trigger: image] (TestBlock)
       Description: I'm a image-to-image workflow!

    • text2img [default] (TestBlock)
       Description: I'm a text-to-image workflow!

)
```

Check out the documentation with `print(auto_pipeline.doc)`:

```py
>>> print(auto_pipeline.doc)
class AutoImageBlocks

  Pipeline generates images given different types of conditions!
  This is an auto pipeline block that works for text2img, img2img and inpainting tasks.
   - inpaint workflow is run when `mask` is provided.
   - img2img workflow is run when `image` is provided (but only when `mask` is not provided).
   - text2img workflow is run when neither `image` nor `mask` is provided.

  Inputs:

      prompt (`None`, *optional*):

      image (`None`, *optional*):

      mask (`None`, *optional*):
```

There is a fundamental trade-off of AutoPipelineBlocks: it trades clarity for convenience. While it is really easy for packaging multiple workflows, it can become confusing without proper documentation. e.g. if we just throw a pipeline at you and tell you that it contains 3 sub-blocks and takes 3 inputs `prompt`, `image` and `mask`, and ask you to run an image-to-image workflow: if you don't have any prior knowledge on how these pipelines work, you would be pretty clueless, right?

This pipeline we just made though, has a docstring that shows all available inputs and workflows and explains how to use each with different inputs. So it's really helpful for users. For example, it's clear that you need to pass `image` to run img2img. This is why the description field is absolutely critical for AutoPipelineBlocks. We highly recommend you to explain the conditional logic very well for each `AutoPipelineBlocks` you would make. We also recommend to always test individual pipelines first before packaging them into AutoPipelineBlocks. 

Let's run this auto pipeline with different inputs to see if the conditional logic works as described. Remember that we have added `print` in each `PipelineBlock`'s `__call__` method to print out its workflow name, so it should be easy to tell which one is running:

```py
>>> _ = auto_pipeline(image="image", mask="mask")
running the inpaint workflow
>>> _ = auto_pipeline(image="image")
running the image-to-image workflow
>>> _ = auto_pipeline(prompt="prompt")
running the text-to-image workflow
>>> _ = auto_pipeline(image="prompt", mask="mask")
running the inpaint workflow
```

However, even with documentation, it can become very confusing when AutoPipelineBlocks are combined with other blocks. The complexity grows quickly when you have nested AutoPipelineBlocks or use them as sub-blocks in larger pipelines.

Let's make another `AutoPipelineBlocks` - this one only contains one block, and it does not include `None` in its `block_trigger_inputs` (which corresponds to the default block to run when none of the trigger inputs are provided). This means this block will be skipped if the trigger input (`ip_adapter_image`) is not provided at runtime.

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks, InsertableDict
inputs = [InputParam(name="ip_adapter_image")]
block_fn = lambda x, y: print("running the ip-adapter workflow")
block_ipa_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a IP-adapter workflow!")

class AutoIPAdapter(AutoPipelineBlocks):
    block_classes = [block_ipa_cls]
    block_names = ["ip-adapter"]
    block_trigger_inputs = ["ip_adapter_image"]
    @property
    def description(self):
        return "Run IP Adapter step if `ip_adapter_image` is provided."
```

Now let's combine these 2 auto blocks together into a `SequentialPipelineBlocks`:

```py
auto_ipa_blocks = AutoIPAdapter()
blocks_dict = InsertableDict()
blocks_dict["ip-adapter"] = auto_ipa_blocks
blocks_dict["image-generation"] = auto_blocks
all_blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)
pipeline = all_blocks.init_pipeline()
```

Let's take a look: now things get more confusing. In this particular example, you could still try to explain the conditional logic in the `description` field here - there are only 4 possible execution paths so it's doable. However, since this is a `SequentialPipelineBlocks` that could contain many more blocks, the complexity can quickly get out of hand as the number of blocks increases.

```py
>>> all_blocks
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  ====================================================================================================
  This pipeline contains blocks that are selected at runtime based on inputs.
  Trigger Inputs: ['image', 'mask', 'ip_adapter_image']
  Use `get_execution_blocks()` with input names to see selected blocks (e.g. `get_execution_blocks('image')`).
  ====================================================================================================


  Description: 


  Sub-Blocks:
    [0] ip-adapter (AutoIPAdapter)
       Description: Run IP Adapter step if `ip_adapter_image` is provided.
                   

    [1] image-generation (AutoImageBlocks)
       Description: Pipeline generates images given different types of conditions!
                   This is an auto pipeline block that works for text2img, img2img and inpainting tasks.
                    - inpaint workflow is run when `mask` is provided.
                    - img2img workflow is run when `image` is provided (but only when `mask` is not provided).
                    - text2img workflow is run when neither `image` nor `mask` is provided.
                   

)

```

This is when the `get_execution_blocks()` method comes in handy - it basically extracts a `SequentialPipelineBlocks` that only contains the blocks that are actually run based on your inputs.

Let's try some examples:

`mask`: we expect it to skip the first ip-adapter since `ip_adapter_image` is not provided, and then run the inpaint for the second block.

```py
>>> all_blocks.get_execution_blocks('mask')
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  Description: 


  Sub-Blocks:
    [0] image-generation (TestBlock)
       Description: I'm a inpaint workflow!

)
```

Let's also actually run the pipeline to confirm:

```py
>>> _ = pipeline(mask="mask")
skipping auto block: AutoIPAdapter
running the inpaint workflow
```

Try a few more:

```py
print(f"inputs: ip_adapter_image:")
blocks_select = all_blocks.get_execution_blocks('ip_adapter_image')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(ip_adapter_image="ip_adapter_image", prompt="prompt")
# expect to see ip-adapter + text2img

print(f"inputs: image:")
blocks_select = all_blocks.get_execution_blocks('image')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(image="image", prompt="prompt")
# expect to see img2img

print(f"inputs: prompt:")
blocks_select = all_blocks.get_execution_blocks('prompt')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(prompt="prompt")
# expect to see text2img (prompt is not a trigger input so fallback to default)

print(f"inputs: mask + ip_adapter_image:")
blocks_select = all_blocks.get_execution_blocks('mask','ip_adapter_image')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(mask="mask", ip_adapter_image="ip_adapter_image")
# expect to see ip-adapter + inpaint
```

In summary, `AutoPipelineBlocks` is a good tool for packaging multiple workflows into a single, convenient interface and it can greatly simplify the user experience. However, always provide clear descriptions explaining the conditional logic, test individual pipelines first before combining them, and use `get_execution_blocks()` to understand runtime behavior in complex compositions.