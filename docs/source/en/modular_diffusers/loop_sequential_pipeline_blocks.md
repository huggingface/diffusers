<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoopSequentialPipelineBlocks

<Tip warning={true}>

🧪 **Experimental Feature**: Modular Diffusers is an experimental feature we are actively developing. The API may be subject to breaking changes.

</Tip>

`LoopSequentialPipelineBlocks` is a subclass of `ModularPipelineBlocks`. It is a multi-block that composes other blocks together in a loop, creating iterative workflows where blocks run multiple times with evolving state. It's particularly useful for denoising loops requiring repeated execution of the same blocks.

<Tip>

Other types of multi-blocks include [SequentialPipelineBlocks](./sequential_pipeline_blocks.md) (for linear workflows) and [AutoPipelineBlocks](./auto_pipeline_blocks.md) (for conditional block selection). For information on creating individual blocks, see the [PipelineBlock guide](./pipeline_block.md).

Additionally, like all `ModularPipelineBlocks`, `LoopSequentialPipelineBlocks` are definitions/specifications, not runnable pipelines. You need to convert them into a `ModularPipeline` to actually execute them. For information on creating and running pipelines, see the [Modular Pipeline guide](modular_pipeline.md).

</Tip>

You could create a loop using `PipelineBlock` like this:

```python
class DenoiseLoop(PipelineBlock):
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        for t in range(block_state.num_inference_steps):
            # ... loop logic here
            pass
        self.set_block_state(state, block_state)
        return components, state
```

But in this tutorial, we will focus on how to use `LoopSequentialPipelineBlocks` to create a "composable" denoising loop where you can add or remove blocks within the loop or reuse the same loop structure with different block combinations.

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
        self.set_block_state(state, block_state)
        return components, state
```

**Step 2: Create Loop Blocks**

Loop blocks are standard `PipelineBlock`s, but their `__call__` method works differently:
* It receives the iteration variable (e.g., `i`) passed by the loop wrapper
* It works directly with `block_state` instead of pipeline state
* No need to call `self.get_block_state()` or `self.set_block_state()`

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