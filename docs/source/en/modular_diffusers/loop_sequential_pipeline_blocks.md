<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoopSequentialPipelineBlocks

[`~modular_pipelines.LoopSequentialPipelineBlocks`] are a multi-block type that composes other [`~modular_pipelines.ModularPipelineBlocks`] together in a loop. Data flows circularly, using `intermediate_inputs` and `intermediate_outputs`, and each block is run iteratively. This is typically used to create a denoising loop which is iterative by default.

This guide shows you how to create [`~modular_pipelines.LoopSequentialPipelineBlocks`].

## Loop wrapper

[`~modular_pipelines.LoopSequentialPipelineBlocks`], is also known as the *loop wrapper* because it defines the loop structure, iteration variables, and configuration. Within the loop wrapper, you need the following variables.

- `loop_inputs` are user provided values and equivalent to [`~modular_pipelines.ModularPipelineBlocks.inputs`].
- `loop_intermediate_inputs` are intermediate variables from the [`~modular_pipelines.PipelineState`] and equivalent to [`~modular_pipelines.ModularPipelineBlocks.intermediate_inputs`].
- `loop_intermediate_outputs` are new intermediate variables created by the block and added to the [`~modular_pipelines.PipelineState`]. It is equivalent to [`~modular_pipelines.ModularPipelineBlocks.intermediate_outputs`].
- `__call__` method defines the loop structure and iteration logic.

```py
import torch
from diffusers.modular_pipelines import LoopSequentialPipelineBlocks, ModularPipelineBlocks, InputParam, OutputParam

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

The loop wrapper can pass additional arguments, like current iteration index, to the loop blocks.

## Loop blocks

A loop block is a [`~modular_pipelines.ModularPipelineBlocks`], but the `__call__` method behaves differently.

- It recieves the iteration variable from the loop wrapper.
- It works directly with the [`~modular_pipelines.BlockState`] instead of the [`~modular_pipelines.PipelineState`].
- It doesn't require retrieving or updating the [`~modular_pipelines.BlockState`].

Loop blocks share the same [`~modular_pipelines.BlockState`] to allow values to accumulate and change for each iteration in the loop.

```py
class LoopBlock(ModularPipelineBlocks):
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

## LoopSequentialPipelineBlocks

Use the [`~modular_pipelines.LoopSequentialPipelineBlocks.from_blocks_dict`] method to add the loop block to the loop wrapper to create [`~modular_pipelines.LoopSequentialPipelineBlocks`].

```py
loop = LoopWrapper.from_blocks_dict({"block1": LoopBlock})
```

Add more loop blocks to run within each iteration with [`~modular_pipelines.LoopSequentialPipelineBlocks.from_blocks_dict`]. This allows you to modify the blocks without changing the loop logic itself.

```py
loop = LoopWrapper.from_blocks_dict({"block1": LoopBlock(), "block2": LoopBlock})
```