<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# States

Blocks rely on the [`~modular_pipelines.PipelineState`] and [`~modular_pipelines.BlockState`] data structures for communicating and sharing data.

| State | Description |
|-------|-------------|
| [`~modular_pipelines.PipelineState`] | Maintains the overall data required for a pipeline's execution and allows blocks to read and update its data. |
| [`~modular_pipelines.BlockState`] | Allows each block to perform its computation with the necessary data from `inputs` and `intermediate_inputs` |

This guide explains how states work and how they connect blocks.

## PipelineState

The [`~modular_pipelines.PipelineState`] is a global state container for all blocks. It maintains the complete runtime state of the pipeline and provides a structured way for blocks to read from and write to shared data.

There are two dict's in [`~modular_pipelines.PipelineState`] for structuring data.

- The `inputs` dict is an **immutable** state containing a copy of user provided values. A value added to `inputs` cannot be changed. Blocks can read from `inputs` but cannot write to it.
- The `intermediates` dict is a **mutable** state containing variables that are passed between blocks and can be modified by them.

```py
PipelineState(
  inputs={
    'prompt': 'a cat'
    'guidance_scale': 7.0
    'num_inference_steps': 25
  },
  intermediates={
    'prompt_embeds': Tensor(dtype=torch.float32, shape=torch.Size([1, 1, 1, 1]))
    'negative_prompt_embeds': None
  },
)
```

## BlockState

The [`~modular_pipelines.BlockState`] is a local view of the relevant variables, `inputs` and `intermediate_inputs`, an individual block needs from [`~modular_pipelines.PipelineState`] for performing it's computations.

Access these variables directly as attributes like `block_state.image`.

```py
BlockState(
    image: <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3ECC494640>
)
```

When a block's `__call__` method is executed, it retrieves the [`BlockState`] with `self.get_block_state(state)`, performs it's operations, and updates [`~modular_pipelines.PipelineState`] with `self.set_block_state(state, block_state)`.

```py
def __call__(self, components, state):
    # retrieve BlockState
    block_state = self.get_block_state(state)
    
    # computation logic on inputs and intermediate_inputs
    
    # update PipelineState
    self.set_block_state(state, block_state)
    return components, state
```

## State interaction

[`~modular_pipelines.PipelineState`] and [`~modular_pipelines.BlockState`] interaction is defined by a block's `inputs`, `intermediate_inputs`, and `intermediate_outputs`.

- `inputs`, a block can modify an input - like `block_state.image` - but the change is local to the [`~modular_pipelines.BlockState`] and won't affect the original input in [`~modular_pipelines.PipelineState`].
- `intermediate_inputs`, is often values created from a previous block. When a block modifies `intermediate_inputs` - like `batch_size` - this change is reflected in both the [`~modular_pipelines.BlockState`] and [`~modular_pipelines.PipelineState`]. Any subsequent blocks are also affected.

  If a previous block doesn't provide an `intermediate_inputs`, then the pipeline makes it available as a user input. However, the value is still a mutable intermediate state.

- `intermediate_outputs`, is a new variable that a block creates from `intermediate_inputs`. It is added to the [`~modular_pipelines.PipelineState`]'s `intermediates` dict and available as an `intermediate_inputs` for subsequent blocks or accessed by users as a final output from the pipeline.

  If a variable is modified in `block_state` but not declared as an `intermediate_outputs`, it won't be added to [`~modular_pipelines.PipelineState`].