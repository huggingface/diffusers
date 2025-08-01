<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PipelineState and BlockState

<Tip warning={true}>

🧪 **Experimental Feature**: Modular Diffusers is an experimental feature we are actively developing. The API may be subject to breaking changes.

</Tip>

In Modular Diffusers, `PipelineState` and `BlockState` are the core data structures that enable blocks to communicate and share data. The concept is fundamental to understand how blocks interact with each other and the pipeline system.

In the modular diffusers system, `PipelineState` acts as the global state container that all pipeline blocks operate on. It maintains the complete runtime state of the pipeline and provides a structured way for blocks to read from and write to shared data.

A `PipelineState` consists of two distinct states:

- **The immutable state** (i.e. the `inputs` dict) contains a copy of values provided by users. Once a value is added to the immutable state, it cannot be changed. Blocks can read from the immutable state but cannot write to it.

- **The mutable state** (i.e. the `intermediates` dict) contains variables that are passed between blocks and can be modified by them.

Here's an example of what a `PipelineState` looks like:

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

Each pipeline blocks define what parts of that state they can read from and write to through their `inputs`, `intermediate_inputs`, and `intermediate_outputs` properties. At run time, they gets a local view (`BlockState`) of the relevant variables it needs from `PipelineState`, performs its operations, and then updates `PipelineState` with any changes.

For example, if a block defines an input `image`, inside the block's `__call__` method, the `BlockState` would contain:

```py
BlockState(
    image: <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3ECC494640>
)
```

You can access the variables directly as attributes: `block_state.image`.

We will explore more on how blocks interact with pipeline state through their `inputs`, `intermediate_inputs`, and `intermediate_outputs` properties, see the [PipelineBlock guide](./pipeline_block.md).