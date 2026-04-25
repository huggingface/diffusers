<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ModularPipelineBlocks

[`~modular_pipelines.ModularPipelineBlocks`] is the basic block for building a [`ModularPipeline`]. It defines what components, inputs/outputs, and computation a block should perform for a specific step in a pipeline. A [`~modular_pipelines.ModularPipelineBlocks`] connects with other blocks, using [state](./modular_diffusers_states), to enable the modular construction of workflows.

A [`~modular_pipelines.ModularPipelineBlocks`] on it's own can't be executed. It is a blueprint for what a step should do in a pipeline. To actually run and execute a pipeline, the [`~modular_pipelines.ModularPipelineBlocks`] needs to be converted into a [`ModularPipeline`].

This guide will show you how to create a [`~modular_pipelines.ModularPipelineBlocks`].

## Inputs and outputs

> [!TIP]
> Refer to the [States](./modular_diffusers_states) guide if you aren't familiar with how state works in Modular Diffusers.

A [`~modular_pipelines.ModularPipelineBlocks`] requires `inputs`, and `intermediate_outputs`.

- `inputs` are values a block reads from the [`~modular_pipelines.PipelineState`] to perform its computation. These can be values provided by a user (like a prompt or image) or values produced by a previous block (like encoded `image_latents`). 

    Use `InputParam` to define `inputs`.

```py
class ImageEncodeStep(ModularPipelineBlocks):
    ...

    @property
    def inputs(self):
        return [
            InputParam(name="image", type_hint="PIL.Image", required=True, description="raw input image to process"),
        ]
    ...
```

- `intermediate_outputs` are new values created by a block and added to the [`~modular_pipelines.PipelineState`]. The `intermediate_outputs` are available as `inputs` for subsequent blocks or available as the final output from running the pipeline.

    Use `OutputParam` to define `intermediate_outputs`.

```py
class ImageEncodeStep(ModularPipelineBlocks):
    ...

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name="image_latents", description="latents representing the image"),
        ]

    ...
```

The intermediate inputs and outputs share data to connect blocks. They are accessible at any point, allowing you to track the workflow's progress.

## Components and configs

The components and pipeline-level configs a block needs are specified in [`ComponentSpec`] and [`~modular_pipelines.ConfigSpec`].

- [`ComponentSpec`] contains the expected components used by a block. You need the `name` of the component and ideally a `type_hint` that specifies exactly what the component is.
- [`~modular_pipelines.ConfigSpec`] contains pipeline-level settings that control behavior across all blocks.

```py
class ImageEncodeStep(ModularPipelineBlocks):
    ...

    @property
    def expected_components(self):
        return [
            ComponentSpec(name="vae", type_hint=AutoencoderKL),
        ]

    @property
    def expected_configs(self):
        return [
            ConfigSpec("force_zeros_for_empty_prompt", True),
        ]

    ...
```

When the blocks are converted into a pipeline, the components become available to the block as the first argument in `__call__`.

## Computation logic

The computation a block performs is defined in the `__call__` method and it follows a specific structure.

1. Retrieve the [`~modular_pipelines.BlockState`] to get a local view of the `inputs`.
2. Implement the computation logic on the `inputs`.
3. Update [`~modular_pipelines.PipelineState`] to push changes from the local [`~modular_pipelines.BlockState`] back to the global [`~modular_pipelines.PipelineState`].
4. Return the components and state which becomes available to the next block.

```py
class ImageEncodeStep(ModularPipelineBlocks):

    def __call__(self, components, state):
        # Get a local view of the state variables this block needs
        block_state = self.get_block_state(state)

        # Your computation logic here
        # block_state contains all your inputs
        # Access them like: block_state.image, block_state.processed_image

        # Update the pipeline state with your updated block_states
        self.set_block_state(state, block_state)
        return components, state
```

## Putting it all together

Here is the complete block with all the pieces connected.

```py
from diffusers import ComponentSpec, AutoencoderKL
from diffusers.modular_pipelines import InputParam, ModularPipelineBlocks, OutputParam


class ImageEncodeStep(ModularPipelineBlocks):

    @property
    def description(self):
        return "Encode an image into latent space."

    @property
    def expected_components(self):
        return [
            ComponentSpec(name="vae", type_hint=AutoencoderKL),
        ]

    @property
    def inputs(self):
        return [
            InputParam(name="image", type_hint="PIL.Image", required=True, description="raw input image to process"),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name="image_latents", type_hint="torch.Tensor", description="latents representing the image"),
        ]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.image_latents = components.vae.encode(block_state.image)
        self.set_block_state(state, block_state)
        return components, state
```

Every block has a `doc` property that is automatically generated from the properties you defined above. It provides a summary of the block's description, components, inputs, and outputs.

```py
block = ImageEncoderStep()
print(block.doc)
class ImageEncodeStep

  Encode an image into latent space.

  Components:
      vae (`AutoencoderKL`)

  Inputs:
      image (`PIL.Image`):
          raw input image to process

  Outputs:
      image_latents (`torch.Tensor`):
          latents representing the image
```