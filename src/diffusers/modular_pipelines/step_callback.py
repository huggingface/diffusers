# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable

import torch

from .modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipeline, ModularPipelineBlocks
from .modular_pipeline_utils import InputParam


class StepCallbackBlock(ModularPipelineBlocks):
    """
    A loop sub-block that calls `callback_on_step_end` at the end of each denoising step, as a [`DiffusionPipeline`]
    does. Add it to a pipeline with [`insert_step_callback`].
    """

    @property
    def description(self) -> str:
        return "Step within the denoising loop that calls `callback_on_step_end`."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "callback_on_step_end",
                type_hint=Callable,
                description=(
                    "A function called at the end of each denoising step with `(pipeline, step, timestep, "
                    "callback_kwargs)`. `callback_kwargs` holds the values of the denoising loop, such as `latents`, "
                    "and the values the function returns replace them."
                ),
            ),
        ]

    def __call__(
        self, components: ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[ModularPipeline, BlockState]:
        if block_state.callback_on_step_end is None:
            return components, block_state

        callback_kwargs = block_state.as_dict()
        # A callback for a standard pipeline reads the number of steps from `pipeline.num_timesteps`. A modular
        # pipeline has no such attribute, so the callback gets it here.
        callback_kwargs["num_timesteps"] = len(block_state.timesteps)
        callback_outputs = block_state.callback_on_step_end(components, i, t, callback_kwargs) or {}
        # Only the values already in the loop are written back, so a callback cannot add a new one.
        for name, value in callback_outputs.items():
            if hasattr(block_state, name):
                setattr(block_state, name, value)

        return components, block_state


def insert_step_callback(pipeline_or_blocks: ModularPipeline | ModularPipelineBlocks):
    """
    Add a [`StepCallbackBlock`] to the end of every denoising loop of a modular pipeline, in place. The pipeline then
    accepts `callback_on_step_end`, as a [`DiffusionPipeline`] does.

    To stop a generation early, raise an exception from the callback. Setting `pipeline._interrupt` has no effect on a
    modular pipeline.

    Args:
        pipeline_or_blocks ([`ModularPipeline`] or [`ModularPipelineBlocks`]):
            A pipeline, or the blocks to build one from.

    Example:
        ```python
        from diffusers import ModularPipeline
        from diffusers.modular_pipelines import insert_step_callback


        def log_step(pipeline, step, timestep, callback_kwargs):
            print(f"step {step + 1} of {callback_kwargs['num_timesteps']}")
            return callback_kwargs


        pipe = ModularPipeline.from_pretrained("Qwen/Qwen-Image-2.1")
        insert_step_callback(pipe)
        pipe.load_components()

        image = pipe(prompt="a cat", callback_on_step_end=log_step, output="images")[0]
        ```
    """
    if isinstance(pipeline_or_blocks, ModularPipeline):
        blocks = pipeline_or_blocks._blocks
    else:
        blocks = pipeline_or_blocks

    loops = []
    pending = [blocks]
    while pending:
        block = pending.pop()
        pending.extend(block.sub_blocks.values())
        # A denoising loop passes the step index `i` and the timestep `t` to its sub-blocks. A loop over chunks or
        # segments passes `k` and runs its denoising steps inside one sub-block, so it has no place for the callback.
        if isinstance(block, LoopSequentialPipelineBlocks) and all(
            {"i", "t"} <= set(inspect.signature(sub_block.__call__).parameters)
            for sub_block in block.sub_blocks.values()
        ):
            loops.append(block)

    if not loops:
        raise ValueError(
            f"{blocks.__class__.__name__} has no denoising loop that runs its steps as sub-blocks, so a step "
            "callback cannot be added."
        )

    for loop in loops:
        loop.sub_blocks["step_callback"] = StepCallbackBlock()
