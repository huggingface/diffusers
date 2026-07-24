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


import pytest
import torch

from diffusers.modular_pipelines import (
    InputParam,
    IterativePipelineBlocks,
    ModularLoopPipelineBlocks,
    ModularPipelineBlocks,
    OutputParam,
    SequentialPipelineBlocks,
)


# Dummy blocks modeled on the Helios chunk-loop use case: an outer autoregressive chunk loop
# (history carried across chunks) containing a full inner timestep denoising loop. Loop variables
# (`k` for the chunk loop, `i`/`t` for the timestep loop) are passed to leaf sub-blocks as call
# arguments; every leaf sub-block of a loop must accept its loop's variables.


class ChunkNoiseGenStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="history", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="chunk_latents")]

    @property
    def description(self):
        return "prepares this chunk's latents from the history"

    def __call__(self, components, state, k):
        block_state = self.get_block_state(state)
        block_state.chunk_latents = block_state.history + k
        self.set_block_state(state, block_state)
        return components, state


class LoopDenoiserStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="chunk_latents", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="noise_pred")]

    @property
    def description(self):
        return "predicts the noise for one timestep"

    def __call__(self, components, state, i, t):
        block_state = self.get_block_state(state)
        block_state.noise_pred = block_state.chunk_latents * 0 + t
        self.set_block_state(state, block_state)
        return components, state


class LoopSchedulerStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="chunk_latents", required=True), InputParam(name="noise_pred", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="chunk_latents")]

    @property
    def description(self):
        return "updates the chunk latents with the noise prediction"

    def __call__(self, components, state, i, t):
        block_state = self.get_block_state(state)
        block_state.chunk_latents = block_state.chunk_latents + block_state.noise_pred
        self.set_block_state(state, block_state)
        return components, state


class InnerDenoiseLoop(IterativePipelineBlocks):
    """Inner timestep loop — itself an assembled loop block, nested inside the chunk loop.

    Like every sub-block of the chunk loop, it accepts the outer loop variable `k` (and ignores it);
    its own sub-blocks accept its own loop variables `i` / `t` instead.
    """

    model_name = "test"
    block_classes = [LoopDenoiserStep, LoopSchedulerStep]
    block_names = ["denoiser", "scheduler"]

    @property
    def description(self):
        return "inner timestep loop"

    @property
    def loop_variables(self):
        return ["i", "t"]

    @property
    def inputs(self):
        return [InputParam(name="timesteps", required=True), *super().inputs]

    @torch.no_grad()
    def __call__(self, components, state, k):
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.timesteps):
            components, state = self.loop_step(components, state, i=i, t=t)
        return components, state


class ChunkUpdateStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="chunk_latents", required=True), InputParam(name="latent_chunks", default=None)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="history"), OutputParam(name="latent_chunks")]

    @property
    def description(self):
        return "records the denoised chunk and updates the history"

    def __call__(self, components, state, k):
        block_state = self.get_block_state(state)
        block_state.history = block_state.chunk_latents
        block_state.latent_chunks = [*(block_state.latent_chunks or []), float(block_state.chunk_latents)]
        self.set_block_state(state, block_state)
        return components, state


class ChunkLoop(IterativePipelineBlocks):
    """Outer chunk loop containing the inner timestep loop as a sub-block."""

    model_name = "test"
    block_classes = [ChunkNoiseGenStep, InnerDenoiseLoop, ChunkUpdateStep]
    block_names = ["noise_gen", "denoise", "update"]

    @property
    def description(self):
        return "outer autoregressive chunk loop"

    @property
    def loop_variables(self):
        return ["k"]

    @property
    def inputs(self):
        return [InputParam(name="num_latent_chunk", required=True), *super().inputs]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        for k in range(block_state.num_latent_chunk):
            components, state = self.loop_step(components, state, k=k)
        return components, state


class TestIterativePipelineBlocksStructure:
    def test_inputs_aggregation(self):
        loop = ChunkLoop()
        input_names = [p.name for p in loop.inputs]

        # inputs of the loop logic itself and of the nested loop are surfaced
        assert "num_latent_chunk" in input_names
        assert "timesteps" in input_names
        # loop variables are call arguments, not inputs
        assert "k" not in input_names
        assert "i" not in input_names
        assert "t" not in input_names
        # cross-chunk carries surface as (optional) iteration-0 seeds
        assert "history" in input_names
        assert "latent_chunks" in input_names

    def test_sub_block_outputs_are_aggregated(self):
        loop = ChunkLoop()
        output_names = [o.name for o in loop.intermediate_outputs]
        assert "history" in output_names
        assert "latent_chunks" in output_names

    def test_loop_block_can_nest_assembled_blocks(self):
        # the nested inner loop stays an assembled IterativePipelineBlocks sub-block
        loop = ChunkLoop()
        assert isinstance(loop.sub_blocks["denoise"], IterativePipelineBlocks)
        assert list(loop.sub_blocks["denoise"].sub_blocks) == ["denoiser", "scheduler"]


class TestIterativePipelineBlocksExecution:
    def _make_pipeline(self):
        return SequentialPipelineBlocks.from_blocks_dict({"chunks": ChunkLoop()}).init_pipeline()

    def test_nested_chunk_loop(self):
        pipe = self._make_pipeline()
        # per chunk: chunk_latents = history + k, then += t for every timestep (1.0 + 2.0),
        # then history <- chunk_latents
        # chunk 0: 0 + 0 + 3 = 3 ; chunk 1: 3 + 1 + 3 = 7 ; chunk 2: 7 + 2 + 3 = 12
        state = pipe(num_latent_chunk=3, timesteps=torch.tensor([1.0, 2.0]), history=torch.tensor(0.0))

        assert state.get("latent_chunks") == [3.0, 7.0, 12.0]
        # the cross-chunk carry persists as a declared output
        assert float(state.get("history")) == 12.0

    def test_loop_variables_do_not_leak_into_state(self):
        pipe = self._make_pipeline()
        state = pipe(num_latent_chunk=2, timesteps=torch.tensor([1.0]), history=torch.tensor(0.0))

        for name in ("k", "i", "t"):
            assert state.get(name) is None
        # declared sub-block outputs persist after the loop (last iteration's value)
        assert state.get("noise_pred") is not None

    def test_sub_block_type_is_validated(self):
        # a regular ModularPipelineBlocks cannot be a loop sub-block: fails at construction
        class PlainStep(ModularPipelineBlocks):
            model_name = "test"

            @property
            def description(self):
                return "regular block, not a loop step"

            def __call__(self, components, state):
                return components, state

        class BadTypeLoop(IterativePipelineBlocks):
            model_name = "test"
            block_classes = [PlainStep]
            block_names = ["plain"]

            @property
            def description(self):
                return "loop with a non-loop sub-block"

        with pytest.raises(ValueError, match="must be a `ModularLoopPipelineBlocks`"):
            BadTypeLoop()

    def test_leaf_signature_is_validated(self):
        # a loop step whose signature doesn't match the loop's variables fails at construction
        class WrongSigStep(ModularLoopPipelineBlocks):
            model_name = "test"

            @property
            def description(self):
                return "loop step with the wrong loop variables"

            def __call__(self, components, state, k):
                return components, state

        class BadSigLoop(IterativePipelineBlocks):
            model_name = "test"
            block_classes = [WrongSigStep]
            block_names = ["wrong"]

            @property
            def description(self):
                return "loop whose sub-block names the wrong loop variables"

            @property
            def loop_variables(self):
                return ["i", "t"]

            @property
            def inputs(self):
                return [InputParam(name="timesteps", required=True), *super().inputs]

            @torch.no_grad()
            def __call__(self, components, state):
                block_state = self.get_block_state(state)
                for i, t in enumerate(block_state.timesteps):
                    components, state = self.loop_step(components, state, i=i, t=t)
                return components, state

        with pytest.raises(ValueError, match="must accept the loop variables"):
            BadSigLoop()

    def test_loop_leaf_standalone_raises(self):
        # outside a loop, a leaf block with loop variables in its signature cannot run
        pipe = SequentialPipelineBlocks.from_blocks_dict({"denoiser": LoopDenoiserStep()}).init_pipeline()
        with pytest.raises(TypeError):
            pipe(chunk_latents=torch.tensor(1.0))
