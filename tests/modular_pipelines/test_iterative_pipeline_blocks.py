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


# Dummy blocks with trivially checkable arithmetic, in the same nested shape as an autoregressive
# video pipeline: an outer loop (variable `k`) whose steps add 1 to `x` and record the result, with a
# nested inner loop (variable `i`) that multiplies `x` by 10 on each of its iterations.
#
#   OuterLoop                       loop over k in range(num_outer_steps)
#   ├─ add_one     AddOneStep       x += 1
#   ├─ times_ten   TimesTenLoop     the inner loop, over i in range(num_inner_steps)
#   │  ├─ compute_delta             delta = x * 9
#   │  └─ apply_delta               x += delta   (net effect of one inner iteration: x *= 10)
#   └─ record      RecordStep       xs.append(x)


class AddOneStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="x", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="x")]

    @property
    def description(self):
        return "adds 1 to x"

    def __call__(self, components, state, k):
        block_state = self.get_block_state(state)
        block_state.x = block_state.x + 1
        self.set_block_state(state, block_state)
        return components, state


class ComputeDeltaStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="x", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="delta")]

    @property
    def description(self):
        return "computes this inner iteration's increment"

    def __call__(self, components, state, i):
        block_state = self.get_block_state(state)
        block_state.delta = block_state.x * 9
        self.set_block_state(state, block_state)
        return components, state


class ApplyDeltaStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="x", required=True), InputParam(name="delta", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="x")]

    @property
    def description(self):
        return "applies the increment to x"

    def __call__(self, components, state, i):
        block_state = self.get_block_state(state)
        block_state.x = block_state.x + block_state.delta
        self.set_block_state(state, block_state)
        return components, state


class InnerLoopWrapper(IterativePipelineBlocks):
    model_name = "test"

    @property
    def description(self):
        return "inner loop over num_inner_steps"

    @property
    def loop_variables(self):
        return ["i"]

    @property
    def loop_inputs(self):
        return [InputParam(name="num_inner_steps", required=True)]

    @torch.no_grad()
    def __call__(self, components, state, **kwargs):  # ignores the outer loop's `k`
        block_state = self.get_block_state(state)
        for i in range(block_state.num_inner_steps):
            components, state = self.loop_step(components, state, i=i)
        return components, state


class TimesTenLoop(InnerLoopWrapper):
    block_classes = [ComputeDeltaStep, ApplyDeltaStep]
    block_names = ["compute_delta", "apply_delta"]


class RecordStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam(name="x", required=True), InputParam(name="xs", default=None)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="xs")]

    @property
    def description(self):
        return "records x after this outer iteration"

    def __call__(self, components, state, **kwargs):  # ignores the loop's `k`: a catch-all is enough
        block_state = self.get_block_state(state)
        block_state.xs = [*(block_state.xs or []), float(block_state.x)]
        self.set_block_state(state, block_state)
        return components, state


class OuterLoopWrapper(IterativePipelineBlocks):
    model_name = "test"

    @property
    def description(self):
        return "outer loop over num_outer_steps"

    @property
    def loop_variables(self):
        return ["k"]

    @property
    def loop_inputs(self):
        return [InputParam(name="num_outer_steps", required=True)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        for k in range(block_state.num_outer_steps):
            components, state = self.loop_step(components, state, k=k)
        return components, state


class OuterLoop(OuterLoopWrapper):
    block_classes = [AddOneStep, TimesTenLoop, RecordStep]
    block_names = ["add_one", "times_ten", "record"]


class TimestepLoopWrapper(IterativePipelineBlocks):
    """Loop wrapper over `i`, `t` — the signature-validation tests assemble it with different steps."""

    model_name = "test"

    @property
    def description(self):
        return "loop over i, t"

    @property
    def loop_variables(self):
        return ["i", "t"]

    @property
    def loop_inputs(self):
        return [InputParam(name="timesteps", required=True)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.timesteps):
            components, state = self.loop_step(components, state, i=i, t=t)
        return components, state


#   OuterLoop                       loop over k in range(num_outer_steps)
#   ├─ add_one     AddOneStep       x += 1
#   ├─ times_ten   TimesTenLoop     the inner loop, over i in range(num_inner_steps)
#   │  ├─ compute_delta             delta = x * 9
#   │  └─ apply_delta               x += delta   (net effect of one inner iteration: x *= 10)
#   └─ record      RecordStep       xs.append(x)


class TestIterativePipelineBlocksStructure:
    def test_inputs_aggregation(self):
        loop = OuterLoop()
        input_names = [p.name for p in loop.inputs]

        # the outer loop logic's own input, declared in its `loop_inputs`
        assert "num_outer_steps" in [p.name for p in loop.loop_inputs]
        assert "num_outer_steps" in input_names
        # the nested inner loop's `loop_inputs` entry is aggregated too
        assert "num_inner_steps" in [p.name for p in loop.sub_blocks["times_ten"].loop_inputs]
        assert "num_inner_steps" in input_names
        # loop variables are call arguments, not inputs
        assert "k" not in input_names
        assert "i" not in input_names
        # `x` is read by the first step (`add_one`) before any step writes it -> a pipeline input
        assert "x" in input_names
        # `xs` is the accumulator: written by `record` at the end of iteration k, read by it again at
        # k + 1 — the read comes first, so it is an (optional, default None) pipeline input
        assert "xs" in input_names
        # `delta` is written by `compute_delta` before `apply_delta` reads it -> satisfied inside the loop
        assert "delta" not in input_names

    def test_sub_block_outputs_are_aggregated(self):
        loop = OuterLoop()
        output_names = [o.name for o in loop.intermediate_outputs]
        assert "x" in output_names
        assert "xs" in output_names
        assert "delta" in output_names

    def test_loop_block_can_nest_assembled_blocks(self):
        # the nested inner loop stays an assembled IterativePipelineBlocks sub-block
        loop = OuterLoop()
        assert isinstance(loop.sub_blocks["times_ten"], IterativePipelineBlocks)
        assert list(loop.sub_blocks["times_ten"].sub_blocks) == ["compute_delta", "apply_delta"]


class TestIterativePipelineBlocksExecution:
    def _make_pipeline(self):
        return SequentialPipelineBlocks.from_blocks_dict({"loop": OuterLoop()}).init_pipeline()

    def test_nested_loop(self):
        pipe = self._make_pipeline()
        # per outer step: x += 1, then the inner loop doubles the digits (x *= 10 per inner step), then record
        # k=0: (0 + 1) * 10 * 10 = 100 ; k=1: (100 + 1) * 10 * 10 = 10100
        state = pipe(x=torch.tensor(0.0), num_outer_steps=2, num_inner_steps=2)

        assert state.get("xs") == [100.0, 10100.0]
        # the carried value persists as a declared output
        assert float(state.get("x")) == 10100.0

    def test_loop_variables_do_not_leak_into_state(self):
        pipe = self._make_pipeline()
        state = pipe(x=torch.tensor(0.0), num_outer_steps=2, num_inner_steps=1)

        for name in ("k", "i"):
            assert state.get(name) is None
        # declared sub-block outputs persist after the loop (last iteration's value)
        assert state.get("delta") is not None

    def test_block_state_is_loop_scoped(self):
        # the loop's block state holds only the loop logic's own inputs; sub-block values live in the pipeline state
        pipe = self._make_pipeline()
        state = pipe(x=torch.tensor(0.0), num_outer_steps=2, num_inner_steps=1)
        block_state = pipe.blocks.sub_blocks["loop"].get_block_state(state)
        assert block_state.as_dict().keys() == {"num_outer_steps"}

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

    @staticmethod
    def _loop_over(step_cls):
        # assemble the shared `TimestepLoopWrapper` with the given step
        class Loop(TimestepLoopWrapper):
            block_classes = [step_cls]
            block_names = ["step"]

        return Loop

    def test_leaf_signature_is_validated(self):
        # a named parameter that isn't a loop variable fails at construction, even with a catch-all
        class UnknownVarStep(ModularLoopPipelineBlocks):
            model_name = "test"

            @property
            def description(self):
                return "loop step naming a variable the loop doesn't have"

            def __call__(self, components, state, k, **kwargs):
                return components, state

        with pytest.raises(ValueError, match="not loop variables"):
            self._loop_over(UnknownVarStep)()

        # naming only some loop variables without a `**kwargs` catch-all fails at construction
        class MissingVarStep(ModularLoopPipelineBlocks):
            model_name = "test"

            @property
            def description(self):
                return "loop step naming only one of the loop variables, without a catch-all"

            def __call__(self, components, state, t):
                return components, state

        with pytest.raises(ValueError, match="must accept the loop variables"):
            self._loop_over(MissingVarStep)()

    def test_leaf_signature_subset_with_kwargs_is_valid(self):
        # naming only the used loop variables plus `**kwargs` is accepted
        class SubsetStep(ModularLoopPipelineBlocks):
            model_name = "test"

            @property
            def description(self):
                return "loop step using only `t`, ignoring the rest via a catch-all"

            def __call__(self, components, state, t, **kwargs):
                return components, state

        self._loop_over(SubsetStep)()  # does not raise

    def test_loop_leaf_standalone_raises(self):
        # outside a loop, a leaf block with loop variables in its signature cannot run
        pipe = SequentialPipelineBlocks.from_blocks_dict({"compute_delta": ComputeDeltaStep()}).init_pipeline()
        with pytest.raises(TypeError):
            pipe(x=torch.tensor(1.0))
