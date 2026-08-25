<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IterativePipelineBlocks

[`~modular_pipelines.IterativePipelineBlocks`] is a multi-block type that runs its sub-blocks multiple times. It is what we use to build a denoising loop: the sub-blocks predict the noise and step the scheduler, the loop runs them once per timestep. You can also nest one [`~modular_pipelines.IterativePipelineBlocks`] under another to build an autoregressive video pipeline that generates chunk after chunk. Every iteration can be [streamed](./modular_pipeline#streaming) to the caller as it completes.

This guide shows you how to write the loop steps, the loop itself, how to nest loops, and how values travel from one iteration to the next.

> [!TIP]
> [`~modular_pipelines.IterativePipelineBlocks`] replaces [`~modular_pipelines.LoopSequentialPipelineBlocks`]; see [the last section](#loopsequentialpipelineblocks) for the differences.

## Loop steps

A loop step is a [`~modular_pipelines.ModularLoopPipelineBlocks`]. It is a regular [`~modular_pipelines.ModularPipelineBlocks`] — it declares `inputs` and `intermediate_outputs`, and reads and writes the [`~modular_pipelines.PipelineState`] through `get_block_state` / `set_block_state` — with one difference: its `__call__` also receives the loop's *loop variables* as arguments. For example, a denoising loop can pass the step index `i` and the timestep `t`.

Loop variables are the loop's own bookkeeping, not pipeline data. They are local to the loop: the loop hands them to each step as plain call arguments for that one iteration, and they are never written to the [`~modular_pipelines.PipelineState`]. Anything that has to outlive the iteration goes through the state instead, like `noise_pred` and `latents` below. (A streaming consumer does see them: each [`~modular_pipelines.StreamEvent`] carries that iteration's values in `event.loop_kwargs`.)

```py
from diffusers.modular_pipelines import ModularLoopPipelineBlocks, InputParam, OutputParam

class DenoiserStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def description(self):
        return "predicts the noise for one timestep"

    @property
    def inputs(self):
        return [InputParam(name="latents", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="noise_pred")]

    def __call__(self, components, state, i, t):
        block_state = self.get_block_state(state)
        block_state.noise_pred = block_state.latents * 0 + t   # stands in for the denoiser
        self.set_block_state(state, block_state)
        return components, state


class SchedulerStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def description(self):
        return "updates the latents with the noise prediction"

    @property
    def inputs(self):
        return [InputParam(name="latents", required=True), InputParam(name="noise_pred", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="latents")]

    def __call__(self, components, state, i, t):
        block_state = self.get_block_state(state)
        block_state.latents = block_state.latents + block_state.noise_pred
        self.set_block_state(state, block_state)
        return components, state
```

Because each step works on the [`~modular_pipelines.PipelineState`], the values it writes (`noise_pred`, the updated `latents`) are visible to the next step in the same iteration and to the next iteration — `latents` is read at the start of every iteration and written at the end of it.

## Loop wrapper

The loop itself is a subclass of [`~modular_pipelines.IterativePipelineBlocks`]. It declares:

- `loop_variables`, the names of the variables it passes to its steps on every iteration. Every step's `__call__` must accept exactly these after `(components, state)`; this is validated when the loop is constructed.
- `loop_inputs`, the inputs the loop logic itself reads — here the `timesteps` it iterates. They join the inputs aggregated from the steps (see below), and they are what `get_block_state` returns for the loop block.
- `__call__`, the loop logic: read the loop's block state, and call `loop_step` once per iteration with the loop variables. `loop_step` runs every step once.

```py
import torch
from diffusers.modular_pipelines import IterativePipelineBlocks

class DenoiseLoop(IterativePipelineBlocks):
    model_name = "test"
    block_classes = [DenoiserStep, SchedulerStep]
    block_names = ["denoiser", "scheduler"]

    @property
    def description(self):
        return "denoises the latents over the timesteps"

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
```

An [`~modular_pipelines.IterativePipelineBlocks`] is a [`~modular_pipelines.SequentialPipelineBlocks`], so it [aggregates](./sequential_pipeline_blocks#aggregated-inputs-and-outputs) its steps' `inputs` and `intermediate_outputs` the same way any assembled block does, and adds `loop_inputs` / `loop_intermediate_outputs` on top. `DenoiseLoop.inputs` is therefore `timesteps` (its own) and `latents` (what the steps need from outside the loop) — but not `noise_pred`, which `DenoiserStep` produces before `SchedulerStep` reads it, so it is satisfied inside the loop. The assembled loop ends up with the same kind of input/output contract as a single block, which is what lets it be dropped into a [`~modular_pipelines.SequentialPipelineBlocks`], or into another loop. Outputs the steps write persist in the state after the loop. Run it like any other block:

```py
pipeline = DenoiseLoop().init_pipeline()
state = pipeline(latents=torch.tensor(0.0), timesteps=torch.tensor([1.0, 2.0, 3.0]))
state.get("latents")   # tensor(6.)  — 0 + 1 + 2 + 3
```

Steps can also be attached after the fact with [`~modular_pipelines.IterativePipelineBlocks.from_blocks_dict`], which keeps the loop logic separate from what runs inside it:

```py
loop = DenoiseLoop.from_blocks_dict({"denoiser": DenoiserStep(), "scheduler": SchedulerStep()})
```

You can also change what runs inside a loop you already have: `sub_blocks` is an ordered dict, so any loop step that takes the same loop variables can be inserted into it.

```py
loop.sub_blocks.insert("log", LogStep(), 2)   # after the denoiser and the scheduler
```

A step inserted this way isn't signature-checked the way the ones a loop is constructed with are — a mismatch raises a `TypeError` on the first iteration.

If the loop logic produces a value of its own — an autoregressive loop collecting the decoded frames of every chunk, for example — declare it in `loop_intermediate_outputs` and write it back with `set_block_state`, exactly as a leaf block would:

```py
@property
def loop_intermediate_outputs(self):
    return [OutputParam(name="history")]

@torch.no_grad()
def __call__(self, components, state):
    block_state = self.get_block_state(state)
    block_state.history = []
    for i, t in enumerate(block_state.timesteps):
        components, state = self.loop_step(components, state, i=i, t=t)
        block_state.history.append(state.get("latents"))
    self.set_block_state(state, block_state)
    return components, state
```

The loop's block state holds only its `loop_inputs` and `loop_intermediate_outputs`. The values the steps produce live in the [`~modular_pipelines.PipelineState`] — read them with `state.get(...)`, as above — so the loop never works from a stale copy.

## Nesting loops

An [`~modular_pipelines.IterativePipelineBlocks`] can be a step of another one. An autoregressive video pipeline generates a chunk of frames at a time, so it is an outer loop over chunks: each of its iterations prepares the chunk's latents from the frames generated so far, runs a full denoising loop over them, and appends the result to the history.

The inner denoising loop is a step of the outer loop, so its `__call__` must accept the outer loop's variables, and it declares `loop_variables` of its own for its own steps. The two sets are independent: `k` arrives as a call argument and the inner loop is free to use it — the wan-animate-2 denoise loop puts the chunk index in its progress bar — but it is not forwarded to the steps, which are passed the inner loop's `i` and `t`.

```py
class ChunkDenoiseLoop(IterativePipelineBlocks):
    model_name = "test"
    block_classes = [DenoiserStep, SchedulerStep]
    block_names = ["denoiser", "scheduler"]

    @property
    def description(self):
        return "denoises one chunk over the timesteps"

    @property
    def loop_variables(self):
        return ["i", "t"]

    @property
    def loop_inputs(self):
        return [InputParam(name="timesteps", required=True)]

    @torch.no_grad()
    def __call__(self, components, state, k):   # `k` comes from the chunk loop
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.timesteps):
            components, state = self.loop_step(components, state, i=i, t=t)
        return components, state


class PrepareChunkStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def description(self):
        return "prepares this chunk's latents from the history"

    @property
    def inputs(self):
        return [InputParam(name="history", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="latents")]

    def __call__(self, components, state, k):
        block_state = self.get_block_state(state)
        block_state.latents = block_state.history + k
        self.set_block_state(state, block_state)
        return components, state


class UpdateHistoryStep(ModularLoopPipelineBlocks):
    model_name = "test"

    @property
    def description(self):
        return "records the denoised chunk"

    @property
    def inputs(self):
        return [InputParam(name="latents", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="history")]

    def __call__(self, components, state, k):
        block_state = self.get_block_state(state)
        block_state.history = block_state.latents
        self.set_block_state(state, block_state)
        return components, state


class ChunkLoop(IterativePipelineBlocks):
    model_name = "test"
    block_classes = [PrepareChunkStep, ChunkDenoiseLoop, UpdateHistoryStep]
    block_names = ["prepare", "denoise", "update"]

    @property
    def description(self):
        return "generates the video chunk by chunk"

    @property
    def loop_variables(self):
        return ["k"]

    @property
    def loop_inputs(self):
        return [InputParam(name="num_chunks", required=True)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        for k in range(block_state.num_chunks):
            components, state = self.loop_step(components, state, k=k)
        return components, state
```

`ChunkDenoiseLoop` is `DenoiseLoop` with `k` added to `__call__`, spelled out in full here so the whole loop is visible in one place. When two loops share their logic and differ only in what runs inside them, write the logic once in a wrapper class and subclass it to attach `block_classes` / `block_names` — that is how `WanAnimate2DenoiseLoopWrapper` serves both the regular and the distilled denoise step.

```py
pipeline = ChunkLoop().init_pipeline()
state = pipeline(num_chunks=2, timesteps=torch.tensor([1.0, 2.0]), history=torch.tensor(0.0))
state.get("history")   # tensor(7.) — chunk 0: 0 + 0 + 3 = 3, chunk 1: 3 + 1 + 3 = 7
```

`history` is carried from one iteration to the next: `UpdateHistoryStep` writes it at the end of one, `PrepareChunkStep` reads it at the start of the next. Because the reader comes before the writer, it is one of the loop's inputs — which is why `pipeline(history=...)` works above — and like any input it has to come from either the user or an earlier block. If seeding it isn't meaningful (a decoder cache, the previous chunk's frames), have the block that runs before the loop declare it as an output and set its initial value; it then drops out of the pipeline's signature.

## Streaming

To let [`~ModularPipeline.stream`] hand back the live state after every iteration, also implement `stream` — the same loop, written as a generator over `stream_step`, which runs one iteration like `loop_step` and additionally yields a [`~modular_pipelines.StreamEvent`] for it (after the events of any nested loop):

```py
class DenoiseLoop(IterativePipelineBlocks):
    ...

    def stream(self, components, state):
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.timesteps):
            components, state = yield from self.stream_step(components, state, i=i, t=t)
        return components, state
```

A nested loop's `stream` takes the outer loop's variables exactly like its `__call__` does. Streaming is opt-in per loop: `pipeline.blocks.supports_streaming` tells you whether every loop on the path implements it. See [Streaming](./modular_pipeline#streaming) for the consumer side, including how to run a single iteration at a time with `loop_step` when a serving engine or a real-time input source needs to own the loop.

## LoopSequentialPipelineBlocks

[`~modular_pipelines.LoopSequentialPipelineBlocks`] is the earlier loop type and is still used by existing pipelines. It differs in three ways: its steps share one flattened [`~modular_pipelines.BlockState`] that the wrapper extracts before the loop (instead of each step reading the [`~modular_pipelines.PipelineState`] itself), it cannot contain another loop, and it cannot stream. Use [`~modular_pipelines.IterativePipelineBlocks`] for new pipelines.
