<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IterativePipelineBlocks

[`~modular_pipelines.IterativePipelineBlocks`] is a multi-block type that runs its sub-blocks multiple times. It is used to build a denoising loop where the sub-blocks predict the noise and step the scheduler, then the loop runs them once per timestep. You can also nest one [`~modular_pipelines.IterativePipelineBlocks`] under another to build an autoregressive video pipeline that generates chunk after chunk. A loop can stream each iteration to the caller when its `stream` method is implemented.

Use this block type when a pipeline must repeat a sequence of blocks while carrying state between iterations and, when implemented, streaming progress to the caller.

Two classes are involved in building a loop.

| Class | Role | State it works with |
|---|---|---|
| [`~modular_pipelines.ModularLoopPipelineBlocks`] | A *loop step*: a regular block that runs inside a loop, once per iteration | The full [`~modular_pipelines.PipelineState`], through its own `get_block_state` / `set_block_state`; also receives the loop variables as call arguments |
| [`~modular_pipelines.IterativePipelineBlocks`] | The *loop* itself: holds the loop logic and runs its steps once per iteration; loops can nest and stream | Its block state holds only its own `loop_inputs`; all data flows between its steps through the [`~modular_pipelines.PipelineState`] |

This guide uses a few closely related terms, so let's define them upfront.

- **Pipeline state** is the shared [`~modular_pipelines.PipelineState`] every block reads and writes. It is the only place where data crosses blocks and survives an iteration.
- **Block state** is one block's declared view of the pipeline state: what its `get_block_state` returns and its `set_block_state` writes back. For a loop wrapper, it holds only the loop's `loop_inputs`.
- **Loop inputs** are the loop wrapper's own input declaration: what the loop logic itself reads to drive the iteration (the `timesteps` it iterates over, for example).
- **Loop variables** are the per-iteration values the loop passes to its steps as call arguments (`i`, `t`). They are never written to the pipeline state.

> [!TIP]
> [`~modular_pipelines.IterativePipelineBlocks`] replaces [`~modular_pipelines.LoopSequentialPipelineBlocks`]; see the [LoopSequentialPipelineBlocks](#loopsequentialpipelineblocks) guide for the differences.

## Loop steps

A loop step is a [`~modular_pipelines.ModularLoopPipelineBlocks`]. It is a regular [`~modular_pipelines.ModularPipelineBlocks`] — it declares `inputs` and `intermediate_outputs`, and reads and writes the [`~modular_pipelines.PipelineState`] through `get_block_state` / `set_block_state` — with one difference. Its `__call__` also receives the loop's *loop variables* as arguments. For example, a denoising loop can pass the step index `i` and the timestep `t`.

Loop variables are local to the loop and act as its bookkeeping, not pipeline data. The loop hands them to each step as plain call arguments for that one iteration, and they are never written to the [`~modular_pipelines.PipelineState`]. Anything that has to outlive the iteration goes through the state instead, like `noise_pred` and `latents` below. A streaming consumer does see them, each [`~modular_pipelines.StreamEvent`] carries that iteration's values in `event.loop_kwargs`.

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

Because each step works on the [`~modular_pipelines.PipelineState`], the values it writes (`noise_pred`, the updated `latents`) are visible to the next step in the same iteration and to the next iteration. `latents` is read at the start of every iteration and written at the end of it.

## Loop wrapper

The loop itself is a subclass of [`~modular_pipelines.IterativePipelineBlocks`]. It declares:

- `loop_variables`, the names of the variables passed to each step on every iteration. Steps receive loop variables as keyword arguments. A step can name only the variables it uses if its `__call__` accepts `**kwargs`. Otherwise, it must name every loop variable. The loop validates these signatures when it is constructed and raises for unknown or missing variables.
- `loop_inputs`, the inputs the loop logic itself reads — here the `timesteps` it iterates. They join the inputs aggregated from the steps (see below), and they are what `get_block_state` returns for the loop block.
- `__call__`, the loop logic: read the loop's block state, and call `loop_step` once per iteration with the loop variables. `loop_step` runs every step once.

The wrapper defines only the loop logic. Which steps run inside it should be attached separately, in a subclass:

```py
import torch
from diffusers.modular_pipelines import IterativePipelineBlocks

class DenoiseLoopWrapper(IterativePipelineBlocks):
    model_name = "test"

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


class DenoiseLoop(DenoiseLoopWrapper):
    block_classes = [DenoiserStep, SchedulerStep]
    block_names = ["denoiser", "scheduler"]
```

This separation means the same loop logic can work with different combinations of loop steps. For example, subclass the wrapper again with different `block_classes`. Steps can also be attached to the loop with [`~modular_pipelines.IterativePipelineBlocks.from_blocks_dict`]:

```py
loop = DenoiseLoopWrapper.from_blocks_dict({"denoiser": DenoiserStep(), "scheduler": SchedulerStep()})
```

You can also change what runs inside a loop you already have: add a step, reorder, swap one out. For example, you can insert a logging step like this:

```py
loop = DenoiseLoopWrapper.from_blocks_dict(loop.sub_blocks.copy().insert("log", LogStep(), 1))
```

An [`~modular_pipelines.IterativePipelineBlocks`] is a [`~modular_pipelines.SequentialPipelineBlocks`], so it [aggregates](./sequential_pipeline_blocks#aggregated-inputs-and-outputs) its steps' `inputs` and `intermediate_outputs` the same way any assembled block does, and adds `loop_inputs` / `loop_intermediate_outputs` on top. `DenoiseLoop.inputs` is therefore `timesteps` (its own) and `latents` (what the steps need from outside the loop) — but not `noise_pred`, which `DenoiserStep` produces before `SchedulerStep` reads it, so it is satisfied inside the loop. The assembled loop ends up with the same kind of input/output contract as a single block, which is what lets it be dropped into a [`~modular_pipelines.SequentialPipelineBlocks`], or into another loop. Outputs the steps write persist in the state after the loop. Run it like any other block:

```py
pipeline = DenoiseLoop().init_pipeline()
state = pipeline(latents=torch.tensor(0.0), timesteps=torch.tensor([1.0, 2.0, 3.0]))
state.get("latents")   # tensor(6.)  — 0 + 1 + 2 + 3
```

The wrapper contains only the loop logic, i.e. how to iterate through its steps, so its `loop_inputs` should be just what that takes (the `timesteps` above). All data flows through the steps, which read and write the pipeline state directly. In the example above, `DenoiserStep` writes `noise_pred` to the state and `SchedulerStep` reads it back and writes the updated `latents`. The wrapper touches none of them. If the loop logic seems to need to do more than iterate, for example, collect results, you should add a loop step for it instead. In `wan_animate_2`, a small collect step appends each segment's decoded frames to `segment_frames`, and the next segment's prep step reads it back to condition on. Under [streaming](#streaming), the partial collection is visible after every iteration. What we don't want is the wrapper doing it inline:

```py
# don't: loop logic collecting results itself
def __call__(self, components, state):
    block_state = self.get_block_state(state)
    segment_frames = []
    for k in range(block_state.num_segments):
        components, state = self.loop_step(components, state, k=k)
        segment_frames.append(state.get("out_frames"))  # reaching into the state: make this a collect loop step
    ...
```

## Nesting loops

An [`~modular_pipelines.IterativePipelineBlocks`] can be a step of another one. An autoregressive video pipeline generates a chunk of frames at a time, so it is an outer loop over chunks. Each of its iterations prepares the chunk's latents from the frames generated so far, runs a full denoising loop over them, and appends the result to the history.

The inner denoising loop is a step of the outer loop, so its `__call__` must accept the outer loop's variables (or take `**kwargs` if it ignores them), and it declares `loop_variables` of its own for its own steps. The two sets are independent. `k` arrives as a call argument and the inner loop is free to use it — the wan-animate-2 denoise loop puts the chunk index in its progress bar — but it is not forwarded to the steps, which are passed the inner loop's `i` and `t`.

This is the structure this section builds:

```
VideoLoop                 outer loop — loop_variables ["k"], iterates over chunks
├─ prepare                PrepareChunkStep    (leaf step, takes k)
├─ denoise                ChunkDenoiseLoop    (the INNER loop — a step of the outer one)
│  ├─ denoiser            DenoiserStep        (takes the inner loop's i, t)
│  └─ scheduler           SchedulerStep
└─ update                 UpdateHistoryStep   (leaf step, takes k)
```

```py
class ChunkDenoiseLoopWrapper(IterativePipelineBlocks):
    model_name = "test"

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
    def __call__(self, components, state, k):   # `k` comes from the outer video loop
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.timesteps):
            components, state = self.loop_step(components, state, i=i, t=t)
        return components, state


class ChunkDenoiseLoop(ChunkDenoiseLoopWrapper):
    block_classes = [DenoiserStep, SchedulerStep]
    block_names = ["denoiser", "scheduler"]
```

`ChunkDenoiseLoop` is the *inner* loop that denoises a single chunk. It becomes a step of the outer loop below, which iterates over the chunks. At each `k`, it prepares the chunk's latents from the history, runs the full inner denoising loop over them, and records the result:

```py
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


class VideoLoopWrapper(IterativePipelineBlocks):
    model_name = "test"

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


class VideoLoop(VideoLoopWrapper):
    block_classes = [PrepareChunkStep, ChunkDenoiseLoop, UpdateHistoryStep]
    block_names = ["prepare", "denoise", "update"]
```

Run the outer loop like any other block:

```py
pipeline = VideoLoop().init_pipeline()
state = pipeline(num_chunks=2, timesteps=torch.tensor([1.0, 2.0]), history=torch.tensor(0.0))
state.get("history")   # tensor(7.) — chunk 0: 0 + 0 + 3 = 3, chunk 1: 3 + 1 + 3 = 7
```

`history` is carried from one iteration to the next. `UpdateHistoryStep` writes it at the end of one, and `PrepareChunkStep` reads it at the start of the next. Because the reader comes before the writer, it is one of the loop's inputs — which is why `pipeline(history=...)` works above — and like any input it has to come from either the user or an earlier block. If seeding it isn't meaningful (a decoder cache, the previous chunk's frames), have the block that runs before the loop declare it as an output and set its initial value. It then drops out of the pipeline's signature.

## Streaming

To let [`~ModularPipeline.stream`] hand back the live state after every iteration, also implement `stream`. This is the same loop written as a generator over `stream_step`, which runs one iteration like `loop_step` and additionally yields a [`~modular_pipelines.StreamEvent`] for it (after the events of any nested loop):

```py
class DenoiseLoop(IterativePipelineBlocks):
    ...

    def stream(self, components, state):
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.timesteps):
            components, state = yield from self.stream_step(components, state, i=i, t=t)
        return components, state
```

A nested loop's `stream` takes the outer loop's variables exactly like its `__call__` does. Streaming is opt-in per loop.`pipeline.blocks.supports_streaming` tells you whether every loop on the path implements it. See [Streaming](./modular_pipeline#streaming) for the consumer side.

## LoopSequentialPipelineBlocks

[`~modular_pipelines.LoopSequentialPipelineBlocks`] is the earlier loop type and is still used by existing pipelines. It differs in three ways:

1. Steps share one flattened [`~modular_pipelines.BlockState`] that the wrapper extracts before the loop (instead of each step reading the [`~modular_pipelines.PipelineState`] itself).
2. It cannot contain another loop.
3. It cannot stream.

Use [`~modular_pipelines.IterativePipelineBlocks`] for new pipelines.
