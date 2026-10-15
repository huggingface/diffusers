<!-- Copyright 2026 The VAST-AI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EdenLiveFlowMatchEulerDiscreteScheduler

This scheduler implements the deterministic flow Euler trajectory used by EdenLive 14B bidirectional inference and
the stochastic trajectory used by the distilled causal checkpoints. The pipeline selects the appropriate
configuration when loading a profile. The 5B bidirectional profile uses [`UniPCMultistepScheduler`] instead.

## Released schedules

| Profile | Scheduler mode | Default evaluations | Model timesteps |
|---|---|---|---|
| `14b-bidir` | Deterministic Euler, `shift=3.0` | 30 per video | Generated from the requested step count |
| `5b-causal` | `stochastic_sampling=True` | 4 per chunk | `[1000, 937.5, 833.3333333333, 625]` |
| `14b-causal` | `stochastic_sampling=True` | 4 per chunk | `[1000, 900, 750, 500]` |

The causal schedules are stored as `default_timesteps` and already contain the released time shift. Passing a
different `num_inference_steps` without replacing the schedule raises an error. Arbitrary step counts are not a
supported inference setting for these distilled checkpoints.

## Set a schedule

Call `set_timesteps()` before stepping through a video or a new causal chunk. It resets both the step index and the
begin index. Choose one schedule source:

- `timesteps`: explicit model times in [0, `num_train_timesteps`], converted to noise levels by dividing by
  `num_train_timesteps` (1000 by default).
- `sigmas`: explicit noise levels in [0, 1]. Do not supply both `timesteps` and `sigmas`.
- Neither argument: use `default_timesteps` if configured; otherwise generate a schedule from a positive
  `num_inference_steps` and apply `shift`.

Explicit values must be finite and strictly decreasing. They are not shifted again. An optional terminal zero is
removed from the model evaluation points, and the internal `sigmas` tensor always receives one terminal zero.
If `num_inference_steps` is provided with an explicit or default schedule, it must match the number of positive
evaluation points. `timesteps` and `sigmas` are stored in FP32 on the requested `device`.

For example, the 14B causal schedule can be constructed independently for a custom inference loop:

```py
from diffusers import EdenLiveFlowMatchEulerDiscreteScheduler

scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(
    stochastic_sampling=True,
    default_timesteps=[1000, 900, 750, 500],
)
scheduler.set_timesteps(device="cuda")
```

For a loaded pipeline, use its existing `pipe.scheduler` to preserve the profile configuration.

## Advance the sample

Pass the model's flow velocity, the current value from `scheduler.timesteps`, and the noisy latent sample to
`step(model_output, timestep, sample, ...)`. Deterministic mode takes an Euler step. Stochastic mode first predicts
a clean endpoint and then adds fresh noise at the next noise level. The terminal step returns the clean endpoint
without drawing noise.

Use `generator` to control stochastic draws, or supply `variance_noise` for a nonterminal stochastic step. The two
arguments are mutually exclusive. Supplied noise must have the same shape and dtype as the sample and be on its
device. This allows a distributed caller to generate noise once and share it across ranks. Noise is unused in
deterministic mode and at the terminal stochastic step.

`step()` returns `FlowMatchEulerDiscreteSchedulerOutput.prev_sample`, or `(prev_sample,)` when
`return_dict=False`. The result preserves the input sample's dtype. `scale_model_input()` returns its input
unchanged. Once the schedule is exhausted, call `set_timesteps()` before processing the next chunk.

## API

[[autodoc]] EdenLiveFlowMatchEulerDiscreteScheduler
    - set_timesteps
    - step
