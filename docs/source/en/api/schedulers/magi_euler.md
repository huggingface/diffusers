# MagiEulerScheduler

MAGI-1 predicts flow velocity directly. Its sampler advances from noise at time 0 toward clean data at time 1:

```python
next_sample = sample + velocity * (next_timestep - timestep)
```

Do not convert the Transformer output from a clean-sample prediction to velocity. Apply guidance to velocity before
calling `step`. Keep the sampling state in FP32; the Transformer manages its internal mixed-precision computation.

## Time schedule

The default schedule squares a uniform grid and then applies the official inverse shift of 3. The scheduler also
supports the reference's square, piecewise and linear schedules, and both 12-step shortcut grid orderings.
`num_inference_steps` is the number of updates **per chunk**, not the total number of sliding-window model calls.

`timesteps` contains normalized model evaluation times without a factor of 1000. `timestep_schedule` includes the
last integration endpoint. Build the grid on the execution device to match the reference's arithmetic. The FP32
default endpoint can be slightly above 1; it is intentionally not clamped or reconstructed from `1 - sigma`.

## Sequential and chunk-wise updates

For a single chunk, use sequential steps:

```python
from diffusers import MagiEulerScheduler

scheduler = MagiEulerScheduler()
scheduler.set_timesteps(64, device=latents.device)
for timestep in scheduler.timesteps:
    velocity = transformer(
        latents, prompt_embeds, timestep.expand(latents.shape[0])
    ).sample
    latents = scheduler.step(velocity, timestep, latents).prev_sample
```

This illustrates the scheduler interface, not the complete MAGI generation loop: text preparation, guidance and
chunk-window management must be supplied separately.

For an active window of equally sized video chunks, pass both endpoint tensors:

```python
latents = scheduler.step(
    guided_velocity,
    timestep=current_chunk_times,
    sample=latents,
    next_timestep=next_chunk_times,
).prev_sample
```

Endpoints can be scalars, `(chunks,)` shared across the batch, or `(batch, chunks)`. A one-dimensional tensor indexes
chunks, not batch items. Explicit endpoint updates leave `step_index` unchanged so chunks can follow different
parts of the same schedule. Sequential calls advance it; `set_timesteps` resets it. Outputs always remain FP32.

The scheduler does not select active chunks, construct attention ranges, manage prefix caches or apply three-way
guidance. The 12-step time grid alone is not a complete distilled workflow; distillation conditioning and the
near-clean-chunk branch belong to the generation loop.

## MagiEulerScheduler

[[autodoc]] MagiEulerScheduler

## MagiEulerSchedulerOutput

[[autodoc]] schedulers.scheduling_magi_euler.MagiEulerSchedulerOutput
