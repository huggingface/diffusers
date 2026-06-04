# Checkpoint Mechanism for Stage Testing

## Overview

Pipelines are monolithic `__call__` methods -- you can't just call "the encode part". The checkpoint mechanism lets you stop, save, or inject tensors at named locations inside the pipeline.

## The Checkpoint class

Add a `_checkpoints` argument to both the diffusers pipeline and the reference implementation.

```python
@dataclass
class Checkpoint:
    save: bool = False   # capture variables into ckpt.data
    stop: bool = False   # halt pipeline after this point
    load: bool = False   # inject ckpt.data into local variables
    data: dict = field(default_factory=dict)
```

## Pipeline instrumentation

The pipeline accepts an optional `dict[str, Checkpoint]`. Place checkpoint calls at boundaries between pipeline stages -- after each encoder, before the denoising loop (capture all loop inputs), after each loop iteration, after the loop (capture final latents before decode).

```python
def __call__(self, prompt, ..., _checkpoints=None):
    # --- text encoding ---
    prompt_embeds = self.text_encoder(prompt)
    _maybe_checkpoint(_checkpoints, "text_encoding", {
        "prompt_embeds": prompt_embeds,
    })

    # --- prepare latents, sigmas, positions ---
    latents = self.prepare_latents(...)
    sigmas = self.scheduler.sigmas
    # ...

    _maybe_checkpoint(_checkpoints, "preloop", {
        "latents": latents,
        "sigmas": sigmas,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
        "video_coords": video_coords,
        # capture EVERYTHING the loop needs -- every tensor the transformer
        # forward() receives. Missing even one variable here means you can't
        # tell if it's the source of divergence during denoise debugging.
    })

    # --- denoising loop ---
    for i, t in enumerate(timesteps):
        noise_pred = self.transformer(latents, t, prompt_embeds, ...)
        latents = self.scheduler.step(noise_pred, t, latents)[0]

        _maybe_checkpoint(_checkpoints, f"after_step_{i}", {
            "latents": latents,
        })

    _maybe_checkpoint(_checkpoints, "post_loop", {
        "latents": latents,
    })

    # --- decode ---
    video = self.vae.decode(latents)
    return video
```

## The helper function

Each `_maybe_checkpoint` call does three things based on the Checkpoint's flags: `save` captures the local variables into `ckpt.data`, `load` injects pre-populated `ckpt.data` back into local variables, `stop` halts execution (raises an exception caught at the top level).

```python
def _maybe_checkpoint(checkpoints, name, data):
    if not checkpoints:
        return
    ckpt = checkpoints.get(name)
    if ckpt is None:
        return
    if ckpt.save:
        ckpt.data.update(data)
    if ckpt.stop:
        raise PipelineStop  # caught at __call__ level, returns None
```

## Injection support

Add `load` support at each checkpoint where you might want to inject:

```python
_maybe_checkpoint(_checkpoints, "preloop", {"latents": latents, ...})

# Load support: replace local variables with injected data
if _checkpoints:
    ckpt = _checkpoints.get("preloop")
    if ckpt is not None and ckpt.load:
        latents = ckpt.data["latents"].to(device=device, dtype=latents.dtype)
```

## Key insight

The checkpoint dict is passed into the pipeline and mutated in-place. After the pipeline returns (or stops early), you read back `ckpt.data` to get the captured tensors. Both pipelines save under their own key names, so the test maps between them (e.g. reference `"video_state.latent"` -> diffusers `"latents"`).

## Memory management for large models

For large models, free the source pipeline's GPU memory before loading the target pipeline. Clone injected tensors to CPU, delete everything else, then run the target with `enable_model_cpu_offload()`.
