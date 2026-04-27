# Pipeline conventions and rules

Shared reference for pipeline-related conventions, patterns, and gotchas.
Linked from `AGENTS.md`, `skills/model-integration/SKILL.md`, and `review-rules.md`.

## Common pipeline conventions

When adding a new pipeline (or reviewing one), skim `pipeline_flux.py`, `pipeline_flux2.py`, `pipeline_qwenimage.py`, `pipeline_wan.py` first to establish the pattern. Most conventions (class structure, mixin set, `__call__` shape — input validation → encode prompt → timesteps → latent prep → denoise loop → decode — `encode_prompt` / `prepare_latents` shape, `output_type` / `generator` / `progress_bar` plumbing, `@torch.no_grad()` on `__call__`, LoRA mixin, `from_single_file` support, etc.) are easiest to internalize by comparison rather than from a fixed list.

## Gotchas

1. **Config-derived static values: prefer `__init__` attributes.** Values that come from a sub-component's config (e.g. `vae_scale_factor`) belong as `self.foo = ...` in `__init__` — not `@property`, not module-level constants. Note the `getattr(...)` fallback — sub-components may not be loaded when the pipeline is constructed (e.g. via `from_pretrained` on a partial config), so don't assume `self.vae` / `self.transformer` exists.
   ```python
   # don't do this — @property for static config value
   @property
   def is_turbo(self) -> bool:
       return bool(getattr(self.transformer.config, "is_turbo", False))

   # don't do this — module-level constant duplicating loadable config
   SAMPLE_RATE = 48000

   # do this — set once in __init__ with a getattr fallback (see pipeline_flux.py:209)
   def __init__(self, ..., vae, transformer, ...):
       ...
       self.register_modules(vae=vae, transformer=transformer, ...)
       self.vae_scale_factor = (
           2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
       )
       self.sample_rate = int(self.vae.config.sampling_rate) if getattr(self, "vae", None) else 48000
   ```
   `@property` is reserved for per-call state — values that depend on something set inside `__call__` (e.g. `do_classifier_free_guidance` reading `self._guidance_scale`).

2. **`@torch.no_grad()` discipline.** Two failure modes:
    - **Missing on `__call__` entirely** — causes GPU OOM from gradient accumulation during inference. Always decorate `__call__` with `@torch.no_grad()`.
    - **Redundant inside helpers** that `__call__` already covers. The decorator puts every descendent in no-grad, so an inner `with torch.no_grad():` is noise — and worse, it forecloses callers who want to invoke `pipe.encode_prompt(...)` with grads enabled (training, embedding optimization). Convention across diffusers (flux, qwen, flux2, stable_audio, audioldm2) is decorator-only.

3. **Reinventing logic that already exists in the repo.** Check `src/diffusers/guiders/` and `src/diffusers/schedulers/` before adding new logic. Reuse what's already there; extend with a small kwarg for minor variations.
    - **Schedulers / guiders** — grep `src/diffusers/guiders/` and `src/diffusers/schedulers/` first. APG, CFG variants, DDIM, DPM++, flow matching Euler etc. are all already in the repo.
    - **Reimplementing what the scheduler already does.** Two examples below, both forms of "the scheduler should own this":
      ```python
      # don't do this - bypassing the scheduler entirely and rolling your own step
      for t in custom_timesteps:
          noise_pred = self.transformer(...)
          latents = latents - sigma * noise_pred   # custom Euler step, no scheduler.step()

      # don't do this — using the scheduler but inlining its default sigma math
      # (this is exactly what FlowMatchEulerDiscreteScheduler computes with shift=N — not a custom case)
      sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
      sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
      self.scheduler.set_timesteps(sigmas=sigmas, device=device)

      # good — let the scheduler own it
      self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
      for t in self.scheduler.timesteps:
          noise_pred = self.transformer(...)
          latents = self.scheduler.step(noise_pred, t, latents).prev_sample
      ```
      If the inlined math matches the scheduler's default, walk through one row by hand to check, delete it and configure the scheduler instead.

4. **Subclassing an existing pipeline for a variant.** Don't use an existing pipeline class (e.g. `FluxPipeline`) to override another (e.g. `FluxImg2ImgPipeline`) inside the core `src/` codebase. Each pipeline lives in its own file with its own class, even if it shares 90% of `__call__` with a sibling. Convention across diffusers — flux, sdxl, wan, qwenimage — is duplicated `__call__` between img2img / text2img / inpaint variants, not subclassing. Reuse private utilities (shared schedulers, prep functions) but not the pipeline class itself.

5. **Copying a method from another pipeline without `# Copied from`.** When you reuse a method like `encode_prompt`, `prepare_latents`, `check_inputs`, or `_prepare_latent_image_ids` from another pipeline, add a `# Copied from` annotation so `make fix-copies` keeps the two in sync. Forgetting it means future refactors to the source drift away from your copy silently — and reviewers waste time spotting near-identical code that should have been linked. The annotation grammar (decorator placement, rename syntax with `with old->new`, etc.) is implemented in [`utils/check_copies.py`](../utils/check_copies.py) — read it for the exact rules.
