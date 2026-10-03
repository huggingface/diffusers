# Numerical Discrepancy Pitfalls

A reference list of things that have caused numerical discrepancies between an original/reference implementation and the diffusers port. It's not a checklist — most won't apply to any given model; consult it only when the diffusers outputs don't match the reference.

## 1. Global CPU RNG
`MultivariateNormal.sample()` uses the global CPU RNG, not `torch.Generator`. Must call `torch.manual_seed(seed)` before each pipeline run. A `generator=` kwarg won't help.

## 2. Timestep dtype
Many transformers expect `int64` timesteps. `get_timestep_embedding` casts to float, so `745.3` and `745` produce different embeddings. Match the reference's casting.

## 3. Guidance parameter mapping
Parameter names may differ: reference `zero_steps=1` (meaning `i <= 1`, 2 steps) vs target `zero_init_steps=2` (meaning `step < 2`, same thing). Check exact semantics.

## 4. `patch_size` in noise generation
If noise generation depends on `patch_size`, it must be passed through. Missing it changes noise spatial structure.

## 5. Float precision differences -- don't dismiss them
Small per-element diffs from a dtype mismatch (e.g. float32 vs bfloat16, ~1e-3 to 1e-2) look harmless, but in an iterative process like the denoising loop they can compound into a large final difference (see #9 and #11). Check whether a precision diff feeds an iterative process before accepting it.

## 6. Scheduler state reset between stages
Some schedulers accumulate state (e.g. `model_outputs` in UniPC) that must be cleared between stages.

## 7. Component access
Standard: `self.transformer`. Modular: `components.transformer`. Missing this causes AttributeError.

## 8. Guider state across stages
In multi-stage denoising, the guider's internal state (e.g. `zero_init_steps`) may need save/restore between stages.

## 9. Noise dtype mismatch

Reference code often generates noise in float32 then casts to model dtype (bfloat16) before storing:

```python
noise = torch.randn(..., dtype=torch.float32, generator=gen)
noise = noise.to(dtype=model_dtype)  # bfloat16 -- values get quantized
```

Diffusers pipelines may keep latents in float32 throughout the loop. The per-element difference is only ~1.5e-02, but this compounds over 30 denoising steps via 1/sigma amplification (#11) and produces completely washed-out output.

**Fix**: Match the reference -- generate noise in the model's working dtype:
```python
latent_dtype = self.transformer.dtype  # e.g. bfloat16
latents = self.prepare_latents(..., dtype=latent_dtype, ...)
```

## 10. RoPE position dtype

RoPE cosine/sine values are sensitive to position coordinate dtype. If reference uses bfloat16 positions but diffusers uses float32, the RoPE output diverges significantly.

## 11. 1/sigma error amplification in Euler denoising

In Euler/flow-matching, the velocity formula divides by sigma: `v = (latents - pred_x0) / sigma`. As sigma shrinks from ~1.0 (step 0) to ~0.001 (step 29), errors are amplified up to 1000x. A 1.5e-02 init difference grows linearly through mid-steps, then exponentially in final steps. This is why dtype mismatches (#9, #10) that seem tiny at init produce visually broken output.

## 12. Config value assumptions

Don't assume config values match the code defaults: the published checkpoint may override them (and so may the diffusers config). Look up the actual config.
