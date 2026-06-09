# Complete Pitfalls Reference

## 1. Global CPU RNG
`MultivariateNormal.sample()` uses the global CPU RNG, not `torch.Generator`. Must call `torch.manual_seed(seed)` before each pipeline run. A `generator=` kwarg won't help.

## 2. Timestep dtype
Many transformers expect `int64` timesteps. `get_timestep_embedding` casts to float, so `745.3` and `745` produce different embeddings. Match the reference's casting.

## 3. Guidance parameter mapping
Parameter names may differ: reference `zero_steps=1` (meaning `i <= 1`, 2 steps) vs target `zero_init_steps=2` (meaning `step < 2`, same thing). Check exact semantics.

## 4. `patch_size` in noise generation
If noise generation depends on `patch_size` (e.g. `sample_block_noise`), it must be passed through. Missing it changes noise spatial structure.

## 5. Variable shadowing in nested loops
Nested loops (stages -> chunks -> timesteps) can shadow variable names. If outer loop uses `latents` and inner loop also assigns to `latents`, scoping must match the reference.

## 6. Float precision differences -- don't dismiss them
Target may compute in float32 where reference used bfloat16. Small per-element diffs (1e-3 to 1e-2) *look* harmless but can compound catastrophically over iterative processes like denoising loops (see Pitfalls #11 and #13). Before dismissing a precision difference: (a) check whether it feeds into an iterative process, (b) if so, trace the accumulation curve over all iterations to see if it stays bounded or grows exponentially. Only truly non-iterative precision diffs (e.g. in a single-pass encoder) are safe to accept.

## 7. Scheduler state reset between stages
Some schedulers accumulate state (e.g. `model_outputs` in UniPC) that must be cleared between stages.

## 8. Component access
Standard: `self.transformer`. Modular: `components.transformer`. Missing this causes AttributeError.

## 9. Guider state across stages
In multi-stage denoising, the guider's internal state (e.g. `zero_init_steps`) may need save/restore between stages.

## 10. Model storage location
NEVER store converted models in `/tmp/` -- temporary directories get wiped on restart. Always save converted checkpoints under a persistent path in the project repo (e.g. `models/ltx23-diffusers/`).

## 11. Noise dtype mismatch (causes washed-out output)

Reference code often generates noise in float32 then casts to model dtype (bfloat16) before storing:

```python
noise = torch.randn(..., dtype=torch.float32, generator=gen)
noise = noise.to(dtype=model_dtype)  # bfloat16 -- values get quantized
```

Diffusers pipelines may keep latents in float32 throughout the loop. The per-element difference is only ~1.5e-02, but this compounds over 30 denoising steps via 1/sigma amplification (Pitfall #13) and produces completely washed-out output.

**Fix**: Match the reference -- generate noise in the model's working dtype:
```python
latent_dtype = self.transformer.dtype  # e.g. bfloat16
latents = self.prepare_latents(..., dtype=latent_dtype, ...)
```

**Detection**: Encode stage test shows initial latent max_diff of exactly ~1.5e-02. This specific magnitude is the signature of float32->bfloat16 quantization error.

## 12. RoPE position dtype

RoPE cosine/sine values are sensitive to position coordinate dtype. If reference uses bfloat16 positions but diffusers uses float32, the RoPE output diverges significantly (max_diff up to 2.0). Different modalities may use different position dtypes (e.g. video bfloat16, audio float32) -- check the reference carefully.

## 13. 1/sigma error amplification in Euler denoising

In Euler/flow-matching, the velocity formula divides by sigma: `v = (latents - pred_x0) / sigma`. As sigma shrinks from ~1.0 (step 0) to ~0.001 (step 29), errors are amplified up to 1000x. A 1.5e-02 init difference grows linearly through mid-steps, then exponentially in final steps, reaching max_diff ~6.0. This is why dtype mismatches (Pitfalls #11, #12) that seem tiny at init produce visually broken output. Use per-step accumulation tracing to diagnose.

## 14. Config value assumptions -- always diff, never assume

When debugging parity, don't assume config values match code defaults. The published model checkpoint may override defaults with different values. A wrong assumption about a single config field can send you down hours of debugging in the wrong direction.

**The pattern that goes wrong:**
1. You see `param_x` has default `1` in the code
2. The reference code also uses `param_x` with a default of `1`
3. You assume both sides use `1` and apply a "fix" based on that
4. But the actual checkpoint config has `param_x: 1000`, and so does the published diffusers config
5. Your "fix" now *creates* divergence instead of fixing it

**Prevention -- config diff first:**
```python
# Reference: read from checkpoint metadata (no model loading needed)
from safetensors import safe_open
import json
ref_config = json.loads(safe_open(checkpoint_path, framework="pt").metadata()["config"])

# Diffusers: read from model config
from diffusers import MyModel
diff_model = MyModel.from_pretrained(model_path, subfolder="transformer")
diff_config = dict(diff_model.config)

# Compare all values
for key in sorted(set(list(ref_config.get("transformer", {}).keys()) + list(diff_config.keys()))):
    ref_val = ref_config.get("transformer", {}).get(key, "MISSING")
    diff_val = diff_config.get(key, "MISSING")
    if ref_val != diff_val:
        print(f"  DIFF {key}: ref={ref_val}, diff={diff_val}")
```

Run this **before** writing any hooks, analysis code, or fixes. It takes 30 seconds and catches wrong assumptions immediately.

**When debugging divergence -- trace values, don't reason about them:**
If two implementations diverge, hook the actual intermediate values at the point of divergence rather than reading code to figure out what the values "should" be. Code analysis builds on assumptions; value tracing reveals facts.

## 15. Decoder config mismatch (causes pixelated artifacts)

The upstream model config may have wrong values for decoder-specific parameters (e.g. `upsample_residual`, `upsample_type`). These control whether the decoder uses skip connections in upsampling -- getting them wrong produces severe pixelation or blocky artifacts.

**Detection**: Feed identical post-loop latents through both decoders. If max pixel diff is large (PSNR < 40 dB) on CPU/float32, it's a real bug, not precision noise. Trace through decoder blocks (conv_in -> mid_block -> up_blocks) to find where divergence starts.

**Fix**: Correct the config value. Don't edit cached files in `~/.cache/huggingface/` -- either save to a local model directory or open a PR on the upstream repo (see Testing Rule #7).

## 16. Incomplete injection tests -- inject ALL variables or the test is invalid

When doing injection tests (feeding reference tensors into the diffusers pipeline), you must inject **every** divergent input, including sigmas/timesteps. A common mistake: the preloop checkpoint saves sigmas but the injection code only loads latents and embeddings. The test then runs with different sigma schedules, making it impossible to isolate the real cause.

**Prevention**: After writing injection code, verify by listing every variable the injected stage consumes and checking each one is either (a) injected from reference, or (b) confirmed identical between pipelines.

## 17. bf16 connector/encoder divergence -- don't chase it

When running on GPU/bfloat16, multi-layer encoders (e.g. 8-layer connector transformers) accumulate bf16 rounding noise that looks alarming (max_diff 0.3-2.7). Before investigating, re-run the component test on CPU/float32. If it passes (max_diff < 1e-4), the divergence is pure precision noise, not a code bug. Don't spend hours tracing through layers -- confirm on CPU/float32 and move on.

## 18. Stale test fixtures

When using saved tensors for cross-pipeline comparison, always ensure both sets of tensors were captured from the same run configuration (same seed, same config, same code version). Mixing fixtures from different runs (e.g. reference tensors from yesterday, diffusers tensors from today after a code change) creates phantom divergence that wastes debugging time. Regenerate both sides in a single test script execution.
