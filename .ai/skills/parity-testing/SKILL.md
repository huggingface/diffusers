---
name: Parity Testing
description: >
  Testing pipeline parity between reference and diffusers implementations:
  checkpoint mechanism, stage tests (encode/decode/denoise), injection debugging,
  visual comparison, comparison utilities, and 18 common pitfalls.
  Trigger: debugging parity, writing conversion tests, investigating divergence.
---

## Part 3: Testing Pipeline Parity

Applies to any conversion: research repo -> diffusers, standard -> modular, or research repo -> modular.

### Principles

1. **Don't combine structural changes with behavioral changes.** For research repo -> diffusers, you must restructure code to fit diffusers APIs (ModelMixin, ConfigMixin, etc.) -- that's unavoidable. But don't also "improve" the algorithm, refactor computation order, or rename internal variables for aesthetics. Keep numerical logic as close to the reference as possible, even if it looks ugly. For standard -> modular, this is stricter: copy loop logic verbatim and only restructure into blocks. In both cases, clean up in a separate commit after parity is confirmed.

2. **Match the reference noise generation first.** The way initial noise/latents are constructed (seed handling, generator, randn call order) often differs between reference and diffusers. If the noise doesn't match, nothing downstream will match, making it impossible to isolate other bugs. Strategy: in the first implementation, replicate the reference's exact noise construction to get parity. After everything else is confirmed working, swap to diffusers-style noise generation as a final step.

3. **Test from the start.** Have component tests ready BEFORE writing conversion code. Test bottom-up: components first, then pipeline stages, then e2e.

### Test strategy

**Step 1: Component parity (CPU/float32) -- always run, as you build.**
Test each component before assembling the pipeline. This is the foundation -- if individual pieces are wrong, the pipeline can't be right. Each component in isolation, strict max_diff < 1e-3. Two modes:
- **Fresh**: convert from checkpoint weights, compare against reference (catches conversion bugs)
- **Saved**: load from saved model on disk, compare against reference (catches stale saves)

Keep component test scripts around -- you will need to re-run them during pipeline debugging with different inputs or config values. For example, you might initially test a transformer with random inputs, then later re-run it with actual pipeline-captured inputs to confirm it still matches. Having the test ready and easy to modify saves significant time.

Template -- one self-contained script per component, reference and diffusers side-by-side:
```python
@torch.inference_mode()
def test_my_component(mode="fresh", model_path=None):
    # 1. Deterministic input
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(1, 3, 64, 64, generator=gen, dtype=torch.float32)

    # 2. Reference: load from checkpoint, run, free
    ref_model = ReferenceModel.from_config(config)
    ref_model.load_state_dict(load_weights("prefix"), strict=True)
    ref_model = ref_model.float().eval()
    ref_out = ref_model(x).clone()
    del ref_model

    # 3. Diffusers: fresh (convert weights) or saved (from_pretrained)
    if mode == "fresh":
        diff_model = convert_my_component(load_weights("prefix"))
    else:
        diff_model = DiffusersModel.from_pretrained(model_path, torch_dtype=torch.float32)
    diff_model = diff_model.float().eval()
    diff_out = diff_model(x)
    del diff_model

    # 4. Compare in same script -- no saving to disk
    max_diff = (ref_out - diff_out).abs().max().item()
    assert max_diff < 1e-3, f"FAIL: max_diff={max_diff:.2e}"
```
Key points: (a) both sides in one script -- never split into separate scripts that save/load intermediates, (b) deterministic input via seeded generator, (c) load one model at a time to fit in CPU RAM, (d) `.clone()` the reference output before deleting the model.

**Step 2: E2E visual (GPU/bfloat16) -- once the pipeline is assembled.**
Both pipelines generate independently with identical seeds/params. Save outputs and compare visually. If outputs look identical, you're done -- no need for deeper testing.

**Step 3: Pipeline stage tests -- only if E2E fails and you need to isolate the bug.**
For small models, run on CPU/float32 for strict comparison. For large models (e.g. 22B params), CPU/float32 is impractical -- use GPU/bfloat16 with `enable_model_cpu_offload()` and relax tolerances (max_diff < 1e-1 for bfloat16 is typical for passing tests; cosine similarity > 0.9999 is a good secondary check).

Test encode and decode stages first -- they're simpler and bugs there are easier to fix. Only debug the denoising loop if encode and decode both pass.

The challenge: pipelines are monolithic `__call__` methods -- you can't just call "the encode part". The solution is a checkpoint mechanism that lets you stop, save, or inject tensors at named locations inside the pipeline.

**Step 1: Add a `_checkpoints` argument to both pipelines.**

The Checkpoint class is minimal:
```python
@dataclass
class Checkpoint:
    save: bool = False   # capture variables into ckpt.data
    stop: bool = False   # halt pipeline after this point
    load: bool = False   # inject ckpt.data into local variables
    data: dict = field(default_factory=dict)
```

The pipeline accepts an optional `dict[str, Checkpoint]`. Place checkpoint calls at boundaries between pipeline stages -- after each encoder, before the denoising loop (capture all loop inputs), after each loop iteration, after the loop (capture final latents before decode). Here's a skeleton showing where they go:

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

Each `_maybe_checkpoint` call does three things based on the Checkpoint's flags: `save` captures the local variables into `ckpt.data`, `load` injects pre-populated `ckpt.data` back into local variables, `stop` halts execution (raises an exception caught at the top level). The helper:

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

**Step 2: Write stage tests using checkpoints.**

Three stages, tested in this order -- **encode, decode, then denoise**:

- **`encode`** (test first): Stop both pipelines at `"preloop"`. Compare **every single variable** that will be consumed by the denoising loop -- not just latents and sigmas, but also prompt embeddings, attention masks, positional coordinates, connector outputs, and any conditioning inputs. If you only compare a subset, you'll miss divergent inputs and waste time debugging the loop for a bug that's actually upstream. List every argument the transformer's forward() takes and make sure each one is compared.
- **`decode`** (test second, before denoise): Run the reference pipeline fully -- checkpoint the post-loop latents AND let it finish to get the decoded output. Then feed those same post-loop latents through the diffusers pipeline's decode path (unpacking, denormalization, VAE decode, etc). Compare the two decoded outputs. **Always test decode before spending time on denoise.** Decoder bugs (e.g. wrong config values, incorrect operation ordering) can cause severe visual artifacts (pixelation, color shifts) that look like denoising bugs but are much simpler to fix. Always visually inspect decoded output -- numerical metrics like PSNR can be misleadingly "close" (e.g. 28 dB) while hiding obvious visual defects.
- **`denoise`** (test last): Run both pipelines with realistic `num_steps` (e.g. 30) so the scheduler computes correct sigmas/timesteps, but stop after 2 loop iterations using `after_step_1`. Don't set `num_steps=2` -- that produces unrealistic sigma schedules. Compare the latents after those 2 steps.

```python
# Encode stage -- stop before the loop, compare ALL inputs:
ref_ckpts = {"preloop": Checkpoint(save=True, stop=True)}
run_reference_pipeline(ref_ckpts)
ref_data = ref_ckpts["preloop"].data

diff_ckpts = {"preloop": Checkpoint(save=True, stop=True)}
run_diffusers_pipeline(diff_ckpts)
diff_data = diff_ckpts["preloop"].data

# Compare EVERY variable consumed by the denoise loop:
compare_tensors("latents", ref_data["latents"], diff_data["latents"])
compare_tensors("sigmas", ref_data["sigmas"], diff_data["sigmas"])
compare_tensors("prompt_embeds", ref_data["prompt_embeds"], diff_data["prompt_embeds"])
compare_tensors("prompt_attention_mask", ref_data["prompt_attention_mask"], diff_data["prompt_attention_mask"])
compare_tensors("video_coords", ref_data["video_coords"], diff_data["video_coords"])
# ... every single tensor the transformer forward() will receive

# Decode stage -- same latents through both decoders:
ref_ckpts = {"post_loop": Checkpoint(save=True)}
run_reference_pipeline(ref_ckpts)
ref_latents = ref_ckpts["post_loop"].data["latents"]
# Feed ref_latents through diffusers decode path, compare output visually AND numerically

# Denoise stage -- realistic steps, early stop after 2 iterations:
ref_ckpts = {"after_step_1": Checkpoint(save=True, stop=True)}
run_reference_pipeline(ref_ckpts)  # uses default num_steps=30
compare_tensors("latents", ref_ckpts[...].data["latents"], diff_ckpts[...].data["latents"])
```

The key insight: the checkpoint dict is passed into the pipeline and mutated in-place. After the pipeline returns (or stops early), you read back `ckpt.data` to get the captured tensors. Both pipelines save under their own key names, so the test maps between them (e.g. reference `"video_state.latent"` -> diffusers `"latents"`).

**E2E-injected visual test**: Once you've identified a suspected root cause using stage tests, confirm it with an e2e-injected run -- inject the known-good tensor from reference and generate a full video. If the output looks identical to reference, you've confirmed the root cause. Fix it, then re-run the standard E2E test to verify.

### Debugging technique: Injection for root-cause isolation

When stage tests show divergence, you need to narrow down *which input* is causing it. The general technique: **inject a known-good tensor from one pipeline into the other** to test whether the remaining code is correct.

The principle is simple -- if you suspect input X is the root cause of divergence in stage S:
1. Run the reference pipeline and capture X
2. Run the diffusers pipeline but **replace** its X with the reference's X (via checkpoint load)
3. Compare outputs of stage S

If outputs now match: X was the root cause. If they still diverge: the bug is in the stage logic itself, not in X.

This is the same pattern applied at different pipeline boundaries:

| What you're testing | What you inject | Where you inject |
|---|---|---|
| Is the decode stage correct? | Post-loop latents from reference | Before decode |
| Is the denoise loop correct? | Pre-loop latents from reference | Before the loop |
| Is step N correct? | Post-step-(N-1) latents from reference | Before step N |

Add `load` support at each checkpoint where you might want to inject:

```python
_maybe_checkpoint(_checkpoints, "preloop", {"latents": latents, ...})

# Load support: replace local variables with injected data
if _checkpoints:
    ckpt = _checkpoints.get("preloop")
    if ckpt is not None and ckpt.load:
        latents = ckpt.data["latents"].to(device=device, dtype=latents.dtype)
```

For large models, free the source pipeline's GPU memory before loading the target pipeline. Clone injected tensors to CPU, delete everything else, then run the target with `enable_model_cpu_offload()`.

**Per-step accumulation tracing**: When injection confirms the loop is correct but you want to understand *how* a small initial difference compounds, capture `after_step_{i}` for every step and plot the max_diff curve. A healthy curve stays bounded; an exponential blowup in later steps points to an amplification mechanism (see Pitfall #13).

### Debugging technique: Visual comparison via frame extraction

For video pipelines, numerical metrics alone can be misleading (max_diff=0.25 might look identical, or max_diff=0.05 might be visibly wrong in specific regions). Extract and view individual frames programmatically:

```python
import numpy as np
from PIL import Image

def extract_frames(video_np, frame_indices):
    """video_np: (frames, H, W, 3) float array in [0, 1]"""
    for idx in frame_indices:
        frame = (video_np[idx] * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(frame)
        img.save(f"frame_{idx}.png")

# Compare specific frames from both pipelines
extract_frames(ref_video, [0, 60, 120])
extract_frames(diff_video, [0, 60, 120])
```

This is especially useful for: (a) confirming a fix works before running expensive full-pipeline tests, (b) diagnosing *what kind* of visual artifact a numerical divergence produces (washed out? color shift? spatial distortion?), (c) e2e-injected tests where you want visual proof that the loop is correct when given identical inputs.

### Testing rules

1. **Never use reference code in the diffusers test path.** Each side must use only its own code. Using reference helper functions inside the diffusers path defeats the purpose -- you're no longer testing the diffusers implementation.
2. **Never monkey-patch model internals in tests.** Do not replace `model.forward` or patch internal methods. A passing test with a patched forward proves nothing about the actual model.
3. **Debugging instrumentation must be non-destructive.** Checkpoint captures (e.g. a `_checkpoint` dict) for debugging are fine, but must not alter control flow or outputs.
4. **Prefer CPU/float32 for numerical comparison when practical.** Float32 avoids bfloat16 precision noise that obscures real bugs. But for large models (22B+), GPU/bfloat16 with `enable_model_cpu_offload()` is necessary -- use relaxed tolerances and cosine similarity as a secondary metric.
5. **Test both fresh conversion AND saved model.** Fresh catches conversion logic bugs; saved catches stale/corrupted weights from previous runs.
6. **Diff configs before debugging.** Before investigating any divergence, dump and compare all config values from both the reference checkpoint and the diffusers model. Reference configs can often be read from checkpoint metadata without loading the model. Don't trust code defaults -- the checkpoint may override them. A 30-second config diff prevents hours of debugging based on wrong assumptions.
7. **Never modify cached/downloaded model configs directly.** If you need to test with a different config value (e.g. fixing `upsample_residual` from `true` to `false`), do NOT edit the file in `~/.cache/huggingface/`. That change is invisible -- no git tracking, no diff, easy to forget. Instead, either (a) save the model to a local repo directory and edit the config there, or (b) open a PR on the upstream HF repo and load with `revision="refs/pr/N"`. Both approaches make the change visible and trackable.
8. **Test decode before denoise.** Always verify the decoder works correctly before spending time on the denoising loop. Feed identical post-loop latents from the reference through both decoders and compare outputs -- both numerically AND visually. Decoder config bugs (e.g. wrong `upsample_residual`) cause severe pixelation or artifacts that are trivial to fix once found, but look like denoising bugs from the E2E output. A decoder bug found after days of denoise debugging is wasted time.
9. **Compare ALL loop inputs in the encode test.** The preloop checkpoint must capture every single tensor the transformer forward() will receive: latents, sigmas/timesteps, prompt embeddings, attention masks, positional coordinates, connector outputs, and any conditioning tensors. If you only compare latents and sigmas, you'll miss divergent conditioning inputs and waste time debugging the loop for a bug that's actually upstream.

### Comparison utilities

```python
def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = 1e-3) -> bool:
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item()
    passed = max_diff < tol
    print(f"  {'PASS' if passed else 'FAIL'} {name}: max={max_diff:.2e}, mean={mean_diff:.2e}, cos={cos:.5f}")
    return passed
```
Cosine similarity is especially useful for GPU/bfloat16 tests where max_diff can be noisy -- `cos > 0.9999` is a strong signal even when max_diff exceeds tolerance.

## Part 4: Common Pitfalls

### 1. Global CPU RNG
`MultivariateNormal.sample()` uses the global CPU RNG, not `torch.Generator`. Must call `torch.manual_seed(seed)` before each pipeline run. A `generator=` kwarg won't help.

### 2. Timestep dtype
Many transformers expect `int64` timesteps. `get_timestep_embedding` casts to float, so `745.3` and `745` produce different embeddings. Match the reference's casting.

### 3. Guidance parameter mapping
Parameter names may differ: reference `zero_steps=1` (meaning `i <= 1`, 2 steps) vs target `zero_init_steps=2` (meaning `step < 2`, same thing). Check exact semantics.

### 4. `patch_size` in noise generation
If noise generation depends on `patch_size` (e.g. `sample_block_noise`), it must be passed through. Missing it changes noise spatial structure.

### 5. Variable shadowing in nested loops
Nested loops (stages -> chunks -> timesteps) can shadow variable names. If outer loop uses `latents` and inner loop also assigns to `latents`, scoping must match the reference.

### 6. Float precision differences -- don't dismiss them
Target may compute in float32 where reference used bfloat16. Small per-element diffs (1e-3 to 1e-2) *look* harmless but can compound catastrophically over iterative processes like denoising loops (see Pitfalls #11 and #13). Before dismissing a precision difference: (a) check whether it feeds into an iterative process, (b) if so, trace the accumulation curve over all iterations to see if it stays bounded or grows exponentially. Only truly non-iterative precision diffs (e.g. in a single-pass encoder) are safe to accept.

### 7. Scheduler state reset between stages
Some schedulers accumulate state (e.g. `model_outputs` in UniPC) that must be cleared between stages.

### 8. Component access
Standard: `self.transformer`. Modular: `components.transformer`. Missing this causes AttributeError.

### 9. Guider state across stages
In multi-stage denoising, the guider's internal state (e.g. `zero_init_steps`) may need save/restore between stages.

### 10. Model storage location
NEVER store converted models in `/tmp/` -- temporary directories get wiped on restart. Always save converted checkpoints under a persistent path in the project repo (e.g. `models/ltx23-diffusers/`).

### 11. Noise dtype mismatch (causes washed-out output)

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

### 12. RoPE position dtype

RoPE cosine/sine values are sensitive to position coordinate dtype. If reference uses bfloat16 positions but diffusers uses float32, the RoPE output diverges significantly (max_diff up to 2.0). Different modalities may use different position dtypes (e.g. video bfloat16, audio float32) -- check the reference carefully.

### 13. 1/sigma error amplification in Euler denoising

In Euler/flow-matching, the velocity formula divides by sigma: `v = (latents - pred_x0) / sigma`. As sigma shrinks from ~1.0 (step 0) to ~0.001 (step 29), errors are amplified up to 1000x. A 1.5e-02 init difference grows linearly through mid-steps, then exponentially in final steps, reaching max_diff ~6.0. This is why dtype mismatches (Pitfalls #11, #12) that seem tiny at init produce visually broken output. Use per-step accumulation tracing to diagnose.

### 14. Config value assumptions -- always diff, never assume

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

### 15. Decoder config mismatch (causes pixelated artifacts)

The upstream model config may have wrong values for decoder-specific parameters (e.g. `upsample_residual`, `upsample_type`). These control whether the decoder uses skip connections in upsampling -- getting them wrong produces severe pixelation or blocky artifacts.

**Detection**: Feed identical post-loop latents through both decoders. If max pixel diff is large (PSNR < 40 dB) on CPU/float32, it's a real bug, not precision noise. Trace through decoder blocks (conv_in -> mid_block -> up_blocks) to find where divergence starts.

**Fix**: Correct the config value. Don't edit cached files in `~/.cache/huggingface/` -- either save to a local model directory or open a PR on the upstream repo (see Testing Rule #7).

### 16. Incomplete injection tests -- inject ALL variables or the test is invalid

When doing injection tests (feeding reference tensors into the diffusers pipeline), you must inject **every** divergent input, including sigmas/timesteps. A common mistake: the preloop checkpoint saves sigmas but the injection code only loads latents and embeddings. The test then runs with different sigma schedules, making it impossible to isolate the real cause.

**Prevention**: After writing injection code, verify by listing every variable the injected stage consumes and checking each one is either (a) injected from reference, or (b) confirmed identical between pipelines.

### 17. bf16 connector/encoder divergence -- don't chase it

When running on GPU/bfloat16, multi-layer encoders (e.g. 8-layer connector transformers) accumulate bf16 rounding noise that looks alarming (max_diff 0.3-2.7). Before investigating, re-run the component test on CPU/float32. If it passes (max_diff < 1e-4), the divergence is pure precision noise, not a code bug. Don't spend hours tracing through layers -- confirm on CPU/f32 and move on.

### 18. Stale test fixtures

When using saved tensors for cross-pipeline comparison, always ensure both sets of tensors were captured from the same run configuration (same seed, same config, same code version). Mixing fixtures from different runs (e.g. reference tensors from yesterday, diffusers tensors from today after a code change) creates phantom divergence that wastes debugging time. Regenerate both sides in a single test script execution.
