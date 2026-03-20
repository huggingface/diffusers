---
name: testing-parity
description: >
  Use when debugging or verifying numerical parity between pipeline
  implementations (e.g., research repo vs diffusers, standard vs modular).
  Also relevant when outputs look wrong — washed out, pixelated, or have
  visual artifacts — as these are usually parity bugs.
---

## Setup — gather before starting

Before writing any test code, gather:

1. **Which two implementations** are being compared (e.g. research repo → diffusers, standard → modular, or research → modular). Use `AskUserQuestion` with structured choices if not already clear.
2. **Two equivalent runnable scripts** — one for each implementation, both expected to produce identical output given the same inputs. These scripts define what "parity" means concretely.

When invoked from the `model-integration` skill, you already have context: the reference script comes from step 2 of setup, and the diffusers script is the one you just wrote. You just need to make sure both scripts are runnable and use the same inputs/seed/params.

## Test strategy

**Component parity (CPU/float32) -- always run, as you build.**
Test each component before assembling the pipeline. This is the foundation -- if individual pieces are wrong, the pipeline can't be right. Each component in isolation, strict max_diff < 1e-3.

Test freshly converted checkpoints and saved checkpoints.
- **Fresh**: convert from checkpoint weights, compare against reference (catches conversion bugs)
- **Saved**: load from saved model on disk, compare against reference (catches stale saves)

Keep component test scripts around -- you will need to re-run them during pipeline debugging with different inputs or config values.

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
Key points: (a) both reference and diffusers component in one script -- never split into separate scripts that save/load intermediates, (b) deterministic input via seeded generator, (c) load one model at a time to fit in CPU RAM, (d) `.clone()` the reference output before deleting the model.

**E2E visual (GPU/bfloat16) -- once the pipeline is assembled.**
Both pipelines generate independently with identical seeds/params. Save outputs and compare visually. If outputs look identical, you're done -- no need for deeper testing.

**Pipeline stage tests -- only if E2E fails and you need to isolate the bug.**
If the user already suspects where divergence is, start there. Otherwise, work through stages in order.

First, **match noise generation**: the way initial noise/latents are constructed (seed handling, generator, randn call order) often differs between the two scripts. If the noise doesn't match, nothing downstream will match. Check how noise is initialized in the diffusers script — if it doesn't match the reference, temporarily change it to match. Note what you changed so it can be reverted after parity is confirmed.

For small models, run on CPU/float32 for strict comparison. For large models (e.g. 22B params), CPU/float32 is impractical -- use GPU/bfloat16 with `enable_model_cpu_offload()` and relax tolerances (max_diff < 1e-1 for bfloat16 is typical for passing tests; cosine similarity > 0.9999 is a good secondary check).

Test encode and decode stages first -- they're simpler and bugs there are easier to fix. Only debug the denoising loop if encode and decode both pass.

The challenge: pipelines are monolithic `__call__` methods -- you can't just call "the encode part". See [checkpoint-mechanism.md](checkpoint-mechanism.md) for the checkpoint class that lets you stop, save, or inject tensors at named locations inside the pipeline.

**Stage test order — encode, decode, then denoise:**

- **`encode`** (test first): Stop both pipelines at `"preloop"`. Compare **every single variable** that will be consumed by the denoising loop -- not just latents and sigmas, but also prompt embeddings, attention masks, positional coordinates, connector outputs, and any conditioning inputs.
- **`decode`** (test second, before denoise): Run the reference pipeline fully -- checkpoint the post-loop latents AND let it finish to get the decoded output. Then feed those same post-loop latents through the diffusers pipeline's decode path. Compare both numerically AND visually.
- **`denoise`** (test last): Run both pipelines with realistic `num_steps` (e.g. 30) so the scheduler computes correct sigmas/timesteps, but stop after 2 loop iterations using `after_step_1`. Don't set `num_steps=2` -- that produces unrealistic sigma schedules.

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
# ... every single tensor the transformer forward() will receive
```

**E2E-injected visual test**: Once you've identified a suspected root cause using stage tests, confirm it with an e2e-injected run -- inject the known-good tensor from reference and generate a full video. If the output looks identical to reference, you've confirmed the root cause.

## Debugging technique: Injection for root-cause isolation

When stage tests show divergence, **inject a known-good tensor from one pipeline into the other** to test whether the remaining code is correct.

The principle: if you suspect input X is the root cause of divergence in stage S:
1. Run the reference pipeline and capture X
2. Run the diffusers pipeline but **replace** its X with the reference's X (via checkpoint load)
3. Compare outputs of stage S

If outputs now match: X was the root cause. If they still diverge: the bug is in the stage logic itself, not in X.

| What you're testing | What you inject | Where you inject |
|---|---|---|
| Is the decode stage correct? | Post-loop latents from reference | Before decode |
| Is the denoise loop correct? | Pre-loop latents from reference | Before the loop |
| Is step N correct? | Post-step-(N-1) latents from reference | Before step N |

**Per-step accumulation tracing**: When injection confirms the loop is correct but you want to understand *how* a small initial difference compounds, capture `after_step_{i}` for every step and plot the max_diff curve. A healthy curve stays bounded; an exponential blowup in later steps points to an amplification mechanism (see Pitfall #13 in [pitfalls.md](pitfalls.md)).

## Debugging technique: Visual comparison via frame extraction

For video pipelines, numerical metrics alone can be misleading. Extract and view individual frames:

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

## Testing rules

1. **Never use reference code in the diffusers test path.** Each side must use only its own code.
2. **Never monkey-patch model internals in tests.** Do not replace `model.forward` or patch internal methods.
3. **Debugging instrumentation must be non-destructive.** Checkpoint captures for debugging are fine, but must not alter control flow or outputs.
4. **Prefer CPU/float32 for numerical comparison when practical.** Float32 avoids bfloat16 precision noise that obscures real bugs. But for large models (22B+), GPU/bfloat16 with `enable_model_cpu_offload()` is necessary -- use relaxed tolerances and cosine similarity as a secondary metric.
5. **Test both fresh conversion AND saved model.** Fresh catches conversion logic bugs; saved catches stale/corrupted weights from previous runs.
6. **Diff configs before debugging.** Before investigating any divergence, dump and compare all config values. A 30-second config diff prevents hours of debugging based on wrong assumptions.
7. **Never modify cached/downloaded model configs directly.** Don't edit files in `~/.cache/huggingface/`. Instead, save to a local directory or open a PR on the upstream repo.
8. **Compare ALL loop inputs in the encode test.** The preloop checkpoint must capture every single tensor the transformer forward() will receive.

## Comparison utilities

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

## Gotchas

See [pitfalls.md](pitfalls.md) for the full list of gotchas to watch for during parity testing.
