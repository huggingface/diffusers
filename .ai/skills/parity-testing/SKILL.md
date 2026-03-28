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
3. **Test directory**: Ask the user if they have a preferred directory for parity test scripts and artifacts. If not, create `parity-tests/` at the repo root.
4. **Lab book**: Ask the user if they want to maintain a `lab_book.md` in the test directory to track findings, fixes, and experiment results across sessions. This is especially useful for multi-session debugging where context gets lost.

When invoked from the `model-integration` skill, you already have context: the reference script comes from step 2 of setup, and the diffusers script is the one you just wrote. You just need to make sure both scripts are runnable and use the same inputs/seed/params.

## Phase 1: CPU/float32 parity (always run)

### Component parity — test as you build

Test each component before assembling the pipeline. This is the foundation -- if individual pieces are wrong, the pipeline can't be right. Each component in isolation, strict max_diff < 1e-3.

Test freshly converted checkpoints and saved checkpoints.
- **Fresh**: convert from checkpoint weights, compare against reference (catches conversion bugs)
- **Saved**: load from saved model on disk, compare against reference (catches stale saves)

Keep component test scripts around -- you will need to re-run them during pipeline debugging with different inputs or config values.

**Write a model interface mapping** as you test each component. This documents every input difference between reference and diffusers models — format, dtype, shape, who computes what. Save it in the test directory (e.g., `parity-tests/model_interface_mapping.md`). This is critical: during pipeline testing, you MUST reference this mapping to verify the pipeline passes inputs in the correct format. Without it, you'll waste time rediscovering differences you already found.

Example mapping (from LTX-2.3):
```markdown
| Input | Reference | Diffusers | Notes |
|---|---|---|---|
| timestep | per-token bf16 sigma, scaled by 1000 internally | passed as sigma*1000 | shape (B,S) not (B,) |
| sigma (prompt_adaln) | raw f32 sigma, scaled internally | passed as sigma*1000 in f32 | NOT bf16 |
| positions/coords | computed inside model preprocessor | passed as kwarg video_coords | cast to model dtype |
| cross-attn timestep | always cross_modality.sigma | always audio_sigma | not conditional |
| encoder_attention_mask | None (no mask) | None or all-ones | all-ones triggers different SDPA kernel |
| RoPE | computed in model dtype (no upcast) | must match — no float32 upcast | cos/sin cast to input dtype |
| output format | X0Model returns x0 | transformer returns velocity | v→x0: (sample - vel * sigma) |
| audio output | .squeeze(0).float() | must match | (2,N) float32 not (1,2,N) bf16 |
```

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

### Pipeline stage tests — encode, decode, then denoise

Use the capture-inject checkpoint method (see [checkpoint-mechanism.md](checkpoint-mechanism.md)) to test each pipeline stage independently. This methodology is the same for both CPU/float32 and GPU/bf16.

Before writing pipeline tests, **review the model interface mapping** from the component test phase and verify them. The mapping tells you which differences between the two models are expected (e.g., reference expects raw sigma but diffusers expects sigma*1000). Without it, you'll waste time investigating differences that are by design, not bugs.

First, **match noise generation**: the way initial noise/latents are constructed (seed handling, generator, randn call order) often differs between the two scripts. If the noise doesn't match, nothing downstream will match.

**Stage test order:**

- **`encode`** (test first): Stop both pipelines at `"preloop"`. Compare **every single variable** that will be consumed by the denoising loop -- not just latents and sigmas, but also prompt embeddings, attention masks, positional coordinates, connector outputs, and any conditioning inputs.
- **`decode`** (test second): Run the reference pipeline fully -- checkpoint the post-loop latents AND let it finish to get the **final output**. Feed those same post-loop latents through the diffusers decode path. Compare the **final output format** -- not raw tensors, but what the user actually gets:
  - **Image**: compare PIL.Image pixels
  - **Video**: compare through the pipeline's export function (e.g. `encode_video`)
  - **Video+Audio**: compare video frames AND audio waveform through `encode_video`
  - This catches postprocessing bugs like float→uint8 rounding, audio format, and codec settings.
- **`denoise`** (test last): Run both pipelines with realistic `num_steps` (e.g. 30) so the scheduler computes correct sigmas/timesteps. For float32, stop after 2 loop iterations using `after_step_1` (don't set `num_steps=2` -- that produces unrealistic sigma schedules). For bf16, run ALL steps (see Phase 2).

Start with coarse checkpoints (`after_step_{i}` — just the denoised latents at each step). If a step diverges, place finer checkpoints within that step (e.g. before/after model call, after CFG, after scheduler step). If the divergence is inside the model forward call, use PyTorch forward hooks (`register_forward_hook`) to capture intermediate outputs from sub-modules (e.g., attention output, feed-forward output) and compare them between the two models to find the first diverging operation.

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

### E2E visual — once stages pass

Both pipelines generate independently with identical seeds/params. Save outputs and compare visually. If outputs look identical, Phase 1 is done.

If CPU/float32 stage tests all pass and E2E outputs are identical → Phase 1 is done, move on.

If E2E outputs are NOT identical despite stage tests passing, **ask the user**: "CPU/float32 parity passes at the stage level but E2E output differs. The output in bf16/GPU may look slightly different from the reference due to precision casting, but the quality should be the same. Do you want to just vibe-check the output quality, or do you need 1:1 identical output with the reference in bf16?"

- If the user says quality looks fine → **done**.
- If the user needs 1:1 identical output in bf16 → Phase 2.

## Phase 2: GPU/bf16 parity (optional — only if user needs 1:1 output)

If CPU/float32 passes, the algorithm is correct. bf16 differences are from precision casting (e.g., float32 vs bf16 in RoPE, CFG arithmetic order, scheduler intermediates), not logic bugs. These can make the output look slightly different from the reference even though the quality is identical. Phase 2 eliminates these casting differences so the diffusers output is **bit-identical** to the reference in bf16.

Phase 2 uses the **exact same stage test methodology** as Phase 1 (encode → decode → denoise with progressive checkpoint refinement), with two differences:

1. **dtype=bf16, device=GPU** instead of float32/CPU
2. **Run the FULL denoising loop** (all steps, not just 2) — bf16 casting differences accumulate over steps and may only manifest after many iterations

See [pitfalls.md](pitfalls.md) #19-#27 for the catalog of bf16-specific gotchas.

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
9. **Don't contaminate test paths.** Each side (reference, diffusers) must use only its own code to generate outputs. For COMPARISON, save both outputs through the SAME function (so codec/format differences don't create false diffs). Example: don't use the reference's `encode_video` for one side and diffusers' for the other.
10. **Re-test standalone model through the actual pipeline if divergence points to the model.** If pipeline stage tests show the divergence is at the model output (e.g., `cond_x0` differs despite identical inputs), re-run the model comparison using capture-inject with real pipeline-generated inputs. Standalone model tests use manually constructed kwargs which may have wrong config values, dtypes, or shapes — the pipeline generates the real ones.

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

## Example scripts

- [examples/test_component_parity_cpu.py](examples/test_component_parity_cpu.py) — Template for CPU/float32 component parity test
- [examples/test_e2e_bf16_parity.py](examples/test_e2e_bf16_parity.py) — Template for GPU/bf16 E2E parity test with capture-inject

## Gotchas

See [pitfalls.md](pitfalls.md) for the full list of gotchas to watch for during parity testing.
