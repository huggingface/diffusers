---
name: integrating-models
description: >
  Use when adding a new model or pipeline to diffusers, setting up file
  structure for a new model, converting a pipeline to modular format, or
  converting weights for a new version of an already-supported model.
---

## Goal

Integrate a new model into diffusers end-to-end. The overall flow:

1. **Gather info** — ask the user for the reference repo, setup guide, a runnable inference script, and other objectives such as standard vs modular.
2. **Confirm the plan** — once you have everything, tell the user exactly what you'll do: e.g. "I'll integrate model X with pipeline Y into diffusers based on your script. I'll run parity tests (model-level and pipeline-level) using the `parity-testing` skill to verify numerical correctness against the reference."
3. **Implement** — write the diffusers code (model, pipeline, scheduler if needed), convert weights, register in `__init__.py`.
4. **Parity test** — use the `parity-testing` skill to verify component and e2e parity against the reference implementation.
5. **Deliver a unit test** — provide a self-contained test script that runs the diffusers implementation, checks numerical output (np allclose), and saves an image/video for visual verification. This is what the user runs to confirm everything works.

Work one workflow at a time — get it to full parity before moving on.

## Setup — gather before starting

Before writing any code, gather info in this order:

1. **Reference repo** — ask for the github link. If they've already set it up locally, ask for the path. Otherwise, ask what setup steps are needed (install deps, download checkpoints, set env vars, etc.) and run through them before proceeding.
2. **Inference script** — ask for a runnable end-to-end script for a basic workflow first (e.g. T2V). Then ask what other workflows they want to support (I2V, V2V, etc.) and agree on the full implementation order together.
3. **Standard vs modular** — standard pipelines, modular, or both?

Use `AskUserQuestion` with structured choices for step 3 when the options are known.

## Standard Pipeline Integration

### File structure for a new model

```
src/diffusers/
  models/transformers/transformer_<model>.py     # The core model
  schedulers/scheduling_<model>.py               # If model needs a custom scheduler
  pipelines/<model>/
    __init__.py
    pipeline_<model>.py                          # Main pipeline
    pipeline_<model>_<variant>.py                # Variant pipelines (e.g. pyramid, distilled)
    pipeline_output.py                           # Output dataclass
  loaders/lora_pipeline.py                       # LoRA mixin (add to existing file)

tests/
  models/transformers/test_models_transformer_<model>.py
  pipelines/<model>/test_<model>.py
  lora/test_lora_layers_<model>.py

docs/source/en/api/
  pipelines/<model>.md
  models/<model>_transformer3d.md                # or appropriate name
```

### Integration checklist

- [ ] Implement transformer model with `from_pretrained` support
- [ ] Implement or reuse scheduler
- [ ] Implement pipeline(s) with `__call__` method
- [ ] Add LoRA support if applicable
- [ ] Register all classes in `__init__.py` files (lazy imports)
- [ ] Write unit tests (model, pipeline, LoRA)
- [ ] Write docs
- [ ] Run `make style` and `make quality`
- [ ] Test parity with reference implementation (see `parity-testing` skill)

### Model conventions, attention pattern, and implementation rules

See [../../models.md](../../models.md) for the attention pattern, implementation rules, common conventions, dependencies, and gotchas. These apply to all model work.

### Model integration specific rules

**Don't combine structural changes with behavioral changes.** Restructuring code to fit diffusers APIs (ModelMixin, ConfigMixin, etc.) is unavoidable. But don't also "improve" the algorithm, refactor computation order, or rename internal variables for aesthetics. Keep numerical logic as close to the reference as possible, even if it looks unclean. For standard → modular, this is stricter: copy loop logic verbatim and only restructure into blocks. Clean up in a separate commit after parity is confirmed.

### Testing

Two test layers must be added for any new pipeline: pipeline-level tests, and (if a new model is introduced) model-level tests. Integration/slow tests and LoRA tests are **not** added in the initial PR — they come later, after discussion with maintainers.

**General rules (apply to both layers):**
- Keep component sizes tiny so the suite runs fast — small `num_layers`, small hidden/attention dims, low resolution, few frames. Reference `tests/pipelines/wan/test_wan.py` (`get_dummy_components` and `get_dummy_inputs`) for the size scale to target.
- No LoRA tests in the initial PR (no `LoraTesterMixin`, no `tests/lora/test_lora_layers_<model>.py`).
- No integration / slow tests in the initial PR — don't add anything gated on `@slow` / `RUN_SLOW=1` yet.

#### Pipeline-level tests

- Location: `tests/pipelines/<model>/test_<model>.py` (one file per pipeline variant, e.g. T2V, I2V).
- Subclass both `PipelineTesterMixin` (from `..test_pipelines_common`) and `unittest.TestCase`.
- Set `pipeline_class`, `params`, `batch_params`, `image_params` from `..pipeline_params`, and any `required_optional_params` / capability flags (`test_xformers_attention`, `supports_dduf`, etc.) that apply.
- Implement `get_dummy_components()` (build all sub-modules with tiny configs and a fixed `torch.manual_seed(0)` before each) and `get_dummy_inputs(device, seed=0)`.
- Skip any inherited tests that don't apply with `@unittest.skip("Test not supported")` rather than deleting them.
- Reference: `tests/pipelines/wan/test_wan.py`.

#### Model-level tests

Only required if the pipeline introduces a new model class (transformer, VAE, etc.). Don't write these by hand — generate them (example command below):

```bash
python utils/generate_model_tests.py src/diffusers/models/transformers/transformer_<model>.py
```

- Run with **no `--include` flags** initially. The generator auto-detects mixins/attributes and emits the always-on testers (`ModelTesterMixin`, `MemoryTesterMixin`, `TorchCompileTesterMixin`, plus `AttentionTesterMixin` / `ContextParallelTesterMixin` / `TrainingTesterMixin` as applicable). Optional testers (quantization, caching, single-file, IP adapter, etc.) are added later, after maintainer discussion.
- The generator writes to `tests/models/transformers/test_models_transformer_<model>.py` (or the matching `unets/` / `autoencoders/` subdir).
- Fill in the `TODO`s in the generated `<Model>TesterConfig`: `pretrained_model_name_or_path`, `get_init_dict()` (tiny config), `get_dummy_inputs()`, `input_shape`, `output_shape`. Keep init dims small for speed.
- Do **not** add `LoraTesterMixin` at the start, even if the model subclasses `PeftAdapterMixin` — strip it from the generated file for the initial PR.
- Reference: `tests/models/transformers/test_models_transformer_flux.py`.

---

## Modular Pipeline Conversion

See [modular.md](../../modular.md) for the full guide on modular pipeline conventions, block types, build order, guider abstraction, gotchas, and conversion checklist.

---

## Weight Conversion Tips

<!-- TODO: Add concrete examples as we encounter them. Common patterns to watch for:
  - Fused QKV weights that need splitting into separate Q, K, V
  - Scale/shift ordering differences (reference stores [shift, scale], diffusers expects [scale, shift])
  - Weight transpositions (linear stored as transposed conv, or vice versa)
  - Interleaved head dimensions that need reshaping
  - Bias terms absorbed into different layers
  Add each with a before/after code snippet showing the conversion. -->
