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

### Test setup

- Slow tests gated with `@slow` and `RUN_SLOW=1`
- All model-level tests must use the `BaseModelTesterConfig`, `ModelTesterMixin`, `MemoryTesterMixin`, `AttentionTesterMixin`, `LoraTesterMixin`, and `TrainingTesterMixin` classes initially to write the tests. Any additional tests should be added after discussions with the maintainers. Use `tests/models/transformers/test_models_transformer_flux.py` as a reference.

---

## Modular Pipeline Conversion

See [modular-conversion.md](modular-conversion.md) for the full guide on converting standard pipelines to modular format, including block types, build order, guider abstraction, and conversion checklist.

---

## Weight Conversion Tips

<!-- TODO: Add concrete examples as we encounter them. Common patterns to watch for:
  - Fused QKV weights that need splitting into separate Q, K, V
  - Scale/shift ordering differences (reference stores [shift, scale], diffusers expects [scale, shift])
  - Weight transpositions (linear stored as transposed conv, or vice versa)
  - Interleaved head dimensions that need reshaping
  - Bias terms absorbed into different layers
  Add each with a before/after code snippet showing the conversion. -->
