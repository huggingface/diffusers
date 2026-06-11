---
name: integrating-models
description: >
  Use when adding a new model or pipeline to diffusers, setting up file
  structure for a new model, converting a pipeline to modular format, or
  converting weights for a new version of an already-supported model.
---

## Goal

Integrate a new model into diffusers end-to-end, to full numerical parity with the reference implementation — one workflow at a time.

## Setup — gather before starting

Before writing any code, gather info in this order:

1. **Reference repo** — ask for the github link. If they've already set it up locally, ask for the path. Otherwise, ask what setup steps are needed (install deps, download checkpoints, set env vars, etc.) and run through them before proceeding.
2. **Inference script** — ask for a runnable end-to-end script for a basic workflow first (e.g. T2V). Then ask what other workflows they want to support (I2V, V2V, etc.) and agree on the full implementation order together.
3. **Standard vs modular** — **default to modular.** [Modular Diffusers](../../modular.md) is the preferred implementation for new pipelines; the standard `DiffusionPipeline` is still supported but no longer the default. We prefer modular especially for models that don't fit a fixed task-based structure (modality baked into the checkpoint) or that are actively evolving.

Ask step 3 as an `AskUserQuestion`, with modular marked as the recommended default.

Once you have everything, **confirm the plan** with the user before implementing — state exactly what you'll do, e.g. "I'll integrate model X with pipeline Y based on your script, and verify the model matches the reference before considering it done."

Then work through the **Integration checklist** below — it covers both standard and modular; only the pipeline step differs.

## Integration checklist

- [ ] **Transformer model**
  - [ ] Implement the model with `from_pretrained` support (conventions: [models.md](../../models.md))
  - [ ] Convert weights (see **Weight / Checkpoint Conversion**)
  - [ ] Parity test against the reference (internal, not shipped — see **Model parity test**)
  - [ ] Register in the relevant `__init__.py` files (lazy imports)
  - [ ] Model-level tests (see **Testing**)
- [ ] **VAE** (if applicable) — reuse an existing `AutoencoderKL*` if possible; if a new one is needed, follow the same sub-steps as the transformer
- [ ] **Scheduler** — reuse an existing scheduler, or add a custom one
- [ ] **Pipeline**
  - [ ] Implement the pipeline — see [modular.md](../../modular.md) for modular pipeline, or [pipelines.md](../../pipelines.md) for standard pipeline
  - [ ] Add a LoRA mixin if applicable
  - [ ] Register in the relevant `__init__.py` files (lazy imports)
  - [ ] Pipeline-level tests (see **Testing**)
- [ ] **Docs** — see **File structure**
- [ ] **Style** — `make style` and `make quality`

## File structure

A new model PR roughly lands these files (the contents of `pipelines/<model>/` and `modular_pipelines/<model>/` live in their guides):

```
src/diffusers/
  models/transformers/transformer_<model>.py   # the model (or models/autoencoders/, models/unets/)
  schedulers/scheduling_<model>.py              # only if a custom scheduler is needed
  loaders/lora_pipeline.py                      # LoRA mixin — add to the existing file
  pipelines/<model>/                            # standard pipeline — see pipelines.md
  modular_pipelines/<model>/                    # modular pipeline — see modular.md
tests/
  models/transformers/test_models_transformer_<model>.py
  pipelines/<model>/test_<model>.py
docs/source/en/
  _toctree.yml                                  # register the new pages in the docs index
  api/models/<model>.md
  api/pipelines/<model>.md
```

## Model integration specific rules

**Match the reference's numerical logic.** Restructuring code to fit diffusers APIs (`ModelMixin`, `ConfigMixin`, blocks for modular, etc.) is expected, and required diffusers conventions (e.g. the attention pattern in [models.md](../../models.md)) take precedence. Beyond those, keep the actual computation as close to the reference as possible — don't reorder operations, change the math, or rename internals for aesthetics, even if it looks unclean. Small deviations make output mismatches very hard to track down.

## Weight / Checkpoint Conversion

Convert the original checkpoint into diffusers format with a standalone script under `scripts/` (e.g. `scripts/convert_<model>_to_diffusers.py`). The flow:

1. Map the original state-dict keys to the diffusers module names (renames + any tensor surgery — see patterns below).
2. Instantiate the diffusers model from its config and load the converted state dict.
3. `save_pretrained(...)` to a **persistent** path (never `/tmp/` — see [pitfalls.md](pitfalls.md) #10), then load it back with `from_pretrained` to confirm it round-trips.

All weights load through the standard paths — `from_pretrained`, or `from_single_file` (add `FromSingleFileMixin` + a weight-mapping) for an original-format single checkpoint. No custom `from_pretrained`, no manual runtime loading. See the loading rule in [models.md](../../models.md).

Common conversion patterns to watch for:
- Fused QKV weights that need splitting into separate Q, K, V
- Scale/shift ordering differences (reference stores `[shift, scale]`, diffusers expects `[scale, shift]`)
- Weight transpositions (linear stored as transposed conv, or vice versa)
- Interleaved head dimensions that need reshaping
- Bias terms absorbed into different layers

## Testing

Two test layers must be added for any new pipeline: pipeline-level tests, and (if a new model is introduced) model-level tests. Integration/slow tests and LoRA tests are **not** added in the initial PR — they come later, after discussion with maintainers.

**General rules (apply to both layers):**
- Keep component sizes tiny so the suite runs fast — small `num_layers`, small hidden/attention dims, low resolution, few frames. Reference `tests/pipelines/wan/test_wan.py` (`get_dummy_components` and `get_dummy_inputs`) for the size scale to target.
- No LoRA tests in the initial PR (no `LoraTesterMixin`, no `tests/lora/test_lora_layers_<model>.py`).
- No integration / slow tests in the initial PR — don't add anything gated on `@slow` / `RUN_SLOW=1` yet.

### Pipeline-level tests

- Location: `tests/pipelines/<model>/test_<model>.py` (one file per pipeline variant, e.g. T2V, I2V).
- Subclass both `PipelineTesterMixin` (from `..test_pipelines_common`) and `unittest.TestCase`.
- Set `pipeline_class`, `params`, `batch_params`, `image_params` from `..pipeline_params`, and any `required_optional_params` / capability flags (`test_xformers_attention`, `supports_dduf`, etc.) that apply.
- Implement `get_dummy_components()` (build all sub-modules with tiny configs and a fixed `torch.manual_seed(0)` before each) and `get_dummy_inputs(device, seed=0)`.
- Skip any inherited tests that don't apply with `@unittest.skip("Test not supported")` rather than deleting them.
- Reference: `tests/pipelines/wan/test_wan.py`.

### Model-level tests

Only required if the pipeline introduces a new model class (transformer, VAE, etc.). Don't write these by hand — generate them (example command below):

```bash
python utils/generate_model_tests.py src/diffusers/models/transformers/transformer_<model>.py
```

- Run with **no `--include` flags** initially. The generator auto-detects mixins/attributes and emits the always-on testers (`ModelTesterMixin`, `MemoryTesterMixin`, `TorchCompileTesterMixin`, plus `AttentionTesterMixin` / `ContextParallelTesterMixin` / `TrainingTesterMixin` as applicable). Optional testers (quantization, caching, single-file, IP adapter, etc.) are added later, after maintainer discussion.
- The generator writes to `tests/models/transformers/test_models_transformer_<model>.py` (or the matching `unets/` / `autoencoders/` subdir).
- Fill in the `TODO`s in the generated `<Model>TesterConfig`: `pretrained_model_name_or_path`, `get_init_dict()` (tiny config), `get_dummy_inputs()`, `input_shape`, `output_shape`. Keep init dims small for speed.
- Do **not** add `LoraTesterMixin` at the start, even if the model subclasses `PeftAdapterMixin` — strip it from the generated file for the initial PR.
- Reference: `tests/models/transformers/test_models_transformer_flux.py`.

## Model parity test

Confirm the diffusers implementation matches the reference. Test each component on **CPU/float32** with a strict tolerance (`max_diff < 1e-3`), comparing the **freshly converted** weights against the reference in a single script — both sides side by side, nothing saved to disk in between. See [pitfalls.md](pitfalls.md) for the common sources of numerical discrepancy.

This is an **internal verification tool for integration — it should not be shipped in the PR** (it imports the reference repo). The tests that ship with the PR are the model-level and pipeline-level tests in **Testing**.

The example below is schematic (placeholder names). `ReferenceModel` is the component **imported from the original repo**, and `convert_my_component` is **the same conversion function you wrote for the conversion script for the component**. You should make sure both load the *same* checkpoint weights and run the *same* input, so any difference is a conversion or implementation bug — not a difference in inputs.

```python
@torch.inference_mode()
def test_my_component():
    # deterministic input — use the same shape & dtype the real model receives at this stage
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(1, 16, 32, 32, generator=gen, dtype=torch.float32)  # adjust to the real input shape

    original_state_dict = load_original_weights(...)  # the original checkpoint — both sides load these same weights

    # reference: the original repo's implementation (load one model at a time to fit in CPU RAM)
    ref_model = ReferenceModel(config)                # ReferenceModel: imported from the original repo
    ref_model.load_state_dict(original_state_dict, strict=True)
    ref_model = ref_model.float().eval()
    ref_out = ref_model(x).clone()                    # clone before freeing the model
    del ref_model

    # diffusers: convert those same weights with your conversion-script function, then run
    diff_model = convert_my_component(original_state_dict)  # convert_my_component: the fn from convert_<model>_to_diffusers.py
    diff_model = diff_model.float().eval()
    diff_out = diff_model(x)

    max_diff = (ref_out - diff_out).abs().max().item()
    assert max_diff < 1e-3, f"FAIL: max_diff={max_diff:.2e}"
```
