---
name: model-integration
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
3. **Standard vs modular** — **default to modular.** [Modular Diffusers](references/modular.md) is the preferred implementation for new pipelines; the standard `DiffusionPipeline` is still supported but no longer the default. We prefer modular especially for models that don't fit a fixed task-based structure (modality baked into the checkpoint) or that are actively evolving.
4. **Hub checkpoint** — ask where the diffusers weights will be hosted: in a separate repo, or in the same repo as the reference checkpoint (see **Hub checkpoint**).

Ask step 3 as an `AskUserQuestion`, with modular marked as the recommended default. Ask step 4 as one too, with no recommended option.

Once you have everything, **confirm the plan** with the user before implementing — state exactly what you'll do, e.g. "I'll integrate model X with pipeline Y based on your script, and verify the model matches the reference before considering it done."

Then work through the **Integration checklist** below

## Integration checklist

A pipeline in Diffusers (be it standard or modular) will have multiple components. These components can be models, schedulers, processors, etc.

- [ ] **Transformer model**
  - [ ] Implement the model with `from_pretrained` support (conventions: [models.md](references/models.md))
  - [ ] Convert weights (see **Weight / Checkpoint Conversion**)
  - [ ] Parity test against the reference (internal, not shipped — see **Model parity test**)
  - [ ] Register in the relevant `__init__.py` files (lazy imports)
  - [ ] Model-level tests (see **Testing**)
- [ ] **VAE** (if applicable) — reuse an existing `AutoencoderKL*` if possible; if a new one is needed, follow the same sub-steps as the transformer
- [ ] **Scheduler** — reuse an existing scheduler, or add a custom one
- [ ] **Pipeline**
  - [ ] Implement the pipeline — see [modular.md](references/modular.md) for modular pipeline, or [pipelines.md](references/pipelines.md) for standard pipeline
  - [ ] Add a LoRA mixin if applicable
  - [ ] Register in the relevant `__init__.py` files (lazy imports)
  - [ ] Pipeline-level tests (see **Testing**)
- [ ] **Hub checkpoint** — assemble the repo layout and its index (see **Hub checkpoint**)
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

**Match the reference's numerical logic.** Restructuring code to fit diffusers APIs (`ModelMixin`, `ConfigMixin`, blocks for modular, etc.) is expected, and required diffusers conventions (e.g. the attention pattern in [models.md](references/models.md)) take precedence. Beyond those, keep the actual computation as close to the reference as possible — don't reorder operations, change the math, or rename internals for aesthetics, even if it looks unclean. Small deviations make output mismatches very hard to track down.

## Weight / Checkpoint Conversion

Convert the original checkpoint into diffusers format with a standalone script under `scripts/` (e.g. `scripts/convert_<model>_to_diffusers.py`). The flow:

1. Map the original state-dict keys to the diffusers module names (renames + any tensor surgery — see patterns below).
2. Instantiate the diffusers model from its config and load the converted state dict.
3. `save_pretrained(...)` to a local path, then load it back with `from_pretrained` to confirm it round-trips.

All weights load through the standard paths — `from_pretrained`, or `from_single_file` (add `FromSingleFileMixin` + a weight-mapping) for an original-format single checkpoint. No custom `from_pretrained`, no manual runtime loading. See the loading rule in [models.md](references/models.md).

Common conversion patterns to watch for model-level components:
- Fused QKV weights that need splitting into separate Q, K, V
- Scale/shift ordering differences (reference stores `[shift, scale]`, diffusers expects `[scale, shift]`)
- Weight transpositions (linear stored as transposed conv, or vice versa)
- Interleaved head dimensions that need reshaping
- Bias terms absorbed into different layers

## Hub checkpoint

Conversion ends with a repo that users can load with `ModularPipeline.from_pretrained`. Decide its layout with the user before the conversion script writes anything — the script's output directory should already be that layout.

**Separate repo or the reference checkpoint's repo — the user's call.** In a separate repo the converted weights are laid out the standard way (`transformer/`, `vae/`, `text_encoder/`, ...). In the reference checkpoint's repo the converted components sit next to the reference ones, and the only practical difference is folder names: when the reference checkpoint already uses a name the diffusers layout needs (`transformer/`, `vae/`, ...), write the converted component to `<name>_diffusers/` and leave the reference folder untouched. [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) is the worked example: `transformer/` and `vae/` hold the reference weights, `transformer_diffusers/` and `vae_diffusers/` hold the converted ones, and `modular_model_index.json` points at the latter through each component's `subfolder`. Components with no name clash keep their plain folder name. This only works with `modular_model_index.json`, so the integration has to be modular if the weights go in the reference checkpoint's repo and a name clashes.

**Self-contained by default.** Every component's loading spec in the index points at this repo, including a component that is identical to another model's (a shared text encoder or VAE). Don't reference another repo just to save disk space. Identical Xet-backed files share one cached copy across repos (See [Share files across repositories](https://huggingface.co/docs/diffusers/main/en/installation#share-files-across-repositories)).

Check the result by loading the pipeline from the assembled repo (`ModularPipeline.from_pretrained(repo)` + `load_components()`) and reading its `modular_model_index.json`: every `pretrained_model_name_or_path` should be the repo itself.

## Testing

Two test layers must be added for any new pipeline: pipeline-level tests, and (if a new model is introduced) model-level tests. Conventions for both layers — file locations, tester mixins, dummy-component rules — live in [testing.md](references/testing.md); follow it when writing the tests.

## Model parity test

Confirm the diffusers implementation matches the reference. Test each component on **CPU/float32** with a strict tolerance (`max_diff < 1e-3`), comparing the **freshly converted** weights against the reference in a single script — both sides side by side, nothing saved to disk in between. See [pitfalls.md](references/pitfalls.md) for the common sources of numerical discrepancy.

This is an **internal verification tool for integration — it should not be shipped in the PR** (it imports the reference repo). The tests that ship with the PR are the model-level and pipeline-level tests in [testing.md](references/testing.md).

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
