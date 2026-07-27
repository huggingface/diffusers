# Testing

Test conventions for new models and pipelines: what a PR must ship, and what to check existing test files against.

Two test layers must be added for any new pipeline: pipeline-level tests, and (if a new model is introduced) model-level tests. Integration/slow tests and LoRA tests are **not** added in the initial PR — they come later, after discussion with maintainers.

## General rules (apply to all layers)

- Keep component sizes tiny so the suite runs fast — small `num_layers`, small hidden/attention dims, low resolution, few frames. Reference `tests/pipelines/wan/test_wan.py` (`get_dummy_components` and `get_dummy_inputs`) for the size scale to target.
- Build dummy components from the **real classes** at tiny config — a real VAE with tiny dims, a real tokenizer from an `hf-internal-testing/tiny-random-*` repo. Don't substitute a hand-rolled mock (a bare `nn.Module` with a `SimpleNamespace` config, a fake tokenizer) without a good reason: a mock is written by copying whatever the pipeline reads from the component today, so it can only confirm the pipeline against itself — the test stays green when the component renames a config field or the pipeline starts reading one the component doesn't have, and catching exactly that pipeline↔component contract is what a pipeline test is for. A good reason to stub: the component is impractical to instantiate and only its I/O matters to the pipeline (e.g. `DummyCosmosSafetyChecker` standing in for the huge Cosmos guardrail) — then make it a shared, purpose-built class honoring the real interface.
- The same applies to test doubles at the call level: don't monkeypatch a component method (e.g. the scheduler's `set_timesteps`) just to capture what the code under test passed to it — that only verifies the caller against itself, not against the real method's contract. Call the real component and assert on its resulting state.
- No LoRA tests in the initial PR (no `LoraTesterMixin`, no `tests/lora/test_lora_layers_<model>.py`).
- No integration / slow tests in the initial PR — don't add anything gated on `@slow` / `RUN_SLOW=1` yet.

## Pipeline-level tests 

### Standard pipelines

Follow the style introduced in [#14113](https://github.com/huggingface/diffusers/pull/14113), which moved the shared infrastructure into the `tests/pipelines/testing_utils/` package and split the old monolithic `unittest.TestCase` into a **config class + composable pytest mixins**. Reference: `tests/pipelines/flux/test_pipeline_flux.py`.

- Location: `tests/pipelines/<model>/test_pipeline_<model>.py` (one file per pipeline variant, e.g. T2V, I2V).
- **These are pytest-style, not `unittest`** — no `unittest.TestCase` subclassing, no `setUp`/`tearDown` (a `cleanup` fixture handles VRAM), and skips use `pytest.skip` / `@pytest.mark.skip`, never `@unittest.skip`. Fixtures like `tmp_path` and the cached `base_pipe_output` are injected into test methods as arguments.
- **Define one config class**, `<Pipeline>PipelineTesterConfig`, subclassing `BasePipelineTesterConfig` (from `..testing_utils`). It holds the whole testing contract and performs no assertions:
  - Set `pipeline_class`, `required_input_params_in_call_signature` (params that must appear in `__call__`'s signature), and `batch_input_params` (params that get batched). Use the canonical sets in `..pipeline_params` where one fits, or an inline `frozenset([...])`.
  - Implement `get_dummy_components(...)` — build every sub-module from the **real classes** at tiny config, each preceded by `torch.manual_seed(0)`.
  - Implement `get_dummy_inputs()` — **no `device` / `seed` arguments** (unlike the old style). Use `self.get_generator(0)` for the generator, keep sizes tiny, and set `output_type="pt"` so tests compare torch tensors directly with `assert_tensors_close` (no numpy round-trip). Remember `"pt"` images are `(batch, channels, height, width)`.
- **Compose the config with one mixin per concern**, one test class each, named `Test<Pipeline>...`. Add only the mixins that apply:
  - `PipelineTesterMixin` — core save/load, dict-vs-tuple equivalence, batching, dtype/device, callbacks. Put pipeline-specific tests as methods on this class.
  - `MemoryTesterMixin` — CPU offload, group offload, layerwise casting.
  - Cache mixins — `PyramidAttentionBroadcastTesterMixin`, `FasterCacheTesterMixin`, `FirstBlockCacheTesterMixin`, `TaylorSeerCacheTesterMixin`, `MagCacheTesterMixin`. Guidance-distilled models override the cache config (e.g. `FASTER_CACHE_CONFIG = {... "is_guidance_distilled": True}`). Don't introduce caching related tests in the first iteration. These tests are added on a case-by-case basis.
  - In the first pass, just add tests related to `PipelineTesterMixin` and `MemoryTesterMixin`.
- **IP-Adapter tests** live in their own class decorated with `@is_ip_adapter`, subclassing only the config (not `PipelineTesterMixin`).

### Modular pipelines

- Location: `tests/modular_pipelines/<model>/test_modular_pipeline_<model>.py` (one test class per blockset / pipeline variant).
- Subclass `ModularPipelineTesterMixin` (from `..test_modular_pipelines_common`) — it runs the pipeline end-to-end (call signature, batch consistency, float16, device placement) against a tiny checkpoint.
- Set `pipeline_class`, `pipeline_blocks_class`, `pretrained_model_name_or_path`, `params` / `batch_params`, and implement `get_dummy_inputs(seed=0)`. Set `expected_workflow_blocks` to pin the block name → class ordering per workflow.
- `pretrained_model_name_or_path` is a tiny repo with real components (tiny transformer, real scheduler / VAE / tokenizer configs). Develop against a personal repo; tiny repos ultimately live under `hf-internal-testing/` — not merge-blocking, a maintainer moves it before or after merge.
- **The tiny repo must mirror the real checkpoint's shape** — same index file type, same pipeline-level config keys, a scheduler configured like the real one. A fixture that doesn't look like the published repos tests a loading/config path no user will ever hit, while the path users *do* hit stays uncovered. If the model ships variants with different configs (base/distilled, different schedules), make one tiny repo and test class per variant — see the flux2 klein base/distilled split.
- **Bespoke tests go on the tester class as methods**, not as module-level functions — the mixin is pytest-style, so fixtures (`tmp_path`, `pytest.raises`, parametrize) all work in methods.
- **Test a block's behavior by running it as a pipeline** — `init_pipeline()` → `load_components()` → call it and assert on outputs (see "Running a modular pipeline" in [modular.md](modular.md)). Config-dependent behavior: flip the value with `update_components(...)` and compare real outputs across the two runs. Input validation: `pytest.raises` around a normal `pipe(...)` call. Don't call `block(components, state)` directly or hand-build a `PipelineState`, and don't assert on declared specs (`inputs` / `intermediate_outputs` name lists) — declarations aren't behavior, and `expected_workflow_blocks` already pins the structure.
- Reference: `tests/modular_pipelines/flux2/test_modular_pipeline_flux2_klein.py` (plus `..._klein_base.py` for the base/distilled variant split).

## Model-level tests

Only required if the pipeline introduces a new model class (transformer, VAE, etc.). Don't write these by hand — generate them (example command below):

```bash
python utils/generate_model_tests.py src/diffusers/models/transformers/transformer_<model>.py
```

- Run with **no `--include` flags** initially. The generator auto-detects mixins/attributes and emits the always-on testers (`ModelTesterMixin`, `MemoryTesterMixin`, `TorchCompileTesterMixin`, plus `AttentionTesterMixin` / `ContextParallelTesterMixin` / `TrainingTesterMixin` as applicable). Optional testers (quantization, caching, single-file, IP adapter, etc.) are added later, after maintainer discussion.
- The generator writes to `tests/models/transformers/test_models_transformer_<model>.py` (or the matching `unets/` / `autoencoders/` subdir).
- Fill in the `TODO`s in the generated `<Model>TesterConfig`: `pretrained_model_name_or_path`, `get_init_dict()` (tiny config), `get_dummy_inputs()`, `input_shape`, `output_shape`. Keep init dims small for speed.
- Do **not** add `LoraTesterMixin` at the start, even if the model subclasses `PeftAdapterMixin` — strip it from the generated file for the initial PR.
- Reference: `tests/models/transformers/test_models_transformer_flux.py`.
