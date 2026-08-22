# Testing

Test conventions for new models and pipelines: what a PR must ship, and what to check existing test files against.

Two test layers must be added for any new pipeline: pipeline-level tests, and (if a new model is introduced) model-level tests. Integration/slow tests and LoRA tests are **not** added in the initial PR — they come later, after discussion with maintainers.

## General rules (apply to all layers)

- Keep component sizes tiny so the suite runs fast — small `num_layers`, small hidden/attention dims, low resolution, few frames. Reference `tests/pipelines/wan/test_wan.py` (`get_dummy_components` and `get_dummy_inputs`) for the size scale to target.
- Build dummy components from the **real classes** at tiny config — a real VAE with tiny dims, a real tokenizer from an `hf-internal-testing/tiny-random-*` repo. Don't substitute a hand-rolled mock (a bare `nn.Module` with a `SimpleNamespace` config, a fake tokenizer) without a good reason: a mock is written by copying whatever the pipeline reads from the component today, so it can only confirm the pipeline against itself — the test stays green when the component renames a config field or the pipeline starts reading one the component doesn't have, and catching exactly that pipeline↔component contract is what a pipeline test is for. A good reason to stub: the component is impractical to instantiate and only its I/O matters to the pipeline (e.g. `DummyCosmosSafetyChecker` standing in for the huge Cosmos guardrail) — then make it a shared, purpose-built class honoring the real interface.
- The same applies to test doubles at the call level: don't monkeypatch a component method (e.g. the scheduler's `set_timesteps`) just to capture what the code under test passed to it — that only verifies the caller against itself, not against the real method's contract. Call the real component and assert on its resulting state.
- No LoRA tests in the initial PR — don't compose the LoRA tester mixins into the pipeline or model test file (see [LoRA tests](#lora-tests)), and don't add a `tests/lora/test_lora_layers_<model>.py`.
- No integration / slow tests in the initial PR — don't add anything gated on `@slow` / `RUN_SLOW=1` yet.

## Pipeline-level tests 

### Standard pipelines

Follow the style introduced in [#14113](https://github.com/huggingface/diffusers/pull/14113), which moved the shared infrastructure into the `tests/pipelines/testing_utils/` package and split the old monolithic `unittest.TestCase` into a **config class + composable pytest mixins**. Reference: `tests/pipelines/flux/test_pipeline_flux.py`.

- Location: `tests/pipelines/<model>/test_pipeline_<model>.py` (one file per pipeline variant, e.g. T2V, I2V).
- **These are pytest-style, not `unittest`** — no `unittest.TestCase` subclassing, no `setUp`/`tearDown` (a `cleanup` fixture handles VRAM), and skips use `pytest.skip` / `@pytest.mark.skip`, never `@unittest.skip`. Fixtures like `tmp_path` and the cached `base_pipe_output` are injected into test methods as arguments.
- **Define one config class**, `<Pipeline>PipelineTesterConfig`, subclassing `BasePipelineTesterConfig` (from `..testing_utils`). It holds the whole testing contract and performs no assertions:
  - Set `pipeline_class`, `required_input_params_in_call_signature` (params that must appear in `__call__`'s signature), and `batch_input_params` (params that get batched). Use the canonical sets in `..pipeline_params` where one fits, or an inline `frozenset([...])`.
  - Set `output_shape` — the per-sample output shape for `get_dummy_inputs()`, i.e. `(channels, height, width)` for an image pipeline and `(num_frames, channels, height, width)` for a video one. Assert against `self.output_shape` in pipeline-specific tests instead of repeating the literal.
  - Implement `get_dummy_components(...)` — build every sub-module from the **real classes** at tiny config, each preceded by `torch.manual_seed(0)`.
  - Implement `get_dummy_inputs()` — **no `device` / `seed` arguments** (unlike the old style). Use `self.get_generator(0)` for the generator, keep sizes tiny, and set `output_type="pt"` so tests compare torch tensors directly with `assert_tensors_close` (no numpy round-trip). Remember `"pt"` images are `(batch, channels, height, width)`.
- **Compose the config with one mixin per concern**, one test class each, named `Test<Pipeline>...`. Add only the mixins that apply:
  - `PipelineTesterMixin` — core save/load, dict-vs-tuple equivalence, batching, dtype/device, callbacks. Put pipeline-specific tests as methods on this class.
  - `MemoryTesterMixin` — CPU offload, group offload, layerwise casting.
  - Cache mixins — `PyramidAttentionBroadcastTesterMixin`, `FasterCacheTesterMixin`, `FirstBlockCacheTesterMixin`, `TaylorSeerCacheTesterMixin`, `MagCacheTesterMixin`. Guidance-distilled models override the cache config (e.g. `FASTER_CACHE_CONFIG = {... "is_guidance_distilled": True}`). Don't introduce caching related tests in the first iteration. These tests are added on a case-by-case basis.
  - In the first pass, just add tests related to `PipelineTesterMixin` and `MemoryTesterMixin`.
- **Declare a component that can't be offloaded — don't hand-write a skip.** Leaf-level offloading hooks only the supported leaf types (`nn.Linear`, `nn.Conv*`, `nn.Embedding` — see `_GO_LC_SUPPORTED_PYTORCH_LAYERS` in `src/diffusers/hooks/_common.py`) and onloads each on its own `forward`, so any code that reads a leaf's `.weight` instead of calling the leaf bypasses that leaf's hook and computes against offloaded weights. Which fix applies depends on who owns the component. For a diffusers model, set `_supports_group_offloading = False` on the `ModelMixin` subclass (as `HunyuanDiT2DModel` does) — both offload mixins honor the flag and skip themselves, so the gap is declared on the model instead of buried in a test file. For a third-party component you can't annotate, such as a `transformers` encoder, list it in `group_offloading_leaf_level_exclude_modules` on the config class (the attribute is also on the old-style `PipelineTesterMixin` in `tests/pipelines/test_pipelines_common.py`); `enable_group_offload` keeps excluded components on the accelerator, so every other component is still covered — where a hand-written skip on `test_pipeline_level_group_offloading_inference` would drop offload coverage for the whole pipeline, including the VAE, which the component-scoped `test_group_offloading_inference` deliberately excludes. Block-level offloading is usually unaffected, hence the level in the name — a component that fails at both levels does need a skip.
  - `torch.nn.MultiheadAttention` is the common instance: it passes `self.out_proj.weight` straight to `torch.nn.functional.multi_head_attention_forward` instead of calling `self.out_proj`, so the hook on `out_proj` never fires. `SiglipVisionModel`'s attention pooling head wraps one — see `tests/pipelines/hunyuan_video/test_hunyuan_video_framepack.py`, whose `image_encoder` is excluded for this reason.
  - `HunyuanDiTAttentionPool` (`src/diffusers/models/embeddings.py`) shows the same failure without an MHA module: a plain `nn.Module` that hands its `q_proj` / `k_proj` / `v_proj` / `c_proj` weights to `torch.nn.functional.multi_head_attention_forward`, so all four projections stay offloaded rather than just one. `HunyuanDiT2DModel` opts out of group offloading entirely with `_supports_group_offloading = False`.
  - Before adding a skip or an exclusion, confirm the failure still reproduces — several existing skips are stale, having outlived the upstream cause.
- **IP-Adapter tests** live in their own class decorated with `@is_ip_adapter`, subclassing only the config (not `PipelineTesterMixin`).

#### LoRA tests

Since [#14268](https://github.com/huggingface/diffusers/pull/14268), a standard pipeline's LoRA tests live **next to its pipeline tests** — another mixin composed with the same `<Pipeline>PipelineTesterConfig` — not in `tests/lora/test_lora_layers_<model>.py`. Reference: `TestFluxPipelineLoRA` / `TestFluxPipelineLoRAMemory` in `tests/pipelines/flux/test_pipeline_flux.py`.

- Mixins live in `tests/pipelines/testing_utils/lora.py` and are exported from `..testing_utils`. One test class each, named `Test<Pipeline>LoRA...`:
  - `LoraTesterMixin` — adapter attach/detach, LoRA scale and attention-kwargs, fuse/unfuse, multi-adapter (set/delete/weight), save/load round-trips, adapter metadata. Runs on CPU.
  - `LoraMemoryTesterMixin` — LoRA × memory optimizations (group offload, model CPU offload, deleting adapters while offloaded). Accelerator-only.
  - `UNetLoraTesterMixin` — per-block scale tests; UNet pipelines only.
- **Give each mixin its own test class.** They are marked `@is_lora`, and a mark applies to every test in the class that inherits it — mixing one into `Test<Pipeline>` would mark those tests as LoRA tests too.
- Run them with `pytest tests/pipelines/ -m "lora"` (what CI does); `pytest -m "not lora"` skips them.
- **Nothing LoRA-specific goes on the config class.** The mixins read the same contract as every other mixin — `pipeline_class`, `get_dummy_components()`, `get_dummy_inputs()` with `output_type="pt"` — and self-skip when `pipeline_class` isn't a `LoraBaseMixin` subclass.
- Components to adapt are derived from `pipeline_class._lora_loadable_modules`. Override `denoiser_target_modules` on the test class only when the denoiser's attention modules aren't named `to_q` / `to_k` / `to_v` / `to_out.0`. A text encoder architecture that isn't registered yet needs an entry in `TEXT_ENCODER_TARGET_MODULES` in `tests/pipelines/testing_utils/lora.py` — not a per-class override.
- **Pipeline-specific LoRA tests are methods on the `Test<Pipeline>LoRA` class**, written against the shared helpers: `self.get_pipeline()`, `self.add_adapters_to_pipeline(pipe, components=[...], **lora_config_kwargs)`, `self.run_pipe(pipe)`, and the class-scoped `base_pipe_output` fixture (baseline output of the un-adapted pipeline). `run_pipe` produces `base_pipe_output`, so the two are directly comparable — don't hand-roll a forward pass to compare against it. See `test_with_alpha_in_state_dict` and `test_lora_expansion_works_for_{absent,extra}_keys` on `TestFluxPipelineLoRA`.
- Load and save through the public API (`pipe.load_lora_weights`, `pipeline_class.save_lora_weights`, `pipe.set_adapters`, `pipe.unload_lora_weights`), and assert the adapter landed with `check_if_lora_correctly_set` from `...models.testing_utils.lora`.
- Nightly LoRA-checkpoint integration tests (loading real Hub LoRAs) go in the same file, in their own `@nightly @require_big_accelerator @require_peft_backend` class — see `TestFluxLoRAIntegration`. Still not part of an initial PR.

### Modular pipelines

- Location: `tests/modular_pipelines/<model>/test_modular_pipeline_<model>.py` (one config class + set of test classes per blockset / pipeline variant).
- **Define one config class**, `<Pipeline>ModularPipelineTesterConfig`, subclassing `BaseModularPipelineTesterConfig` (from `..testing_utils`). Set `pipeline_class`, `pipeline_blocks_class`, `pretrained_model_name_or_path`, `params` / `batch_params`, and implement `get_dummy_inputs(seed=0)`. Set `expected_workflow_blocks` to pin the block name → class ordering per workflow. The config holds the whole testing contract and performs no assertions.
- **Then one test class per concern**, each composing the config with a tester mixin from `..testing_utils`. Keep them separate — pytest reads class-level markers off the whole MRO, so folding a marked mixin (`@is_memory`, ...) into the same class as the others would tag every test in it:
  - `ModularPipelineTesterMixin` — call signature, batch consistency, float16, device placement, NaN-free output. Put pipeline-specific tests as methods on this class.
  - `ModularLoadingTesterMixin` — `save_pretrained`/`from_pretrained` round-trips, `modular_model_index.json` contents, `load_components`/`unload_components`.
  - `ModularWorkflowTesterMixin` — everything driven by the blocks class's `_workflow_map`; skips itself when there is none.
  - `ModularMemoryTesterMixin` — auto CPU offload, group offload, device memory reclaimed on unload.
  - `ModularGuiderTesterMixin` — only for pipelines with a `guider` component.
  - `ModularAutoOffloadTesterMixin` — opt-in, for pipelines with several offloadable model components; asserts on the offload *decisions* under simulated memory pressure.
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
- Do **not** add the model-level `LoraTesterMixin` (from `tests/models/testing_utils/lora.py`, distinct from the pipeline-level one) at the start, even if the model subclasses `PeftAdapterMixin` — strip it from the generated file for the initial PR.
- Reference: `tests/models/transformers/test_models_transformer_flux.py`.
