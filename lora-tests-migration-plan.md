# Plan: migrate `tests/lora/` to pipeline-level LoRA tester mixins

## Goal

Retire the unittest-style `PeftLoraLoaderMixinTests` (`tests/lora/utils.py`, ~2550 lines) in favor of
pytest-style LoRA tester mixins under `tests/pipelines/testing_utils/`, mirroring:

- how the model level does it: `tests/models/testing_utils/lora.py` (`LoraTesterMixin` composed with a
  `BaseModelTesterConfig`, e.g. `TestFluxTransformerLoRA`), and
- how pipeline tests already compose feature mixins with a shared config:
  `tests/pipelines/flux/test_pipeline_flux.py` (`FluxPipelineTesterConfig` + `PipelineTesterMixin`,
  `MemoryTesterMixin`, cache mixins, ...).

Each pipeline's LoRA tests then live next to its other pipeline tests as one more composed class
(e.g. `TestFluxPipelineLoRA(FluxPipelineTesterConfig, LoraTesterMixin)`), and the per-pipeline files in
`tests/lora/` get deleted as they migrate. Flux is the pilot.

## Current state (what we migrate from)

`PeftLoraLoaderMixinTests` carries its own component-building machinery that duplicates what the new
pipeline configs already provide:

- Class attrs `transformer_cls/transformer_kwargs/unet_kwargs/vae_kwargs/scheduler_cls/...` +
  `get_dummy_components(scheduler_cls, use_dora, lora_alpha)` returning
  `(components, text_lora_config, denoiser_lora_config)`.
- `get_dummy_inputs()` returning a `(noise, input_ids, pipeline_inputs)` tuple (only the third element is
  ever used), `output_type="np"`, numpy comparisons, manual baseline caching via
  `cached_non_lora_output` / `get_base_pipe_output()`.
- ~45 test methods: unittest assertions, `tempfile.TemporaryDirectory()`, `parameterized.expand` (plus an
  MRO hack in `test_group_offloading_inference_denoiser` to keep it overridable).

`tests/lora/test_lora_layers_flux.py` composes it into:

- `FluxLoRATests` (FluxPipeline) — 3 Flux-specific tests + 4 `unittest.skip("Not supported in Flux.")`
  overrides for UNet-only tests.
- `FluxControlLoRATests` (FluxControlPipeline) — 7 Control-LoRA/shape-expansion tests + same 4 skips.
- `FluxLoRAIntegrationTests`, `FluxControlLoRAIntegrationTests` — slow/nightly hub-checkpoint tests.

## Target architecture

### New file: `tests/pipelines/testing_utils/lora.py`

Three mixins, split the same way the existing pipeline mixins split by concern/hardware
(`common.py` / `memory.py` / `cache.py`), all exported from `tests/pipelines/testing_utils/__init__.py`:

```python
@is_lora
@require_peft_backend
class LoraTesterMixin(BasePipelineOutputMixin):
    """Core LoRA tests, runnable on CPU. ~35 tests."""

@is_lora
@require_peft_backend
class LoraMemoryTesterMixin(BasePipelineOutputMixin):
    """LoRA x memory optimizations: group offload, model CPU offload, layerwise casting. ~5 tests.

    `@require_torch_accelerator` is applied per-test on the offload tests only — the layerwise-casting
    tests run on CPU today (the PEFT-main CI job is a CPU runner) and must keep doing so."""

@is_lora
@require_peft_backend
class UNetLoraTesterMixin(BasePipelineOutputMixin):
    """UNet-only LoRA tests (block-scale dicts, padding-mode). Composed only into SD/SDXL classes."""
```

Splitting the UNet-only tests out kills the four `unittest.skip("Not supported in Flux.")` overrides in
every transformer-based pipeline — composition instead of skips, same pattern as the separate
IP-Adapter test class in `test_pipeline_flux.py`.

### Config contract

The mixins consume the **existing** `BasePipelineTesterConfig` subclasses unchanged
(`pipeline_class`, `get_dummy_components()`, `get_dummy_inputs()` with `output_type="pt"`,
`get_generator()`, the autouse `cleanup` fixture), plus LoRA-specific class attributes with defaults:

```python
class LoraTesterMixin(BasePipelineOutputMixin):
    lora_rank = 4
    lora_alpha = 4
    text_encoder_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    denoiser_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    supports_text_encoder_loras = True
```

The old `unet_kwargs`-vs-`transformer_kwargs` switch becomes a runtime helper:

```python
def get_denoiser(self, pipe):
    return pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
```

`setup_method` skips when `pipeline_class` is not a `LoraBaseMixin` subclass (mirrors the model-level
mixin's `PeftAdapterMixin` guard), so composing the mixin onto a LoRA-less pipeline fails loudly at
collection, not with attribute errors.

### Fixtures (pytest-first, per the ask)

- **`base_pipe_output`** — reuse the existing class-scoped fixture from `BasePipelineOutputMixin`
  verbatim; it replaces `cached_non_lora_output` / `get_base_pipe_output()` / `_compute_baseline_output()`.
  Every "LoRA should change the output" / "unload should restore the output" test takes it as an argument.
- **`text_lora_config` / `denoiser_lora_config`** — function-scoped fixtures building the default
  `LoraConfig` from the class attrs above:

  ```python
  @pytest.fixture
  def denoiser_lora_config(self):
      return LoraConfig(
          r=self.lora_rank, lora_alpha=self.lora_alpha,
          target_modules=self.denoiser_target_modules, init_lora_weights=False,
      )
  ```

  Tests needing non-default configs (DoRA, `rank_pattern`, alpha sweeps, `lora_bias`) build their own
  `LoraConfig` inline — the old `get_dummy_components(use_dora=..., lora_alpha=...)` plumbing disappears.
- **`tmp_path`** — replaces every `tempfile.TemporaryDirectory()` block (as in the model-level mixin).
- **`pytest.mark.parametrize`** — replaces `parameterized.expand` for
  `test_lora_adapter_metadata_is_loaded_correctly` / `..._save_load_inference` (alpha in `[4, 8, 16]`)
  and `test_group_offloading_inference_denoiser` (`offload_type`/`use_stream`). This also deletes the
  MRO-inspection hack, since pytest-parametrized methods override cleanly.
- Assertions: plain `assert`; tensor comparisons switch from `np.allclose` to `assert_tensors_close` /
  `not torch.allclose(...)` since new configs produce `output_type="pt"` tensors.
- Determinism: instead of `pipe(**inputs, generator=torch.manual_seed(0))`, each invocation re-fetches
  `inputs = self.get_dummy_inputs()` (fresh seeded generator) — same convention as
  `PipelineTesterMixin` and the `base_pipe_output` fixture, so outputs are comparable against it.

### Helpers carried over (module-level in the new file)

- `check_if_lora_correctly_set` — import from `tests/models/testing_utils/lora.py` instead of duplicating
  (pipeline tests already import from the models tree, e.g. `create_flux_ip_adapter_state_dict`).
- `determine_attention_kwargs_name`, `initialize_dummy_state_dict`, `state_dicts_almost_equal`,
  `check_module_lora_metadata` — port as-is.
- `_get_lora_state_dicts`, `_get_lora_adapter_metadata`, `_get_modules_to_save`,
  `add_adapters_to_pipeline` — port as mixin methods (assertions converted).
- The transformers>=5.6 `text_model.`-prefix repair trio
  (`_transformers_strips_text_model_prefix`, `_capture/_restore_text_encoder_lora_tensors`,
  `_needs_text_encoder_lora_repair`) — port as-is; still needed for save/load roundtrip tests on
  multi-text-encoder pipelines.

### Test disposition map (all 45 methods of `PeftLoraLoaderMixinTests`)

**→ `LoraTesterMixin` (core):**
`test_simple_inference_with_text_lora`, `..._and_scale`, `..._fused`, `..._unloaded`, `..._save_load`,
`test_simple_inference_with_partial_text_lora`, `test_simple_inference_save_pretrained_with_text_lora`
(all gated on `supports_text_encoder_loras` + `"text_encoder" in _lora_loadable_modules`),
`test_simple_inference_with_text_denoiser_lora_save_load`, `..._and_scale`, `..._unloaded`,
`..._unfused`, `test_simple_inference_with_text_lora_denoiser_fused`, `..._fused_multi`,
`test_simple_inference_with_text_denoiser_multi_adapter`, `..._delete_adapter`, `..._weighted`,
`test_lora_scale_kwargs_match_fusion`, `test_simple_inference_with_dora`,
`test_set_adapters_match_attention_kwargs`, `test_lora_B_bias`,
`test_correct_lora_configs_with_different_ranks`, `test_lora_unload_add_adapter`,
`test_inference_load_delete_load_adapters`, `test_get_adapters`, `test_get_list_adapters`,
`test_wrong_adapter_name_raises_error`, `test_multiple_wrong_adapter_name_raises_error`,
`test_missing_keys_warning`, `test_unexpected_keys_warning`, `test_logs_info_when_no_lora_keys_found`,
`test_lora_fuse_nan` (keep `@skip_mps` + xfail), `test_low_cpu_mem_usage_with_injection`,
`test_low_cpu_mem_usage_with_loading` (keep version-gate decorators),
`test_lora_adapter_metadata_is_loaded_correctly`, `test_lora_adapter_metadata_save_load_inference`
(parametrized).

**→ `LoraMemoryTesterMixin`:**
`test_group_offloading_inference_denoiser` (+ its `_test_...` body, parametrized),
`test_lora_loading_model_cpu_offload`, `test_lora_group_offloading_delete_adapters`,
`test_layerwise_casting_inference_denoiser`, `test_layerwise_casting_peft_input_autocast_denoiser`
(keep the existing xfail conditions).

**→ `UNetLoraTesterMixin`:**
`test_simple_inference_with_text_denoiser_block_scale`, `..._block_scale_for_all_dict_options`,
`test_simple_inference_with_text_denoiser_multi_adapter_block_lora`, `test_modify_padding_mode`.

**Dropped:** `test_simple_inference` (only validated the baseline runs and `output_shape`; the
class-scoped `base_pipe_output` fixture fails loudly if baseline inference breaks, and output shape is
covered by `PipelineTesterMixin`). The `output_shape` property requirement goes away with it.
Also dropped: `test_simple_inference_with_text_denoiser_lora_unfused_torch_compile` — permanently
`unittest.skip`-ed upstream ("failing for now") with a body that unconditionally accesses `pipe.unet`,
so it never ran and cannot run for transformer pipelines; dead code, not ported.

**Reworked:** `test_simple_inference_save_pretrained_with_text_lora` — with an attached adapter,
transformers' `save_pretrained` writes only the adapter files and records the model's `name_or_path`
instance attribute (snapshotted by `add_adapter` into the adapter config's `base_model_name_or_path`)
as the base checkpoint to reload from. Config-built dummy text encoders carry no such reference, so
the test now saves the bare base text encoders to `tmp_path` and points `name_or_path` at them before
attaching adapters, making the `save_pretrained` -> `from_pretrained` roundtrip self-contained for
every pipeline. (The old suite only worked because its text encoders came from Hub tiny checkpoints.)

## Flux migration (pilot)

### 1. `tests/pipelines/flux/test_pipeline_flux.py`

Add:

```python
class TestFluxPipelineLoRA(FluxPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Flux pipeline."""

    # Flux-specific tests migrated from FluxLoRATests:
    def test_with_alpha_in_state_dict(self, tmp_path): ...
    def test_lora_expansion_works_for_absent_keys(self, base_pipe_output, tmp_path): ...
    def test_lora_expansion_works_for_extra_keys(self, base_pipe_output, tmp_path): ...


class TestFluxPipelineLoRAMemory(FluxPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x offloading tests for the Flux pipeline."""
```

Notes:

- No `unittest.skip` overrides needed anymore (UNet-only tests live in `UNetLoraTesterMixin`).
- Components come from `FluxPipelineTesterConfig.get_dummy_components()` (locally-constructed tiny
  CLIP/T5 instead of the hub tiny checkpoints the old file downloaded — faster and deterministic via
  `torch.manual_seed(0)`). Flux's `_lora_loadable_modules = ["transformer", "text_encoder"]`, so the
  existing runtime guards keep T5 (`text_encoder_2`) out of text-LoRA paths automatically.
- Inputs come from `FluxPipelineTesterConfig.get_dummy_inputs()` (`output_type="pt"`); the migrated
  Flux-specific tests switch their `np.allclose` checks to torch comparisons accordingly.

Also add the integration classes, converted to the pytest style already used by `TestFluxPipelineSlow`
in the same file:

```python
@nightly
@require_big_accelerator
@require_peft_backend
class TestFluxLoRAIntegration:   # from FluxLoRAIntegrationTests
    # autouse cleanup fixture (gc + backend_empty_cache), pipeline built per-test or via fixture;
    # @pytest.mark.parametrize replaces parameterized.expand
```

### 2. FluxControl: out of scope for now

`tests/pipelines/flux/test_pipeline_flux_control.py` is still on the old unittest `PipelineTesterMixin`
and stays untouched in this migration. `FluxControlLoRATests` and `FluxControlLoRAIntegrationTests`
remain in `tests/lora/test_lora_layers_flux.py` on the old `PeftLoraLoaderMixinTests` until the
FluxControl pipeline test file itself migrates to the new config style (follow-up phase).

### 3. Trim `tests/lora/test_lora_layers_flux.py`

Remove `FluxLoRATests` and `FluxLoRAIntegrationTests` (now covered in
`tests/pipelines/flux/test_pipeline_flux.py`); keep only the two FluxControl classes. The file is
deleted entirely once FluxControl migrates.

### 4. CI: extend the PEFT-main LoRA job (same PR)

Without a CI change the migration silently loses coverage: the fast pipeline CPU job
(`.github/workflows/pr_tests.yml`, `run_fast_tests` / `pytest tests/pipelines`) runs in the
`diffusers-pytorch-cpu` image, which does **not** have peft installed (peft is not in the `test`
extra in `setup.py`, and the job only adds transformers/accelerate). The migrated tests would be
collected there but skipped wholesale by `@require_peft_backend`.

LoRA tests actually run in the dedicated `run_lora_tests` job, which installs PEFT from git main and
currently covers `tests/lora/` plus `tests/models/ -k "lora"` (the model-level migration precedent).
Add a pipelines pass to that job in the same PR:

```yaml
pytest -n 4 --max-worker-restart=0 --dist=loadfile \
  --make-reports=tests_pipelines_lora_peft_main \
  tests/pipelines/ -m "lora"
```

`-m lora` selects exactly what the mixins mark via `@is_lora` (more precise than the `-k` name
matching used for models). The existing `tests/lora/` line stays until the directory is fully
migrated. Also add the new report name to the job's "Failure short reports" step.

## Validation

- `pytest tests/pipelines/flux/test_pipeline_flux.py -m lora` — new tests collect and pass on CPU
  (offload mixin auto-skips without accelerator).
- Parity check: collected test count for `TestFluxPipelineLoRA` + `TestFluxPipelineLoRAMemory` (new) ≥
  old `pytest "tests/lora/test_lora_layers_flux.py::FluxLoRATests" --collect-only` count minus the 4
  `unittest.skip` overrides and `test_simple_inference` (intentionally dropped/relocated as documented
  above).
- `pytest tests/lora/test_lora_layers_flux.py` (now FluxControl-only) still green.
- `pytest -m "not lora" tests/pipelines/flux/` still green (marker filtering works).
- `make style` before the PR; self-review per `review-rules.md`.

## Audit (implemented): pipeline-level LoRA tests moved to the model level

Several tests inherited from `tests/lora/utils.py` never exercise pipeline-specific LoRA machinery
(`load_lora_weights`/`save_lora_weights` prefix handling, `pipe.fuse_lora(components=...)`,
cross-component `set_adapters`, text-encoder LoRA) — they only drive model-level APIs
(`PeftAdapterMixin` in `src/diffusers/loaders/peft.py`, `src/diffusers/utils/peft_utils.py`) and use
the pipeline merely as a forward harness. These were moved to `tests/models/testing_utils/lora.py`'s
`LoraTesterMixin` (renamed with a `lora` infix where needed so the CI's `tests/models/ -k "lora"`
selector keeps matching) and removed from the pipeline mixin:

| Test | Why it is model-level |
| --- | --- |
| `test_lora_B_bias` | Only `denoiser.add_adapter(config)` with `lora_bias` toggled; a model forward pass shows the same signal. |
| `test_correct_lora_configs_with_different_ranks` | Only `add_adapter`/`delete_adapters` with `rank_pattern`/`alpha_pattern`; resolution happens in `utils/peft_utils.py`. |
| `test_simple_inference_with_dora` (denoiser half) | `add_adapter(use_dora=True)` + forward. The model mixin's `test_save_load_lora_adapter` already has an (unused) `use_dora` parameter — exercise it there. Text-encoder DoRA is transformers-owned. |
| `test_low_cpu_mem_usage_with_injection` | Raw peft `inject_adapter_in_model` + `set_peft_model_state_dict` on components; touches no diffusers pipeline API at all. |
| `test_lora_fuse_nan` | `safe_fusing` lives on model-level `PeftAdapterMixin.fuse_lora` (`loaders/peft.py:661`); the pipeline variant only adds "NaN propagates through the VAE". |
| `test_missing_keys_warning`, `test_unexpected_keys_warning` | The warnings are emitted from `utils/peft_utils.py` via model-level `load_lora_adapter`; the pipeline layer only strips the `transformer.` prefix first. |
| `test_logs_info_when_no_lora_keys_found` (denoiser half) | "No LoRA keys associated to ..." is logged in model-level `load_lora_adapter` (`loaders/peft.py:377`). The text-encoder half (`load_lora_into_text_encoder`) is pipeline-level and stays. |
| `test_layerwise_casting_inference_denoiser`, `test_layerwise_casting_peft_input_autocast_denoiser` (part 1) | Adapter + `apply_layerwise_casting` on the denoiser; no pipeline API. The models testing_utils has casting tests but not casting x LoRA. Part 2 of the autocast test (`load_lora_weights` path) is the only pipeline-specific piece. |
| `test_modify_padding_mode` (UNet mixin) | Contains no LoRA logic at all — historical leftover from the old SD LoRA suite. Drop, or fold into model-level UNet tests when SD/SDXL migrate. |

Everything else stays pipeline-level: prefixed multi-component save/load roundtrips, fuse/unfuse via
`pipe.fuse_lora(components=...)`, cross-component `set_adapters` (incl. block-scale dicts and
attention-kwargs scale parity), `unload_lora_weights`, `get_active_adapters`/`get_list_adapters`,
all text-encoder LoRA handling, metadata via `save_lora_weights`/`lora_state_dict`, offloading x
`load_lora_weights` interplay, and the Flux expansion/norm/alpha tests.

Note the model-level mixin already covered adapter save/load roundtrip, wrong-adapter-name and
metadata roundtrip/corruption — the moves above fill genuine gaps (DoRA, `lora_bias`,
rank/alpha patterns, `safe_fusing`, low-cpu-mem injection, key warnings, casting x LoRA) rather than
duplicating coverage.

Implementation notes:
- The moved tests are structure-agnostic where the old per-pipeline overrides were not, so no
  per-model overrides are needed: `test_lora_fuse_nan` corrupts LoRA weights via `BaseTunerLayer`
  iteration instead of navigating hardcoded block-tower names (which Flux2 and Z-Image had to
  override before); `test_correct_lora_configs_with_different_ranks` prefers `attn`/`attention`
  module names but falls back to any `to_k` match; a `_model_output` helper flattens list outputs
  (Z-Image). Verified green across all 110 collected tests of `tests/models/ -k "lora" -m lora`.
- DoRA landed as a `use_dora` parametrization of the existing model-level `test_save_load_lora_adapter`.
- The moved model-level names: `test_lora_low_cpu_mem_usage_with_injection`, `test_lora_fuse_nan`,
  `test_lora_B_bias`, `test_correct_lora_configs_with_different_ranks`, `test_lora_missing_keys_warning`,
  `test_lora_unexpected_keys_warning`, `test_logs_info_when_no_lora_keys_found`,
  `test_lora_layerwise_casting_inference`, `test_lora_layerwise_casting_peft_input_autocast`.
- The pipeline-level `test_logs_info_when_no_lora_keys_found` was slimmed to its pipeline-specific
  parts: no-op `load_lora_weights` leaves the output unchanged + `load_lora_into_text_encoder` warns.
- `LoraMemoryTesterMixin` now holds only the offloading x `load_lora_weights` tests (all
  accelerator-gated); the casting x LoRA tests live at the model level.
- `test_modify_padding_mode` was dropped (no LoRA content).
- Resulting counts for Flux: 41 pipeline-level LoRA tests, 14 model-level LoRA tests
  (`TestFluxTransformerLoRA`).

## Follow-up phases (separate PRs, same recipe)

1. Migrate remaining transformer-based pipelines whose pipeline test files already use the new config
   style (`test_lora_layers_sd3.py`, `_lumina2.py`, `_sana.py`, `_cogvideox.py`, `_wan.py`, ...):
   one `TestXPipelineLoRA` class each, moving pipeline-specific extra tests and skips
   (as `pytest.skip` in overrides only where genuinely pipeline-specific). FluxControl joins here once
   `test_pipeline_flux_control.py` gets a new-style `FluxControlPipelineTesterConfig`, at which point
   `tests/lora/test_lora_layers_flux.py` is deleted.
2. Migrate the UNet pipelines (`test_lora_layers_sd.py`, `_sdxl.py`) composing
   `UNetLoraTesterMixin` in addition to the core mixin; these carry the most bespoke tests.
3. Once all `test_lora_layers_*.py` files are migrated, delete `tests/lora/utils.py`.
   `tests/lora/test_lora_loader_utils.py` (pure loader-utility tests, no pipeline mixin) moves under
   `tests/lora/` → `tests/others/` or stays as the sole survivor; decide at cleanup time.
4. Final CI cleanup: once `tests/lora/` is empty, drop its line from the `run_lora_tests` job — the
   `tests/pipelines/ -m "lora"` pass added in the Flux PR (see "CI" above) then carries all
   pipeline-level LoRA coverage. The pass must stay in the PEFT-main job (not the plain pipeline CPU
   job) since that is the only fast-test environment with peft installed.
