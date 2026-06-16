# Pipeline-Level Test Refactor Plan

Bring `tests/pipelines/` in line with the model-level testing framework in
`tests/models/testing_utils/`: a **config class + composable mixins** pattern,
driven by `pytest` instead of `unittest.TestCase`. Fold the LoRA suite
(`tests/lora/`) into it, and keep only the quantization tests that genuinely
need pipeline-level specifications.

---

## 1. Where we are today

### Model-level (the target pattern)
`tests/models/testing_utils/` is a package of small, single-responsibility
modules:

| Module | Provides |
|---|---|
| `common.py` | `BaseModelTesterConfig` (the config interface) + `ModelTesterMixin` |
| `attention.py`, `cache.py`, `compile.py`, `ip_adapter.py`, `lora.py`, `memory.py`, `parallelism.py`, `quantization.py`, `single_file.py`, `training.py` | one `*TesterMixin` family each |
| `__init__.py` | flat re-export of every mixin |

A concrete test file (`test_models_transformer_flux.py`) defines **one config
class** (`FluxTransformerTesterConfig(BaseModelTesterConfig)`) carrying
`model_class`, `get_init_dict()`, `get_dummy_inputs()`, `output_shape`, etc.,
then composes it with mixins, one `Test*` class per concern:

```python
class TestFluxTransformer(FluxTransformerTesterConfig, ModelTesterMixin): ...
class TestFluxTransformerLoRA(FluxTransformerTesterConfig, LoraTesterMixin): ...
class TestFluxTransformerBitsAndBytes(FluxTransformerTesterConfig, BitsAndBytesTesterMixin): ...
```

Key properties of the pattern:
- **`pytest`-native**: tests take `tmp_path`, use `@pytest.mark.parametrize`,
  `pytest.skip`, `pytest.raises`. No `setUp`/`tearDown`, no `self.assert*`.
- **Config is data, mixins are behavior**. Subclasses override a property/method
  to specialize a single test (see `TestFluxTransformerGGUF.get_dummy_inputs`).
- Determinism via top-level `enable_full_determinism()` and a `generator`
  property on the config.

### Pipeline-level (what we are refactoring)
`tests/pipelines/test_pipelines_common.py` (3104 lines) is a **single
monolith** holding ~12 mixins, all `unittest`-flavored and consumed by **208
test files**:

| Mixin | Concern |
|---|---|
| `PipelineTesterMixin` | core: save/load, signature, batching, dtype, offload, callbacks, variants, dduf, device-map, group-offload |
| `PipelineLatentTesterMixin` | pt/np/pil input/output equivalence, multi-VAE |
| `PipelineFromPipeTesterMixin` | `from_pipe()` round-trips |
| `PipelineKarrasSchedulerTesterMixin` | Karras scheduler shapes |
| `IPAdapterTesterMixin`, `FluxIPAdapterTesterMixin` | IP-Adapter |
| `SDFunctionTesterMixin` | VAE slicing/tiling, FreeU, fused QKV |
| `PyramidAttentionBroadcast/FasterCache/FirstBlockCache/TaylorSeerCache/MagCache TesterMixin` | cache hooks |
| `PipelinePushToHubTester` | hub upload |

Config interface a subclass implements today (kept, just formalized):
`pipeline_class`, `params`, `batch_params`, `callback_cfg_params`,
`required_optional_params`, `image_params`, `image_latents_params`,
`test_*` feature flags, `get_dummy_components()`, `get_dummy_inputs(device, seed)`.

### LoRA (`tests/lora/`)
`utils.py` (2540 lines) holds `PeftLoraLoaderMixinTests`
(`@require_peft_backend`) — 46 pipeline-level LoRA tests + ~10 helpers —
consumed by 19 `test_lora_layers_*.py` files. Each concrete file is
`class XxxLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests)` with its own
`get_dummy_components()` / `get_dummy_inputs()` and component class attributes
(`pipeline_class`, `scheduler_cls`, `transformer_kwargs`, text-encoder ids…).
This config interface **overlaps heavily** with the pipeline config interface —
it is essentially "pipeline config + LoRA target-module config."

This is the pipeline analogue of the model-level
`tests/models/testing_utils/lora.py` (`LoraTesterMixin`,
`LoraHotSwappingForModelTesterMixin`), which stays model-level.

### Quantization
- **Model-level mixins** already exist and are comprehensive
  (`tests/models/testing_utils/quantization.py`): BnB / Quanto / TorchAo / GGUF /
  ModelOpt / AutoRound, each with a `*TesterMixin`, `*CompileTesterMixin`,
  `*ConfigMixin`.
- **Standalone `tests/quantization/`** is **mixed**: per-backend files contain
  both model-only tests (redundant with the mixins above) and genuinely
  pipeline-level tests.

Pipeline-level quantization behaviors that must survive:
1. `PipelineQuantizationConfig` — multi-component / mixed-backend quant. **[done]**
   moved to `tests/pipelines/test_pipeline_quantization.py` (`TestPipelineQuantization`).
2. `pipe.enable_model_cpu_offload()` with quantized components.
3. `pipe.enable_group_offload()` + quant.
4. `torch.compile` + `pipe.enable_model_cpu_offload()` + quant — the **one**
   compile combination model-level can't reach, because `enable_model_cpu_offload`
   is a pipeline-only orchestration method (and it exercises regional
   `compile_repeated_blocks`). Plain quant+compile and quant+compile+group-offload
   are already covered by the model-level `QuantizationCompileTesterMixin`, so we
   do **not** duplicate them here (`test_torch_compile_utils.py`).
5. End-to-end pipeline inference quality with a quantized component.
6. LoRA loading into a quantized pipeline.
7. Pipeline serialization with per-component quant configs.

Everything else (layer-verification, param counts, footprint w/o offload,
single-model dequantize, model dtype/device rules, model serialization,
model-only compile) is **redundant** and dropped from the pipeline layer.

---

## 2. Target structure

Mirror the model package exactly:

```
tests/pipelines/testing_utils/
├── __init__.py            # flat re-export of every config + mixin
├── common.py              # [done] BasePipelineTesterConfig + PipelineTesterMixin (core)
├── ip_adapter.py          # [done] FluxIPAdapterTesterMixin
├── cache.py               # [done] CacheTesterMixin base + PAB / FasterCache / FirstBlockCache / TaylorSeer / MagCache
├── memory.py              # [done] MemoryTesterMixin umbrella: offload (seq/model), device-map, layerwise-casting, group-offload
├── utils.py               # [done] to_np, assert_mean_pixel_difference, qkv-fusion checks
├── lora.py                # [deferred] PipelineLoraTesterMixin (port of PeftLoraLoaderMixinTests — see §9)
├── latent.py              # [deferred] PipelineLatentTesterMixin
├── from_pipe.py           # [deferred] PipelineFromPipeTesterMixin
├── scheduler.py           # [deferred] PipelineKarrasSchedulerTesterMixin
└── sd_function.py         # [deferred] SDFunctionTesterMixin (VAE slicing/tiling, FreeU, fused QKV)
```

Standalone pipeline-level tests live as their own `test_*.py` modules, **not** in
`testing_utils/` (which only holds composable mixins/helpers — a standalone test
class there would never be pytest-collected):

```
tests/pipelines/
├── test_pipeline_push_to_hub.py     # [done] TestPipelinePushToHub (ex PipelinePushToHubTester)
└── test_pipeline_quantization.py    # [done] TestPipelineQuantization — PipelineQuantizationConfig (multi-component / mixed-backend)
```

`[done]` modules are implemented (Flux pilot for the mixins; the two standalone
modules above). `[deferred]` modules are follow-ups: `lora.py` is a large, distinct
subsystem ported on its own (see §9); the rest don't apply to Flux (it's not
SD/SDXL, doesn't use Karras schedulers, has no `from_pipe` lineage), so they're
added when the first pipeline that needs them is migrated. They are direct ports
of the corresponding mixins still in `test_pipelines_common.py`.

### Grouping convention (mirror the model-level package)

Follow exactly how `tests/models/testing_utils/` bundles its mixins, so the
mental model carries over 1:1:

- **Umbrella where the model package has one.** Memory is the clear case:
  model-level exposes `MemoryTesterMixin(CPUOffloadTesterMixin,
  GroupOffloadTesterMixin, LayerwiseCastingTesterMixin)` and concrete files
  compose the single umbrella (`TestFluxTransformerMemory`). The pipeline
  package does the same — `MemoryTesterMixin` bundles offload + device-map +
  group-offload + layerwise-casting, composed as one `Test<Pipe>Memory`.
- **Shared base + separate composition where the model package keeps them
  apart.** Caches share a `CacheTesterMixin` base (common machinery + config
  contract) but each backend (PAB/FasterCache/FirstBlockCache/TaylorSeer/MagCache)
  is its own `Test*` class with its own config/skip. No umbrella.
- **Standalone where there's nothing per-pipeline to compose.** The Hub
  push-to-hub test and the `PipelineQuantizationConfig` test are generic (fixed
  components / fixed model), so they're plain `test_*.py` modules, not mixins.

### Category markers + assertion style (mirror model-level)

- **Markers.** Each mixin carries the same class-level `pytest.mark` category as
  its model-level counterpart so suites can be filtered (`pytest -m "not memory"`,
  `-m cache`, …): `@is_cpu_offload` on the offload mixin, `@is_group_offload` on
  the group-offload mixin, `@is_memory` + `@require_accelerator` on the
  `MemoryTesterMixin` umbrella, `@is_cache` on each cache mixin, `@is_ip_adapter`
  on the IP-Adapter mixin. (Marks propagate through the MRO, so an umbrella test
  carries all of its bases' marks — same as model-level.)
- **Assertions.** Closeness checks use `assert_outputs_close(...)` — a thin
  `testing_utils/utils.py` wrapper over the model-level `assert_tensors_close`
  that accepts numpy/torch pipeline outputs and gives the same concise diff
  messages (max abs diff, location, mismatch count). "Should differ" checks and
  structural checks (hook installed, dtype, shape, NaN) stay as plain `assert`s.

### `BasePipelineTesterConfig` (new — the spine)
A single config base, analogous to `BaseModelTesterConfig`, formalizing the
contract subclasses already follow:

Required:
- `pipeline_class`
- `get_dummy_components()`
- `get_dummy_inputs(device, seed=0)`

Optional (defaults provided):
- `params`, `batch_params`, `callback_cfg_params`, `required_optional_params`
- `image_params`, `image_latents_params` (for latent mixin)
- feature flags: `test_attention_slicing`, `test_layerwise_casting`,
  `test_group_offloading`
- `torch_dtype`, and a `get_generator(seed)` helper returning
  `torch.Generator("cpu").manual_seed(seed)` (cpu generator for determinism,
  mirroring the model-level config's `generator`)

Concrete configs should expose their data via `@property` (e.g. `pipeline_class`,
`params`, `batch_params`, `test_layerwise_casting`) rather than plain class
attributes, matching the model-level `*TesterConfig` style.

> xformers tests are intentionally **not** ported — the dedicated xformers
> attention path is legacy and the model-level framework does not test it
> either; pipelines rely on the native attention backends.

LoRA adds, in `lora.py`, a thin extension of this config (target modules,
text-encoder ids, `denoiser_cls`/`transformer_cls`) so LoRA test classes reuse
the same `get_dummy_components`.

### Fixtures + canonical pipe builder (on `BasePipelineTesterConfig`)

Prefer `pytest` fixtures over per-test boilerplate. The config base exposes:

| Name | Kind / scope | Returns / does |
|---|---|---|
| `cleanup` | autouse fixture (function) | `gc.collect()` + `backend_empty_cache()` + `torch.compiler.reset()` before/after each test; skips deprecated pipelines (replaces `setUp`/`tearDown`) |
| `build_pipe()` | helper method | the **canonical preamble**: text encoders in eval mode, default attention processors, `.to(torch_device)`, progress bar off. Returns a fresh pipe. Every comparison test constructs through this so the only difference is the behavior under test |
| **`base_pipe_output`** | fixture, **class scope (memoized)** | the headline fixture: `build_pipe()` → run on the standard dummy inputs (with `torch.manual_seed(0)`) → return the output. Computed **once per test class** and reused by every comparison test |

`base_pipe_output` is the pytest-native replacement for the LoRA suite's
`get_base_pipe_output()` / `_compute_baseline_output()` (and for the many ad-hoc
"compute output with no offload / no quant, then compare" preambles). Because the
baseline is deterministic per config class (`enable_full_determinism()` at module
top + seeded inputs + `build_pipe`'s eval-mode text encoders so dropout is off),
it is memoized at class scope:

```python
class BasePipelineTesterConfig:
    def build_pipe(self):
        components = self.get_dummy_components()
        for k in components:
            if "text_encoder" in k and hasattr(components[k], "eval"):
                components[k].eval()
        pipe = self.pipeline_class(**components)
        for c in pipe.components.values():
            if hasattr(c, "set_default_attn_processor"):
                c.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    @pytest.fixture(scope="class")
    def base_pipe_output(self, request):
        cfg = request.cls()                       # class-scoped: no instance yet
        pipe = cfg.build_pipe()
        inputs = cfg.get_dummy_inputs(torch_device)
        torch.manual_seed(0)
        return pipe(**inputs)[0]
```

Comparison tests then take the fixture as a parameter and construct their pipe
through `build_pipe()` — only the behavior under test differs. Used today by
`test_save_load_local` and the memory mixin's CPU/sequential offload, device-map,
and pipeline-level group-offload tests:

```python
def test_model_cpu_offload_forward_pass(self, base_pipe_output):
    pipe = self.build_pipe()
    pipe.enable_model_cpu_offload(device=torch_device)
    inputs = self.get_dummy_inputs(torch_device)
    torch.manual_seed(0)
    out = pipe(**inputs)[0]
    assert np.abs(to_np(out) - to_np(base_pipe_output)).max() < 2e-4
```

> Per-mixin test inventories (the skeleton of each module) are in
> **§9 Appendix: Mixin skeletons**.

---

## 3. Per-area migration

### 3a. Core common mixins
1. Split `test_pipelines_common.py` into the modules above, **content-preserved**
   first, behavior-rewrite second.
2. Convert `unittest` → `pytest`:
   - `setUp`/`tearDown` → an autouse fixture in `BasePipelineTesterConfig`
     (VRAM cleanup + deprecation skip).
   - `tempfile.TemporaryDirectory()` → `tmp_path`.
   - `self.assertEqual/assertTrue/...` → bare `assert`; `assertRaises` →
     `pytest.raises`; `unittest.skipIf` → `pytest.mark.skipif`.
   - data-driven loops → `@pytest.mark.parametrize` (mirrors how model mixins
     parametrize dtype/quant configs).
3. Keep module-level helpers (`to_np`, `assert_mean_pixel_difference`,
   qkv-fusion checks, IP-adapter state-dict builders) in `utils.py`, re-exported.

### 3b. LoRA (fold `tests/lora/` in)
1. Port `PeftLoraLoaderMixinTests` → `testing_utils/lora.py` as
   `PipelineLoraTesterMixin` (+ keep a hotswap mixin if applicable), built on
   `BasePipelineTesterConfig`. Module-level helpers
   (`check_if_lora_correctly_set`, `state_dicts_almost_equal`,
   `determine_attention_kwargs_name`, the transformers>=5.6 repair shims) go to
   `testing_utils/utils.py`.
2. **Co-locate** LoRA tests with the pipeline they test, mirroring the model
   pattern (`TestFluxTransformerLoRA` lives beside `TestFluxTransformer`).
   The 19 `test_lora_layers_*.py` become a `Test<Pipe>LoRA(<Pipe>TesterConfig,
   PipelineLoraTesterMixin)` class in the corresponding
   `tests/pipelines/<family>/` file (or a sibling `test_*_lora.py` in that
   folder). This deletes the duplicate `get_dummy_components` that currently
   lives in both `tests/lora/` and `tests/pipelines/`.
3. Flux-specific extras (`test_with_alpha_in_state_dict`,
   `test_lora_expansion_works_for_*`) become overrides on the concrete class,
   exactly like `TestFluxTransformer.test_deprecated_inputs_*`.
4. Pipelines that opt out of a LoRA test (block-scale, padding-mode) override
   the method to `pytest.skip(...)` instead of today's `unittest.skip`.

### 3c. Quantization (keep pipeline-only)
1. **[done]** `PipelineQuantizationConfig` (multi-component / mixed-backend) is
   the genuinely pipeline-level surface. It's a standalone, pipeline-agnostic test,
   so it moved verbatim from `tests/quantization/test_pipeline_level_quantization.py`
   to `tests/pipelines/test_pipeline_quantization.py` as `TestPipelineQuantization`
   (a `pytest` non-mixin class: `@is_quantization` + the require/slow decorators, an
   autouse cleanup fixture, `@pytest.mark.parametrize`, `pytest.raises`), and the
   original was deleted.
2. **[deferred]** Per-pipeline quant **mixins** in `testing_utils/quantization.py`:
   - `PipelineQuantCpuOffloadTesterMixin` (model-cpu-offload + quant, plus the
     compile variant — the only pipeline-specific compile combination; see §1),
     `PipelineQuantGroupOffloadTesterMixin`, `PipelineQuantLoraTesterMixin`.
   - each consumes `BasePipelineTesterConfig` + a small quant-config hook,
     parametrized over backends like the model mixins. No single existing source
     to port verbatim, and they need real backends + GPU — hence deferred.
3. **Drop** from the pipeline layer everything redundant with the model mixins
   (listed in §1). Leave the standalone model-only quant tests where they are,
   or migrate them under the model mixins separately (out of scope here).

---

## 4. First PR scope (mixins + one pilot pipeline)

The work ships incrementally. The **first PR** establishes the framework and
proves it on a single pipeline — **Flux** — without touching the other 207
files. It does **not** attempt the repo-wide migration.

**In scope for PR #1:**
1. Scaffold the `tests/pipelines/testing_utils/` package: `__init__.py` +
   `BasePipelineTesterConfig` (in `common.py`) + `utils.py` helpers.
2. Add the individual mixins **that Flux actually composes**, ported from the
   monolith and rewritten `pytest`-native:
   - core `PipelineTesterMixin` (`common.py`)
   - `FluxIPAdapterTesterMixin` (`ip_adapter.py`)
   - the cache family with the shared `CacheTesterMixin` base (`cache.py`)
   - the `MemoryTesterMixin` umbrella (`memory.py`)
   - shared helpers (`utils.py`)

   Deferred (not in this PR): `lora.py` (large standalone subsystem — §9), and
   the Flux-inapplicable mixins (`latent`, `from_pipe`, `scheduler`,
   `sd_function`, `hub`, `quantization`).
3. **Supply those mixins to Flux only.** Write `FluxPipelineTesterConfig` +
   the composed `Test*` classes in `tests/pipelines/flux/test_pipeline_flux.py`,
   mirroring `test_models_transformer_flux.py`:
   ```python
   class TestFluxPipeline(FluxPipelineTesterConfig, PipelineTesterMixin): ...        # + Flux-specific tests
   class TestFluxPipelineMemory(FluxPipelineTesterConfig, MemoryTesterMixin): ...    # umbrella
   class TestFluxPipelineIPAdapter(FluxPipelineTesterConfig, FluxIPAdapterTesterMixin): ...
   class TestFluxPipelinePAB(FluxPipelineTesterConfig, PyramidAttentionBroadcastTesterMixin): ...
   class TestFluxPipelineFasterCache(FluxPipelineTesterConfig, FasterCacheTesterMixin): ...   # one per cache
   # ... FirstBlockCache / TaylorSeerCache / MagCache
   ```
4. Keep `test_pipelines_common.py` and `tests/lora/utils.py` **untouched** as
   shims so the other 207 pipeline files and 18 LoRA files keep passing.

> The `utils/generate_pipeline_tests.py` scaffolder (§8) and the Flux LoRA /
> pipeline-quant compositions are follow-ups, not part of this first PR.

**Goal of PR #1:** validate the config-class + mixin pattern end-to-end on one
family — naming, the `BasePipelineTesterConfig` contract, the LoRA config
extension, the pipeline-quant hook, and CI behavior — before committing to the
repo-wide rollout. Subsequent PRs convert the remaining families
(§5) and finally remove the shims.

---

## 5. Migration mechanics (208 files)

This is the riskiest part — 208 pipeline files + 19 LoRA files import from the
monolith. Do it incrementally, not big-bang.

1. **Build the package alongside the monolith.** Keep
   `test_pipelines_common.py` as a **thin compatibility shim** that re-exports
   every name from `testing_utils/` (so existing imports keep working). Same for
   `tests/lora/utils.py` re-exporting `PipelineLoraTesterMixin`.
2. **Pilot on Flux** (`tests/pipelines/flux/`): write
   `FluxPipelineTesterConfig` + `Test*` classes using the new package, including
   the LoRA and pipeline-quant classes. Validate the full pattern end-to-end on
   one family before scaling.
3. **Convert families incrementally**, deleting the corresponding
   `tests/lora/test_lora_layers_*.py` as each family's LoRA tests are folded in.
4. **Remove the shims** once all 208 files are migrated; delete
   `test_pipelines_common.py` and `tests/lora/utils.py`.
5. **`make style` + `make fix-copies`** after each batch — many pipeline tests
   carry `# Copied from` headers; the source must be migrated before copies, or
   the link temporarily broken and restored.

### Sequencing
1. **[done]** Scaffold `testing_utils/` package + `BasePipelineTesterConfig` + `utils.py`.
2. **[done]** Port core `PipelineTesterMixin` → `common.py` (pytest rewrite);
   monolith left untouched as the import shim.
3. **[done]** Port the mixins Flux composes (`ip_adapter`, `cache`, `memory`);
   move the standalone Hub + `PipelineQuantizationConfig` tests to their own
   `test_*.py` modules. Remaining common mixins (latent, from_pipe, scheduler,
   sd_function) deferred until a pipeline needs them.
4. Port LoRA → `lora.py` (its own follow-up; see §9).
5. Add per-pipeline quant **mixins** → `testing_utils/quantization.py` (§3c).
6. **[done]** Pilot: `tests/pipelines/flux/` on the implemented mixins.
7. Roll out family-by-family; delete `tests/lora/*` and quant redundancies as we go.
8. Remove shims; final `make style` + `make fix-copies`; CI green.

---

## 6. Risks & decisions to confirm

- **`get_dummy_inputs` signature drift.** Model-level uses
  `get_dummy_inputs()` (global `torch_device`); pipeline-level uses
  `get_dummy_inputs(device, seed=0)`. Keep the pipeline signature in
  `BasePipelineTesterConfig` (don't force-align with model-level).
- **`unittest` → `pytest` blast radius.** 208 files. The shim layer makes this
  incremental and reversible; without it, this is a single massive PR.
- **`# Copied from` ordering.** Must migrate copy *sources* before *targets*;
  budget `make fix-copies` churn.
- **LoRA test relocation.** Decision needed: co-locate in the pipeline file
  (matches model pattern) vs. keep a parallel `tests/lora/` tree that imports
  the new mixin. Recommendation: **co-locate** (kills duplicate
  `get_dummy_components`), but it touches more files.
- **Quant redundancy cut.** Confirm the model-level mixins already cover each
  dropped pipeline test before deleting, to avoid coverage gaps.

---

## 7. Definition of done

- `tests/pipelines/testing_utils/` exists, mirrors the model package, fully
  `pytest`-native.
- `test_pipelines_common.py` and `tests/lora/utils.py` removed (or reduced to
  nothing).
- Each pipeline family declares one `*PipelineTesterConfig` and composes
  `Test*` classes (core, LoRA, pipeline-quant) like the Flux **model** file.
- LoRA suite folded in; `tests/lora/test_lora_layers_*.py` removed.
- Pipeline-level quant retained; model-redundant quant dropped.
- `utils/generate_pipeline_tests.py` scaffolds a new pipeline's test file from
  its source, mirroring `utils/generate_model_tests.py`.
- `make style` + `make fix-copies` clean; CI green.

---

## 8. Test generator (`utils/generate_pipeline_tests.py`)

The model framework ships a scaffolder, `utils/generate_model_tests.py`, that
AST-parses a model file and emits a `*TesterConfig` + composed `Test*` classes.
We add the pipeline analogue so new pipelines get a correctly-wired test file in
one command, instead of hand-copying the Flux template.

```
python utils/generate_pipeline_tests.py src/diffusers/pipelines/flux/pipeline_flux.py
python utils/generate_pipeline_tests.py src/diffusers/pipelines/flux/pipeline_flux.py --include lora bnb --dry-run
```

### What it does (mirrors the model generator)
1. **AST-parse** the pipeline file: find the class inheriting `DiffusionPipeline`,
   collect its base mixins, and extract the `__call__` signature (param names,
   types, defaults) and `__init__` components.
2. **Map mixins/signals → testers** (tables below), always including the core
   `PipelineTesterMixin`.
3. **Infer the pipeline category** (text-to-image / img2img / inpaint / …) from
   the class/module name and `__call__` params (`image`, `mask_image`,
   `strength`) to pick the right `params` / `batch_params` constants from
   `tests/pipelines/pipeline_params.py`.
4. **Emit** a `<Name>PipelineTesterConfig` (with `pipeline_class`,
   `get_dummy_components()`/`get_dummy_inputs()` stubs commented with the real
   `__init__`/`__call__` params as TODO guides, suggested `params`/`batch_params`)
   plus one `Test*` class per selected mixin — same `pytest`-native shape as the
   Flux model test file.
5. **Resolve the output path** to `tests/pipelines/<family>/test_pipeline_<name>.py`.

### Mapping tables (pipeline-specific)

```python
ALWAYS_INCLUDE_TESTERS = ["PipelineTesterMixin", "MemoryTesterMixin"]  # umbrella, like model gen

# base loader/util mixins on the pipeline class  ->  tester mixin
MIXIN_TO_TESTER = {
    "LoraLoaderMixin":               "PipelineLoraTesterMixin",
    "StableDiffusionLoraLoaderMixin":"PipelineLoraTesterMixin",
    "SD3LoraLoaderMixin":            "PipelineLoraTesterMixin",
    "FluxLoraLoaderMixin":           "PipelineLoraTesterMixin",
    "IPAdapterMixin":                "IPAdapterTesterMixin",
    "FluxIPAdapterMixin":            "FluxIPAdapterTesterMixin",
    "StableDiffusionMixin":          "SDFunctionTesterMixin",
}

# signals derived from __call__ params / return type
SIGNAL_TO_TESTER = {
    "returns_image":     "PipelineLatentTesterMixin",   # has VAE / image output
    "is_sd_or_sdxl":     "PipelineFromPipeTesterMixin",  # from_pipe lineage
    "uses_karras_sched": "PipelineKarrasSchedulerTesterMixin",
}

OPTIONAL_TESTERS = [   # opt-in via --include, like the model generator (composable mixins only)
    ("PyramidAttentionBroadcastTesterMixin",  "pab_cache"),
    ("FirstBlockCacheTesterMixin",            "fbc_cache"),
    ("FasterCacheTesterMixin",                "faster_cache"),
    ("TaylorSeerCacheTesterMixin",            "taylorseer_cache"),
    ("MagCacheTesterMixin",                   "mag_cache"),
    ("PipelineLoraTesterMixin",               "lora"),     # force even if mixin not detected
]
# Hub and PipelineQuantizationConfig are standalone test modules (not composable
# per-pipeline mixins), so the generator does not emit them.
```

### Generated shape (Flux example)

```python
enable_full_determinism()


class FluxPipelineTesterConfig(BasePipelineTesterConfig):
    @property
    def pipeline_class(self):
        return FluxPipeline

    @property
    def params(self):   # __call__ params: prompt, prompt_2, height, width, guidance_scale, ...
        return TEXT_TO_IMAGE_PARAMS - {"negative_prompt", "cross_attention_kwargs"}

    @property
    def batch_params(self):
        return TEXT_TO_IMAGE_BATCH_PARAMS

    def get_dummy_components(self):
        # __init__ components: transformer, vae, text_encoder, text_encoder_2, ...
        # TODO: build dummy components
        return {}

    def get_dummy_inputs(self, device, seed=0):
        # TODO: fill in dummy inputs
        return {}


class TestFluxPipeline(FluxPipelineTesterConfig, PipelineTesterMixin): ...
class TestFluxPipelineLatent(FluxPipelineTesterConfig, PipelineLatentTesterMixin): ...
class TestFluxPipelineMemory(FluxPipelineTesterConfig, MemoryTesterMixin): ...        # umbrella
class TestFluxPipelineLoRA(FluxPipelineTesterConfig, PipelineLoraTesterMixin): ...
class TestFluxPipelineFasterCache(FluxPipelineTesterConfig, FasterCacheTesterMixin): ...
```

The generator is part of **PR #1** (alongside the mixins and the hand-validated
Flux file): use it to produce the Flux scaffold, then fill in the
`get_dummy_*` bodies — which doubles as the first real test of the generator.

---

## 9. Appendix: Mixin skeletons

Each module below is a config-consuming mixin (no `unittest.TestCase`). Test
names are the post-rewrite, `pytest`-native versions of what lives in the
monolith / `tests/lora/utils.py` today. Method bodies omitted — this is the
contract each mixin commits to.

Signatures are abbreviated: every comparison-style test takes the inherited
fixtures (`pipe`, `dummy_inputs`, `base_pipe_output`, `tmp_path`) instead of
rebuilding the pipeline or recomputing the baseline. Only `common.py` spells the
fixture parameters out in full below; the other modules follow the same
convention.

### `common.py` — `BasePipelineTesterConfig` + `PipelineTesterMixin`

```python
class BasePipelineTesterConfig:
    # ---- required ----
    pipeline_class
    def get_dummy_components(self): ...
    def get_dummy_inputs(self, device, seed=0): ...
    # ---- optional (defaults) ----
    params, batch_params, callback_cfg_params, required_optional_params
    image_params, image_latents_params
    test_attention_slicing = True
    test_layerwise_casting = False
    test_group_offloading = False
    torch_dtype = torch.float32
    def get_generator(self, seed): ...          # torch.Generator("cpu").manual_seed(seed)
    # canonical pipeline construction shared by base_pipe_output and the comparison tests
    def build_pipe(self): ...                    # eval text encoders + default attn procs + to(torch_device)
    # ---- fixtures (inherited by every mixin/subclass) ----
    @pytest.fixture(autouse=True)
    def cleanup(self): ...                       # VRAM cleanup + skip deprecated (ex setUp/tearDown)
    @pytest.fixture(scope="class")
    def base_pipe_output(self, request): ...     # memoized baseline output (see §2)


class PipelineTesterMixin:
    # --- save / load ---
    def test_save_load_local(self, tmp_path, base_pipe_output): ...   # compares loaded vs base_pipe_output
    def test_save_load_optional_components(self, tmp_path): ...
    def test_save_load_float16(self, tmp_path): ...
    def test_serialization_with_variants(self, tmp_path): ...
    def test_loading_with_variants(self, tmp_path): ...
    def test_loading_with_incorrect_variants_raises_error(self, tmp_path): ...
    # --- API surface ---
    def test_pipeline_call_signature(self): ...
    def test_components_function(self): ...
    # (test_StableDiffusionMixin_component dropped — SD-only; belongs in sd_function.py, not core)
    # --- batching / determinism ---
    def test_inference_batch_consistent(self): ...
    def test_inference_batch_single_identical(self): ...
    def test_dict_tuple_outputs_equivalent(self): ...
    def test_num_images_per_prompt(self): ...
    # --- dtype / device ---
    def test_to_device(self): ...
    def test_to_dtype(self): ...
    def test_float16_inference(self): ...                      # cuda/xpu only
    def test_torch_dtype_dict(self): ...                       # per-component dtype
    # --- attention paths ---
    def test_attention_slicing_forward_pass(self): ...   # gated by test_attention_slicing
    # --- guidance / callbacks ---
    def test_cfg(self): ...
    def test_callback_inputs(self): ...
    def test_callback_cfg(self): ...
    # --- prompt encoding ---
    def test_encode_prompt_works_in_isolation(self): ...
```

### `memory.py` — offload, device-map, casting (umbrella, mirrors model-level)

All memory-placement concerns live here: CPU/sequential offload, group
offload, layerwise casting, and `device_map` loading. Following the model-level
convention (`MemoryTesterMixin(CPUOffloadTesterMixin, GroupOffloadTesterMixin,
LayerwiseCastingTesterMixin)`), the sub-mixins are bundled into a single
**umbrella** `MemoryTesterMixin` that concrete pipelines compose as one
`Test<Pipe>Memory` class.

```python
@is_cpu_offload
class PipelineOffloadTesterMixin:
    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output): ...
    def test_model_cpu_offload_forward_pass(self, base_pipe_output): ...
    def test_cpu_offload_forward_pass_twice(self): ...
    def test_sequential_offload_forward_pass_twice(self): ...
    def test_pipeline_with_accelerator_device_map(self, tmp_path, base_pipe_output): ...   # moved out of core

class LayerwiseCastingTesterMixin:
    def test_layerwise_casting_inference(self): ...             # gated by test_layerwise_casting

@is_group_offload
class GroupOffloadTesterMixin:
    def test_group_offloading_inference(self): ...             # gated by test_group_offloading
    def test_pipeline_level_group_offloading_sanity_checks(self): ...
    def test_pipeline_level_group_offloading_inference(self, base_pipe_output): ...

@is_memory
@require_accelerator
class MemoryTesterMixin(PipelineOffloadTesterMixin, GroupOffloadTesterMixin, LayerwiseCastingTesterMixin): ...

# umbrella — what concrete pipelines actually compose (cf. model-level MemoryTesterMixin)
class MemoryTesterMixin(
    PipelineOffloadTesterMixin, GroupOffloadTesterMixin, LayerwiseCastingTesterMixin
): ...
```

### `latent.py` — `PipelineLatentTesterMixin`

```python
class PipelineLatentTesterMixin:
    image_params           # required override
    image_latents_params   # required override
    def get_dummy_inputs_by_type(self, device, seed, input_image_type, output_type): ...
    def test_pt_np_pil_outputs_equivalent(self): ...
    def test_pt_np_pil_inputs_equivalent(self): ...
    def test_latents_input(self): ...
    def test_multi_vae(self): ...
```

### `from_pipe.py` — `PipelineFromPipeTesterMixin`

```python
class PipelineFromPipeTesterMixin:
    original_pipeline_class                      # SD / SDXL / Kolors selector
    def get_dummy_inputs_pipe(self, device, seed=0): ...
    def get_dummy_inputs_for_pipe_original(self, device, seed=0): ...
    def test_from_pipe_consistent_config(self): ...
    def test_from_pipe_consistent_forward_pass(self): ...
    def test_from_pipe_consistent_forward_pass_cpu_offload(self): ...
```

### `scheduler.py` — `PipelineKarrasSchedulerTesterMixin`

```python
class PipelineKarrasSchedulerTesterMixin:
    def test_karras_schedulers_shape(self): ...
```

### `ip_adapter.py` — IP-Adapter mixins

```python
@is_ip_adapter
class IPAdapterTesterMixin:               # [deferred] standard UNet variant
    def _get_dummy_image_embeds(self, cross_attention_dim=32): ...
    def _get_dummy_faceid_image_embeds(self, cross_attention_dim=32): ...
    def _get_dummy_masks(self, input_size=64): ...
    def test_pipeline_signature(self): ...
    def test_ip_adapter(self): ...
    def test_ip_adapter_cfg(self): ...
    def test_ip_adapter_masks(self): ...
    def test_ip_adapter_faceid(self): ...

@is_ip_adapter
class FluxIPAdapterTesterMixin:          # [done] Flux variant (no masks/faceid/cfg split)
    def test_pipeline_signature(self): ...
    def test_ip_adapter(self): ...
```

### `sd_function.py` — `SDFunctionTesterMixin`

```python
class SDFunctionTesterMixin:
    def test_vae_slicing(self): ...
    def test_vae_tiling(self): ...
    def test_freeu(self): ...             # @skip_mps (ComplexFloat)
    def test_fused_qkv_projections(self): ...
```

### `cache.py` — cache-hook mixins (one config + one inference test each)

Mirrors model-level `cache.py`: a shared `CacheTesterMixin` base holds the
common implementation + config contract, and each cache backend is composed
**separately** (own config, independent skip) — there is no cache umbrella, just
like the model-level Flux file uses one `Test*` class per cache type.

```python
class CacheTesterMixin:
    # shared base: common cache test machinery (_test_cache_inference) + config contract (cf. model-level)
    ...

@is_cache
class PyramidAttentionBroadcastTesterMixin(CacheTesterMixin):
    pab_config
    def test_pyramid_attention_broadcast_layers(self): ...
    def test_pyramid_attention_broadcast_inference(self): ...

@is_cache
class FasterCacheTesterMixin(CacheTesterMixin):
    faster_cache_config
    def test_faster_cache_basic_warning_or_errors_raised(self): ...
    def test_faster_cache_inference(self): ...
    def test_faster_cache_state(self): ...

@is_cache
class FirstBlockCacheTesterMixin(CacheTesterMixin):
    first_block_cache_config
    def test_first_block_cache_inference(self): ...

@is_cache
class TaylorSeerCacheTesterMixin(CacheTesterMixin):
    taylorseer_cache_config
    def test_taylorseer_cache_inference(self): ...

@is_cache
class MagCacheTesterMixin(CacheTesterMixin):
    mag_cache_config
    def test_mag_cache_inference(self): ...
```

### `tests/pipelines/test_pipeline_push_to_hub.py` — standalone — **[done]**

Generic Hub integration test (builds its own fixed SD components, no per-pipeline
config). Stays a `unittest.TestCase` because `@is_staging_test` is a
`unittest.skip`-based decorator and the test isn't composed with the config/fixtures.

```python
@is_staging_test
class TestPipelinePushToHub(unittest.TestCase):   # ex PipelinePushToHubTester (moved out of the monolith)
    def test_push_to_hub(self): ...
    def test_push_to_hub_in_organization(self): ...
    def test_push_to_hub_library_name(self): ...   # @skipIf(not is_jinja_available())
```

### `lora.py` — `PipelineLoraTesterMixin` (ex `PeftLoraLoaderMixinTests`, 46 tests) — **[deferred]**

**Status:** deferred to its own follow-up PR. LoRA is a large, distinct subsystem
(46 tests, ~2540 lines in `tests/lora/utils.py`) with its own config interface and
`unittest`/`@parameterized` idioms, so it is ported on its own rather than bundled
into the Flux pilot. A trial re-export wiring was validated (Flux LoRA: 47 passed,
5 skipped) before being reverted, so the composition path is known-good.

The follow-up will:
- port `PeftLoraLoaderMixinTests` → `PipelineLoraTesterMixin` pytest-native
  (drop `unittest.TestCase`, convert `self.assert*` → `assert`, `@parameterized`
  → `@pytest.mark.parametrize`), preserving comments and the transformers>=5.6
  text-encoder repair shims;
- move the module-level helpers (`check_if_lora_correctly_set`,
  `determine_attention_kwargs_name`, `state_dicts_almost_equal`, …) into
  `testing_utils/utils.py`;
- migrate the 19 `tests/lora/test_lora_layers_*.py` files to compose it, then
  delete `tests/lora/`.

Its config interface (a LoRA-specific extension): `scheduler_cls`/`scheduler_kwargs`,
`denoiser_target_modules`, `text_encoder_target_modules`, `has_two/three_text_encoders`,
`transformer_cls`/`unet_kwargs`, text-encoder & tokenizer ids/classes, `output_shape`,
and `get_dummy_inputs(with_generator=True) -> (noise, input_ids, pipeline_inputs)`.
The tests it will provide:

```python
class PipelineLoraTesterMixin:
    # --- baseline / text-encoder LoRA ---
    def test_simple_inference(self): ...
    def test_simple_inference_with_text_lora(self): ...
    def test_simple_inference_with_text_lora_and_scale(self): ...
    def test_simple_inference_with_text_lora_fused(self): ...
    def test_simple_inference_with_text_lora_unloaded(self): ...
    def test_simple_inference_with_text_lora_save_load(self): ...
    def test_simple_inference_with_partial_text_lora(self): ...
    def test_simple_inference_save_pretrained_with_text_lora(self): ...
    # --- text + denoiser LoRA ---
    def test_simple_inference_with_text_denoiser_lora_save_load(self): ...
    def test_simple_inference_with_text_denoiser_lora_and_scale(self): ...
    def test_simple_inference_with_text_lora_denoiser_fused(self): ...
    def test_simple_inference_with_text_denoiser_lora_unfused(self): ...
    def test_simple_inference_with_text_denoiser_lora_unloaded(self): ...
    def test_simple_inference_with_text_denoiser_lora_unfused_torch_compile(self): ...
    # --- block-level scale (skippable per pipeline) ---
    def test_simple_inference_with_text_denoiser_block_scale(self): ...
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self): ...
    # --- multi-adapter ---
    def test_simple_inference_with_text_denoiser_multi_adapter(self): ...
    def test_simple_inference_with_text_denoiser_multi_adapter_weighted(self): ...
    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self): ...
    def test_simple_inference_with_text_denoiser_multi_adapter_delete_adapter(self): ...
    def test_simple_inference_with_text_lora_denoiser_fused_multi(self): ...
    def test_lora_unload_add_adapter(self): ...
    def test_inference_load_delete_load_adapters(self): ...
    # --- introspection / config / metadata ---
    def test_get_adapters(self): ...
    def test_get_list_adapters(self): ...
    def test_correct_lora_configs_with_different_ranks(self): ...
    def test_lora_B_bias(self): ...
    def test_lora_adapter_metadata_is_loaded_correctly(self): ...
    def test_lora_adapter_metadata_save_load_inference(self): ...
    def test_set_adapters_match_attention_kwargs(self): ...
    def test_lora_scale_kwargs_match_fusion(self): ...
    # --- errors / warnings ---
    def test_wrong_adapter_name_raises_error(self): ...
    def test_multiple_wrong_adapter_name_raises_error(self): ...
    def test_lora_fuse_nan(self): ...
    def test_missing_keys_warning(self): ...
    def test_unexpected_keys_warning(self): ...
    def test_logs_info_when_no_lora_keys_found(self): ...
    # --- DoRA / padding / low-mem ---
    def test_simple_inference_with_dora(self): ...
    def test_modify_padding_mode(self): ...
    def test_low_cpu_mem_usage_with_injection(self): ...
    def test_low_cpu_mem_usage_with_loading(self): ...
    # --- casting / offload (LoRA-aware) ---
    def test_layerwise_casting_inference_denoiser(self): ...
    def test_layerwise_casting_peft_input_autocast_denoiser(self): ...
    def test_group_offloading_inference_denoiser(self): ...     # parametrized block/leaf
    def test_lora_loading_model_cpu_offload(self): ...
    def test_lora_group_offloading_delete_adapters(self): ...
```

Pipeline-specific extras stay as overrides on the concrete class, e.g. Flux:
`test_with_alpha_in_state_dict`, `test_lora_expansion_works_for_absent_keys`,
`test_lora_expansion_works_for_extra_keys`; block-scale / padding tests
overridden to `pytest.skip(...)`.

### `tests/pipelines/test_pipeline_quantization.py` — standalone — **[done]**

`PipelineQuantizationConfig` is the genuinely pipeline-level quant surface
(multi-component / mixed-backend quant driven through `DiffusionPipeline.from_pretrained`).
The existing test (`PipelineQuantizationTests`) is standalone and pipeline-agnostic
(fixed `tiny-flux-pipe`), so it moved into this `test_*.py` module as a single
non-mixin `pytest` class `TestPipelineQuantization` — class-level `@is_quantization`
+ the require/`@slow` decorators, an autouse cleanup fixture, `tmp_path` for
save/load, `@pytest.mark.parametrize` for the kwargs/mapping variants, and
`pytest.raises`/bare `assert`s. It covers config-set-through-kwargs/granular,
validation errors, save/load round-trip, invalid-component warnings, repr, and
single-component quant.

**Deferred (per-pipeline quant mixins):** the composable, per-pipeline quant tests
the plan envisioned — `enable_model_cpu_offload`+quant (incl. the regional
`compile_repeated_blocks` combo), pipeline-level `enable_group_offload`+quant, and
LoRA-into-a-quantized-pipeline — are a follow-up. They need real backends + GPU and
have no single existing source to port verbatim. When added, they live as mixins in
`testing_utils/quantization.py` and compose into a `Test<Pipe>Quant` class.

Dropped entirely (covered by the model-level mixins in
`tests/models/testing_utils/quantization.py`): layer-verification, parameter
counts, footprint without offload, single-model dequantize, model dtype/device
rules, model-only serialization, and **quant+compile** /
**quant+compile+group-offload** (model-level `QuantizationCompileTesterMixin`
already covers these — only `enable_model_cpu_offload`+compile would be
pipeline-specific).

### `utils.py` — module-level helpers (no tests)

`to_np`, `assert_outputs_close` (numpy/torch wrapper over the model-level
`assert_tensors_close`), `check_same_shape`, `assert_mean_pixel_difference`,
`check_qkv_fusion_matches_attn_procs_length`, `check_qkv_fusion_processors_exist`,
`check_qkv_fused_layers_exist`, IP-adapter state-dict builders. (LoRA helpers —
`check_if_lora_correctly_set`, `determine_attention_kwargs_name`, the
transformers>=5.6 repair shims — move here with the deferred `lora.py` port.)
