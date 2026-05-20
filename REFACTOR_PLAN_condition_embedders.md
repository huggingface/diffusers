# Refactor: Move pipeline-local model classes into `src/diffusers/models/`

## Motivation

Several `ModelMixin` / `ConfigMixin` subclasses currently live under `src/diffusers/pipelines/<pipeline>/` (e.g. `modeling_audioldm2.py`, `connectors.py`, `vocoder.py`). They are model components, not pipeline glue, and the modular work in PR #13732 forced the question of where to put a new one. Going forward, model classes should live under `src/diffusers/models/`. This refactor introduces a new `condition_embedders/` submodule for projection/conditioning encoders, an `others/` submodule for the long-tail pipeline-local oddities, and migrates the remaining classes into existing submodules (`unets/`, `autoencoders/`, `transformers/`).

## Scope

The classes below were located via codebase inventory. Each row gives current source → new home. **Every class** in this list gets a deprecation shim at its old import path — even classes that aren't re-exported from a pipeline `__init__.py`, because users may still do `from diffusers.pipelines.<pipeline>.<module> import ClassName` directly.

### → `models/unets/`

| Class | From | New file |
|---|---|---|
| `AudioLDM2UNet2DConditionModel` | `pipelines/audioldm2/modeling_audioldm2.py:163` | `models/unets/unet_2d_condition_audioldm2.py` |

### → `models/autoencoders/` (mild scope drift to "latent / audio codecs")

| Class | From | New file |
|---|---|---|
| `LTXLatentUpsamplerModel` | `pipelines/ltx/modeling_latent_upsampler.py:76` | `models/autoencoders/latent_upsampler_ltx.py` |
| `LTX2LatentUpsamplerModel` | `pipelines/ltx2/latent_upsampler.py:170` | `models/autoencoders/latent_upsampler_ltx2.py` |
| `LTX2Vocoder` | `pipelines/ltx2/vocoder.py:279` | `models/autoencoders/vocoder_ltx2.py` |
| `LTX2VocoderWithBWE` | `pipelines/ltx2/vocoder.py:479` | (same file as above) |
| `AceStepAudioTokenizer` | `pipelines/ace_step/modeling_ace_step.py:665` | `models/autoencoders/audio_tokenizer_ace_step.py` |
| `AceStepAudioTokenDetokenizer` | `pipelines/ace_step/modeling_ace_step.py:565` | (same file as above) |

### → `models/condition_embedders/` (new)

| Class | From | New file |
|---|---|---|
| `AudioLDM2ProjectionModel` | `pipelines/audioldm2/modeling_audioldm2.py:78` | `models/condition_embedders/projection_audioldm2.py` |
| `StableAudioProjectionModel` | `pipelines/stable_audio/modeling_stable_audio.py:114` | `models/condition_embedders/projection_stable_audio.py` |
| `LTX2TextConnectors` | `pipelines/ltx2/connectors.py:331` | `models/condition_embedders/text_connector_ltx2.py` |
| `ReduxImageEncoder` | `pipelines/flux/modeling_flux.py:31` | `models/condition_embedders/image_encoder_redux.py` |
| `CLIPImageProjection` | `pipelines/stable_diffusion/clip_image_project_model.py:21` | `models/condition_embedders/projection_clip_image.py` |
| `AceStepConditionEncoder` | `pipelines/ace_step/modeling_ace_step.py:752` | `models/condition_embedders/condition_encoder_ace_step.py` |
| `AceStepLyricEncoder` | `pipelines/ace_step/modeling_ace_step.py:127` | (same file as above) |
| `AceStepTimbreEncoder` | `pipelines/ace_step/modeling_ace_step.py:233` | (same file as above) |

`AceStepLyricEncoder` and `AceStepTimbreEncoder` are not re-exported from any `__init__.py`, but the shim still goes in `modeling_ace_step.py` because the deep-import path `from diffusers.pipelines.ace_step.modeling_ace_step import AceStepLyricEncoder` is part of the implicit public surface.

### → `models/others/` (new — pipeline-local oddities with no obvious peers)

| Class | From | New file |
|---|---|---|
| `ShapERenderer` (+ `MLPNeRSTFModel`, `ShapEParamsProjModel`, `MLPNeRFModelOutput`, plus the `BoundingBoxVolume` / `StratifiedRaySampler` / `ImportanceRaySampler` / `VoidNeRFModel` helpers) | `pipelines/shap_e/renderer.py` | `models/others/renderer_shap_e.py` |
| `IFWatermarker` | `pipelines/deepfloyd_if/watermark.py:10` | `models/others/watermark_if.py` |
| `StableUnCLIPImageNormalizer` | `pipelines/stable_diffusion/stable_unclip_image_normalizer.py:22` | `models/others/image_normalizer_stable_unclip.py` |

`ChatGLMModel` (`pipelines/kolors/text_encoder.py`) is intentionally **excluded** — it is a HuggingFace `transformers` PreTrainedModel re-implementation, not a `ModelMixin`, and the right long-term home is upstream `transformers`. Leaving it in `pipelines/kolors/` keeps that boundary visible. (Open for discussion if reviewers disagree.)

## Implementation order

This ships as a **single PR**, but built up via a sequence of self-contained commits so that reviewers can step through the pattern once and then skim the rest. The first commit lands the convention (new submodule scaffold + one full end-to-end migration including the deprecation shim, the `__init__.py` wiring, and the first-party import flip); every subsequent commit mirrors that pattern for the next class or group, so individual commits can be bisected if something breaks.

Suggested commit sequence (push after each so reviewers see the progression):

1. **Scaffold + first migration (sets the pattern).** Create `models/condition_embedders/__init__.py` and `models/others/__init__.py`. Migrate `CLIPImageProjection` (small, single-file, public-API export, only one downstream caller) end-to-end: new file, shim with `deprecate()`, top-level `__init__.py` re-export, `make fix-copies`, first-party caller update. This commit's diff is the template for the rest.
2. **AudioLDM2 split.** `modeling_audioldm2.py` hosts both a UNet and a projection model; the file gets split into `models/unets/unet_2d_condition_audioldm2.py` and `models/condition_embedders/projection_audioldm2.py`. The old file becomes a shim that re-exports both classes (plus any internal helpers it still owns that the pipeline imports). Flip `pipeline_audioldm2.py` to import from the new paths.
3. **AceStep split.** `modeling_ace_step.py` hosts five classes across two destinations (autoencoders + condition_embedders). Split into per-destination files; old file becomes a shim for **all five** classes (`AceStepAudioTokenizer`, `AceStepAudioTokenDetokenizer`, `AceStepConditionEncoder`, `AceStepLyricEncoder`, `AceStepTimbreEncoder`). Flip `pipeline_ace_step.py` line 29 from `from .modeling_ace_step import AceStepAudioTokenDetokenizer, AceStepAudioTokenizer, AceStepConditionEncoder` to the new `from ...models.autoencoders import ...` / `from ...models.condition_embedders import ...` imports.
4. **Stable Audio + Flux Redux.** `StableAudioProjectionModel` and `ReduxImageEncoder`. `ReduxImageEncoder` is also re-exported from `qwenimage/__init__.py` — update that re-export to point at the new location too.
5. **LTX / LTX2 family.** All five LTX classes (`LTXLatentUpsamplerModel`, `LTX2LatentUpsamplerModel`, `LTX2Vocoder`, `LTX2VocoderWithBWE`, `LTX2TextConnectors`). These are not in the top-level `diffusers.__init__.py` — only in the pipeline `__init__.py` — so the top-level re-export is skipped, but the pipeline-level `__init__.py` should still re-export from the new path. Flip all `ltx*/pipeline_*.py` and `ltx*/__init__.py` relative imports.
6. **UNet AudioLDM2** can be folded into commit 2 (same source file). Listed here as a reminder that the `models/unets/` registration also needs to happen.
7. **`others/` migrations.** `ShapERenderer` trio, `IFWatermarker`, `StableUnCLIPImageNormalizer`. Same recipe.

A single PR is the right granularity because the changes are mostly mechanical and reviewers benefit from seeing the full move in one place. The commit boundaries are for navigability inside that PR.

## Per-class change recipe

For each class being moved, the change is:

1. **Create the new file** at the target path. Move the class definition verbatim, plus any private helpers it owns (module-level constants, helper functions, internal `nn.Module` subclasses). Adjust relative imports for the new depth — `from ..pipeline_utils import X` (pipelines, depth-2) becomes `from ..modeling_utils import X` / `from ...utils import ...` (models, depth-3 to utils), etc.
2. **Register the new public name.** Add the class to the appropriate `models/<subdir>/__init__.py`, then re-export from `models/__init__.py`, then ensure `src/diffusers/__init__.py` exports it from the new path (if it was previously in the top-level `__init__`).
3. **Turn the old file into a shim** using `diffusers.utils.deprecate`. Keep the file present — do not delete. Replace its body with:
   ```python
   from ...models.condition_embedders.projection_audioldm2 import AudioLDM2ProjectionModel as _AudioLDM2ProjectionModel
   from ...utils import deprecate


   class AudioLDM2ProjectionModel(_AudioLDM2ProjectionModel):
       def __init__(self, *args, **kwargs):
           deprecate(
               "AudioLDM2ProjectionModel",
               "1.0.0",
               "Importing `AudioLDM2ProjectionModel` from `diffusers.pipelines.audioldm2.modeling_audioldm2` is "
               "deprecated. Import it from `diffusers.models.condition_embedders` instead "
               "(or `from diffusers import AudioLDM2ProjectionModel`).",
           )
           super().__init__(*args, **kwargs)
   ```
   Subclassing rather than re-assigning preserves `isinstance` checks and gives a clean place to fire the warning on instantiation (not on import). For files that hosted multiple classes (e.g. `modeling_audioldm2.py`, `modeling_ace_step.py`), repeat the shim block for each moved class in the same file. The version slot (`"1.0.0"` above) is the pinned removal target — confirm against current `diffusers.__version__` (today: `0.39.0.dev0`) and pick a version that gives at least one full release cycle of warning.
4. **Update first-party imports** to point at the new location:
   - **Pipeline files in the same folder.** Concrete example: `src/diffusers/pipelines/ace_step/pipeline_ace_step.py:29` is currently `from .modeling_ace_step import AceStepAudioTokenDetokenizer, AceStepAudioTokenizer, AceStepConditionEncoder` — flip to `from ...models.autoencoders import AceStepAudioTokenizer, AceStepAudioTokenDetokenizer` and `from ...models.condition_embedders import AceStepConditionEncoder`. Apply the same flip to every pipeline file in the inventory (audioldm2, stable_audio, ltx, ltx2, flux, qwenimage, shap_e, deepfloyd_if, stable_diffusion).
   - Conversion scripts under `scripts/`.
   - Tests under `tests/`.
   - Cross-pipeline re-exports: `qwenimage/__init__.py` currently re-exports `ReduxImageEncoder` from the flux modeling file — point it at the new path.

   Do not add deprecation imports to first-party code — fix the import sites directly so we are not warning ourselves.
5. **Dummy objects.** Re-run `utils/check_dummies.py` (or `make fix-copies`) so the auto-generated `utils/dummy_pt_objects.py` reflects the new export paths.

## Deprecation warning policy

- Use `diffusers.utils.deprecate(class_name, version, message)` (matches the rest of the library). The version slot is meaningful: once `diffusers.__version__ >= version`, `deprecate()` raises a `ValueError` telling whoever sees it that the shim needs to be deleted. That gives us an automatic, in-CI nudge to clean up rather than letting shims rot forever.
- One warning per class, fired in `__init__` of the shim subclass, not at module import time. Importing a module shouldn't spam — only constructing a deprecated class should warn. (Users may legitimately have `from ... import X` in a file they never instantiate; we shouldn't punish them.)
- Warning message format (consistent across the refactor):
  > `Importing \`{ClassName}\` from \`{old.dotted.path}\` is deprecated. Import it from \`{new.dotted.path}\` instead (or \`from diffusers import {ClassName}\` when re-exported at the top level).`
- Pin every shim to the **same** removal version so the whole batch can be deleted in one cleanup PR.

## Things I checked and decided against

- **Moving `ChatGLMModel`** — see above; it's a `transformers` reimpl, not a `ModelMixin`.
- **Deleting the old files immediately** — would break `from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModel` and similar deep imports we can't see in third-party code. The shim is cheap and reversible.
- **Re-exporting via `__getattr__` on the old module** — works, but harder to attach a per-class warning to and confuses static analyzers / IDEs. Subclass + `__init__` warning is clearer.
- **Skipping shims for "internal-only" classes (`AceStepLyricEncoder`, `AceStepTimbreEncoder`)** — rejected. Even without an `__init__.py` re-export, third-party code may import them directly from the modeling module. The shim cost is one extra subclass; the breakage risk is real. Shim them.

## Validation checklist (run once before pushing the PR; spot-check after each commit)

- [ ] `make style && make quality` clean.
- [ ] `make fix-copies` regenerates dummies with no leftover diff.
- [ ] For every moved class: `python -c "from <old.dotted.path> import <ClassName>; <ClassName>()"` emits the deprecation warning but does not raise (the `__init__` may require args — adapt to a real construction or use `inspect.signature` to assert importability).
- [ ] For every moved class: `python -c "from <new.dotted.path> import <ClassName>"` succeeds with no warning.
- [ ] For every class previously in the top-level `__init__`: `python -c "from diffusers import <ClassName>"` still works.
- [ ] Add a dedicated **loading-only** test file (e.g. `tests/models/test_relocated_class_loading.py`) that does a tiny `from_pretrained` call against a published checkpoint for each moved class — small models, no inference, no slow-marker — purely to confirm config resolution still works after the move. `_class_name` in saved configs resolves to the class name (not the import path), so loading should work transparently as long as the new class is reachable from `diffusers.<subdir>`; the test exists to catch the case where it isn't. Keeping these in their own file keeps the per-pipeline test suites untouched and the loading check easy to delete once the shims are removed.
