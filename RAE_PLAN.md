# RAE Plan (diffusers `rae` branch)

This plan tracks concrete fixes/additions for `AutoencoderRAE`, grounded in:
- current implementation: `src/diffusers/models/autoencoders/autoencoder_rae.py`
- original RAE codebase: `../RAE/src/stage1/rae.py`, `../RAE/src/stage1/encoders/*`, `../RAE/src/stage1/decoders/decoder.py`
- older branch baseline: `rae_orig`

## What was checked

- Current branch implementation and tests:
  - `src/diffusers/models/autoencoders/autoencoder_rae.py`
  - `tests/models/autoencoders/test_models_autoencoder_rae.py`
- Original implementation references:
  - `../RAE/src/stage1/rae.py`
  - `../RAE/src/stage1/encoders/dinov2.py`
  - `../RAE/src/stage1/encoders/siglip2.py`
  - `../RAE/src/stage1/encoders/mae.py`
  - `../RAE/src/stage1/decoders/decoder.py`
- `rae_orig` implementation and tests:
  - `rae_orig:src/diffusers/models/autoencoders/autoencoder_rae.py`
  - `rae_orig:tests/models/autoencoders/test_models_autoencoder_rae.py`

## Status

### Completed

1. Fix encoder default-path mismatch by encoder type.
- Done in `src/diffusers/models/autoencoders/autoencoder_rae.py`:
  - `encoder_name_or_path` now defaults to `None`.
  - Added per-encoder default map (`dinov2`, `siglip2`, `mae`), matching original intent in `../RAE/src/stage1/encoders/*.py`.

2. Freeze MAE encoder consistently (and align encoder freezing behavior).
- Done in `src/diffusers/models/autoencoders/autoencoder_rae.py`:
  - Added `requires_grad_(False)` for MAE and SigLIP2 encoders.
  - Added `@torch.no_grad()` on `MAEEncoder.forward`.
- Reference behavior: `../RAE/src/stage1/encoders/mae.py`, `../RAE/src/stage1/rae.py`.

3. Make latent stats handling robust with config + runtime buffers.
- Done in `src/diffusers/models/autoencoders/autoencoder_rae.py`:
  - Fixed config-vs-buffer name collision by using runtime buffers `_latents_mean` and `_latents_std`.
  - Added conversion helper for tensor/list/tuple stats inputs.
- Follow-up done:
  - Aligned public config names to `latents_mean` and `latents_std` (matching diffusers conventions).
  - Removed deprecated alias args to keep the new API clean within this PR.
- This fixes the constructor failure caught by new fast tests.

4. Restore fast unit coverage for RAE.
- Done in `tests/models/autoencoders/test_models_autoencoder_rae.py`:
  - Added non-slow `AutoencoderRAETests` using a tiny local registered encoder.
  - Coverage includes:
    - encode/decode/forward shape contract
    - scaling factor roundtrip
    - latent normalization math
    - slicing parity
    - noise behavior in train vs eval
- Result: non-slow subset passes (`5 passed, 4 skipped`).

5. Tighten processor fallback behavior and make it visible.
- Done in `src/diffusers/models/autoencoders/autoencoder_rae.py`:
  - narrowed fallback exception handling to expected load errors
  - added warning log when default mean/std fallback is used

6. Implement `use_encoder_loss` training path following diffusers API conventions.
- Done:
  - Kept `AutoencoderRAE.forward` aligned with other diffusers autoencoders (no model-level `return_loss` API).
  - Moved reconstruction and optional encoder-feature loss computation into the training script:
    `examples/research_projects/autoencoder_rae/train_autoencoder_rae.py`.
  - Encoder forward supports optional gradient-to-input for feature consistency loss in training scripts.

7. Add a diffusers-style RAE training example script.
- Done:
  - `examples/research_projects/autoencoder_rae/train_autoencoder_rae.py`
  - `examples/research_projects/autoencoder_rae/README.md`
- Script follows stage-1 convention: frozen encoder + train decoder + reconstruction loss + optional encoder feature loss.

### Remaining

1. Ensure hub collection compatibility in converter/loading utilities.
- Evidence: `nyu-visionx/RAE-collections` mixes decoder filenames (`model.pt` and `dinov2_decoder.pt`) and dataset casing (`imagenet1k` vs `ImageNet1k`).
- Action:
  - Add robust filename and folder-case resolution in conversion/loading script.
  - Add tests for path discovery logic.

## Secondary TODOs (Open)

1. Add dedicated unit tests for converter helper functions.
- Cover `resolve_decoder_file`, `resolve_stats_file`, and `extract_latent_stats`.

2. Expand docs with end-to-end conversion/training snippets.
- Add a short “convert then finetune” flow using `scripts/convert_rae_to_diffusers.py`.

3. Optional: align naming for consistency.
- Current uses `encoder_cls`; `rae_orig` used `encoder_type`; original `../RAE` used class-name registry.
- Action:
  - Keep one public name and support alias for backward compatibility.

## Suggested implementation order (remaining work)

1. Hub collection compatibility helper/converter.
2. Docs and model API page for `AutoencoderRAE`.
3. Minor cleanup (`use_encoder_loss`, naming aliases).

## Acceptance criteria

- Non-slow RAE tests run in CI and cover core encode/decode behavior.
- All three encoders work with defaults and do not accidentally train encoder weights.
- `save_pretrained` / `from_pretrained` roundtrip works with latent stats.
- Loading/conversion from RAE collection is resilient to known filename/casing variants.
