# SUPIR Pipeline Design Document

Status: scaffold landed, full implementation pending. Tracks
[huggingface/diffusers#7219](https://github.com/huggingface/diffusers/issues/7219).

## What is SUPIR

SUPIR (Scaling Up to Excellence) is an image restoration / super-resolution
system from Yu et al. that combines a large generative prior (SDXL), a
degradation-robust encoder, a trimmed ControlNet-style adaptor with a
ZeroSFT connector, and a restoration-guided sampler. Optional caption
guidance is sourced from an external multi-modal LLM (LLaVA in the
reference implementation).

Reference materials:

- Paper: https://arxiv.org/abs/2401.13627
- Project page: https://supir.xpixel.group/
- Reference repo: https://github.com/Fanghua-Yu/SUPIR
- Pretrained weights / model card: https://huggingface.co/camenduru/SUPIR

## Why a scaffold first

The full SUPIR pipeline is large. It introduces new modules
(`GLVControl`, `LightGLVUNet`, `ZeroSFT`), a fine-tuned VAE encoder, a
custom EDM-style sampler, and a non-trivial weight-conversion path from
the upstream `.ckpt` files into diffusers `from_pretrained` layout.
Landing the public API surface first lets us:

- expose `SUPIRPipeline` in diffusers' import structure (and dummy
  fallbacks), so docs / typing / downstream packaging can refer to it;
- add gating tests that lock the `__call__` argument shape;
- decouple module porting work from API churn;
- give contributors clear seams to fill in (the helper stubs in
  `pipeline_supir.py` map 1:1 to the planned modules below).

The scaffold raises `NotImplementedError` for any path that requires the
real model. It is not a working pipeline.

## Planned components and porting plan

The reference repo lives in
[Fanghua-Yu/SUPIR/SUPIR/modules](https://github.com/Fanghua-Yu/SUPIR/tree/master/SUPIR).
Mapping each piece into diffusers:

### Stage 1 - degradation-robust encoder

- Source: fine-tuned SDXL VAE encoder. Reference repo loads it from
  `SUPIR/SUPIR_v0_Q_F.ckpt` together with the standard SDXL VAE decoder.
- Diffusers home: reuse `AutoencoderKL`. Add an optional
  `from_single_file` weight-conversion script under
  `scripts/convert_supir_to_diffusers.py` that splits the SUPIR
  checkpoint into the standard `vae` / `unet` / `controlnet` subfolders.
- Pipeline integration: implemented in
  `SUPIRPipeline.prepare_low_quality_latents`.

### Stage 2 - trimmed ControlNet adaptor with ZeroSFT

- Source: `GLVControl` and `ZeroSFT` in
  [`SUPIR_v0.py`](https://github.com/Fanghua-Yu/SUPIR/blob/master/SUPIR/modules/SUPIR_v0.py).
- Two options:
  1. introduce a new `SUPIRControlNetModel` under
     `src/diffusers/models/controlnets/supir.py` that subclasses
     `ControlNetModel`, replaces the encoder block ViT layers with the
     trimmed variant, and swaps the residual injection for ZeroSFT;
  2. keep `ControlNetModel` and add a `ZeroSFTBlock` mixin applied at
     load time. Option 1 is preferred for clarity.
- Pipeline integration: registered under `controlnet=` in `__init__`.
  The scaffold currently types this as `ControlNetModel` to keep the
  public surface stable; this will be widened to a union with the new
  type once option 1 lands.

### Stage 3 - generative prior (SDXL UNet)

- No model changes. Reuse `UNet2DConditionModel` and the dual-encoder
  text path from `StableDiffusionXLPipeline`. The scaffold's
  `encode_prompt` stub will be filled in by porting that method.

### Stage 4 - restoration-guided sampler

- Source: SUPIR's modified EDM sampler (paper section 3.4) - LQ-anchored
  guidance plus EDM stochasticity (`s_churn`, `s_noise`).
- Diffusers home: a new `restoration_guided_step` helper on the
  pipeline (already stubbed). The base scheduler stays a
  `KarrasDiffusionSchedulers` instance; SUPIR wraps each step rather
  than replacing the scheduler.

### Stage 5 - optional caption guidance

- The reference repo invokes LLaVA out-of-process to caption the LQ
  image and feed the result into the SDXL prompt path.
- Diffusers home: keep this **out** of `SUPIRPipeline` itself. Provide
  a small helper in `examples/community/supir_llava_caption.py` so
  users opt in. The pipeline only consumes the resulting `prompt`.

## Weight conversion

- `SUPIR/SUPIR_v0_Q_F.ckpt` and `SUPIR_v0_Q.ckpt` (Stage I and Stage II
  checkpoints) need to be mapped onto the diffusers folder layout.
- Plan: add `scripts/convert_supir_to_diffusers.py` that:
  1. loads the upstream checkpoint with `safetensors`;
  2. emits the SDXL VAE/UNet shards untouched (they match the public
     SDXL release);
  3. converts the SUPIR-specific ControlNet+ZeroSFT weights into the
     keys expected by `SUPIRControlNetModel`;
  4. writes a `model_index.json` referencing the SUPIR pipeline class.
- Out of scope for the scaffold PR.

## Testing strategy

- Scaffold PR: a single `tests/pipelines/supir/test_supir.py` module
  that pins the `__call__` argument shape (so future implementation
  PRs cannot silently break the public API) and `xfail`s the actual
  inference path until it lands.
- Implementation PR(s): add the standard pipeline test matrix
  (`PipelineTesterMixin`), a slow integration test against the
  upstream weights, and a tiled-inference test for >= 1024 px inputs.

## Open questions

- Does diffusers want the trimmed-ControlNet variant living under
  `models/controlnets/` (alongside `ControlNetModel`) or namespaced to
  `models/supir/`?
- Should the LLaVA caption path live in `examples/community/` (current
  plan) or as a `from_pretrained` hook on the pipeline?
- The reference repo conditions on negative prompts plus an
  EDM-derived "clean prompt" - we need to decide whether to expose the
  clean prompt as a separate `__call__` argument or fold it into
  `prompt_2`.

These are tracked on the GitHub issue.
