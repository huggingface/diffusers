# Unified Latents (UL) Implementation Plan for Diffusers

This plan describes how to implement the method from `2602.17270.md` in a Diffusers-native way, with an incremental path from research script to reusable library components.

## Brief Method Summary

Unified Latents (UL) jointly trains three components: a deterministic encoder, a latent diffusion prior, and a diffusion decoder. The encoder produces `z_clean`, then a fixed amount of Gaussian noise is added to produce `z0`; this fixed encoder noise is explicitly tied to the prior's maximum precision (`lambda_z(0)=5` in the paper). Training uses two diffusion losses:

- Prior loss `L_z` in latent space with unweighted ELBO-style denoising (`w=1`), so the latent bitrate is directly regularized by the diffusion prior.
- Decoder loss `L_x` in image space conditioned on `z0`, with sigmoid reweighting and a decoder loss factor `c_lf` to control reconstruction-vs-modeling tradeoff and avoid posterior collapse.

Stage 1 trains encoder + prior + decoder jointly with `L = L_z + L_x`. Stage 2 freezes the encoder and retrains a latent base diffusion model (with sigmoid weighting) for improved generation quality from sampled latents.

## Confirmed Author Clarifications

The following items were confirmed by the paper authors and should be used as implementation ground truth:

1. L2 reduction and bitrate units:
- Use true squared sums in denoising terms: `||.||^2 = sum_{c,h,w}(...)^2`.
- Compute bits-per-pixel by summing all terms and dividing full loss by `num_pixels * ln(2)`.
- With decoder terms removed, latent-only terms correspond to latent bitrate (latent bpp).

2. Stage-1 decoder objective:
- Use sigmoid-weighted ELBO with ELBO prefactor:
- `0.5 * exp(lambda) * (-d lambda / dt) * w(lambda) * ||x - x_hat||^2`
- and `w(lambda) = sigmoid(lambda - b)`.

3. Stage-2 base objective:
- Also use sigmoid-weighted ELBO (same principle as stage 1).
- Tuning note: smaller sigmoid bias is generally preferred for base models, especially at higher resolution.

4. Stage-2 target and sampling:
- Train against clean latent (encoder mean, `z_clean`) to reduce variance.
- During sampling, stop when reaching `logsnr_0` because decoder conditioning expects noisy latent at that endpoint.
- Both v-prediction and direct `z0` prediction can work; direct `z0` tends to behave worse at initialization.
- Preferred choice: v-prediction (flow-matching style velocity prediction is expected to work similarly).

5. Figure-3 epsilon/x-space equivalence:
- `0.5 * exp(lambda) * (-d lambda/dt) * w(lambda) * ||x - x_hat||^2`
- is equivalent to
- `0.5 * (-d lambda/dt) * w(lambda) * ||epsilon - epsilon_hat||^2`.
- The paper uses x-space in practice for convenience in math/numerics.

## Paper Section 5.1 Architecture (Target)

Per `2602.17270.md` Section 5.1, the reference architecture is:

- Encoder: ResNet with channels `[128, 256, 512, 512]`, with 2 residual blocks in downsampling stages and 3 blocks in final stage.
- Prior model (stage 1): single-level ViT with 8 blocks and 1024 channels.
- Base model (stage 2): 2-stage ViT with channels `[512, 1024]` and blocks `[6, 16]`, dropout `0.1`.
- Decoder: UViT with conv down/up channels `[128, 256, 512]`, middle transformer with 8 blocks and 1024 channels, dropout `0.1`.

Planned Diffusers mapping (as close as possible in this repo):

- Encoder: VAE-style `Encoder` backbone configured to `[128, 256, 512, 512]` and deterministic latent head.
- Prior: `DiTTransformer2DModel` configured to 8 layers and width 1024-equivalent (`heads * head_dim = 1024`).
- Decoder: UViT-style approximation with conv down/up + attention blocks and `[128, 256, 512]`, dropout `0.1`.
- If exact UViT conditioning path is unavailable, use a concat-conditioned conv+attention UNet approximation while preserving the channel/dropout profile above.

## Scope

Implement UL with two stages:

1. Stage 1 (joint latent learning):
- Encoder `E(x) -> z_clean`
- Prior diffusion model on latents (`z_t -> z_clean`) with **unweighted ELBO-style MSE** and true squared-sum reduction
- Diffusion decoder (`x_t, z0 -> x`) with **sigmoid-weighted ELBO MSE** (`w(lambda)=sigmoid(lambda-b)`) and true squared-sum reduction
- Fixed encoder noise linked to prior max precision (`lambda_z(0)=5`, i.e. `z0 = alpha0 * z_clean + sigma0 * eps`)

2. Stage 2 (base model on frozen latents):
- Freeze encoder (and optionally decoder)
- Train base model with sigmoid-weighted ELBO using clean-latent target (`z_clean`) and preferred v-prediction parameterization

## Paper-to-Diffusers Mapping

Core equations in `2602.17270.md` map to:

- Prior loss `L_z`: standard denoising MSE over latent noise levels with `w(lambda)=1`
- Decoder loss `L_x`: denoising MSE in image space with `w(lambda)=sigmoid(lambda - b)` (optionally scaled by `c_lf`), including ELBO prefactor
- Total stage-1 objective: `L = L_z + L_x`

Implementation-level mapping:

- Encoder: `ModelMixin` autoencoder encoder path (deterministic output `z_clean`)
- Prior denoiser: transformer/UNet-style model operating in latent space
- Decoder denoiser: conditional image diffusion model conditioned on `z0`
- Schedulers: custom logSNR-aware training utilities to compute `alpha(t), sigma(t), lambda(t), d lambda / dt` and per-sample weights

## Proposed Deliverables

1. Training utilities for UL losses/schedules.
2. Research training scripts for stage 1 and stage 2.
3. Optional reusable UL model wrappers (if we decide to productize in core API).
4. Tests (math/unit/smoke) and docs.

## Phase 0: Design + API Decisions

Decide minimal architecture choices before coding:

- Latent shape/compression ratio (e.g. 16x downsample, channel count)
- Prior backbone (`Transformer2DModel` vs `UNet2DModel` in latent space)
- Decoder backbone (`UNet2DConditionModel`-style conditioning vs custom UViT-like module)
- Conditioning injection strategy for `z0` in decoder (concat, cross-attn, FiLM)
- Schedule parameterization (continuous `t in [0,1]` vs discretized timesteps with logSNR lookup)

Output:
- frozen config schema for stage-1/stage-2 scripts (`configs/ul/*.yaml` or argparse equivalent)

## Phase 1: Core UL Math Utilities

Add shared training helpers (likely in `src/diffusers/training_utils.py` or a new UL helper module):

- `sample_t(batch, device, antithetic=False)`
- `logsnr_schedule(t, schedule_type, lambda_min, lambda_max)`
- `alpha_sigma_from_logsnr(lambda_t)`
- `decoder_weight(lambda_t, bias_b, loss_factor, mode="sigmoid")`
- `prior_weight(lambda_t)` returning ones
- Optional exact/approx `d lambda / dt` utilities for ELBO-consistent scaling

Acceptance checks:
- Unit tests for monotonic schedule behavior and alpha/sigma identities (`alpha^2 + sigma^2 ~= 1`)
- Weighting tests for expected limits at low/high noise

## Phase 2: Stage-1 Training Script (MVP)

Add `examples/research_projects/unified_latents/train_ul_stage1.py`:

Per step:
1. Encode image: `z_clean = E(x)`
2. Prior branch:
- sample `t_z, eps_z`
- build `z_t = alpha_z(t_z) * z_clean + sigma_z(t_z) * eps_z`
- predict `z_clean_hat`
- compute `L_z`
3. Decoder branch:
- sample `eps0`, make `z0 = alpha_z(0) * z_clean + sigma_z(0) * eps0`
- sample `t_x, eps_x`, make `x_t`
- predict `x_hat = D(x_t, z0, t_x)`
- compute `L_x` with sigmoid weighting and `c_lf`
4. Optimize `L = L_z + L_x`

Additional requirements:
- EMA for trainable modules
- mixed precision + accelerate
- checkpointing for encoder/prior/decoder (separable and joint)
- logging of proxy bitrate metrics from prior KL upper-bound terms

Acceptance checks:
- script runs on a tiny dataset split
- losses decrease for both branches
- reconstruction sample grid is produced

## Phase 3: Stage-2 Base Latent Model Training

Add `examples/research_projects/unified_latents/train_ul_stage2_base.py`:

- Load and freeze stage-1 encoder
- Build latent dataset on-the-fly from `z_clean` with fixed-noise forward process tied to `lambda_z(0)`
- Train base latent diffusion model with sigmoid-weighted ELBO, using clean-latent training target and preferred v-prediction parameterization
- Keep max logSNR tied to stage-1 prior (`lambda_z(0)=5`)

Acceptance checks:
- stage-2 model samples latents stably
- decoder + stage-2 samples produce valid images

## Phase 4: Inference / Sampling Pipeline

Add `examples/research_projects/unified_latents/sample_ul.py` first, then optional pipeline:

Sampling sequence:
1. sample `z1 ~ N(0, I)`
2. sample latents via stage-2 base model and stop at `logsnr_0` (decoder conditioning endpoint)
3. sample image with decoder conditioned on `z0`

Optional productized pipeline:
- `src/diffusers/pipelines/unified_latents/pipeline_unified_latents.py`
- components: `base_model`, `decoder_model`, schedulers, optional tokenizer/text conditioner for T2I extension

Acceptance checks:
- end-to-end generation script works from saved checkpoints
- deterministic outputs with fixed seed

## Phase 5: Model/API Productization (Optional but Recommended)

If we want reusable library APIs instead of only research scripts:

- Add `AutoencoderUL` (encoder + diffusion decoder interface helpers)
- Add `ULPriorModel`/`ULBaseModel` wrappers or document approved generic backbones
- Add config + `save_pretrained/from_pretrained` coverage

Keep this phase after MVP to reduce integration risk.

## Phase 6: Tests

Add targeted tests under `tests/`:

- `tests/training/test_ul_weighting.py`: schedule + weights math
- `tests/models/autoencoders/test_autoencoder_ul.py` (if productized)
- `tests/pipelines/unified_latents/test_pipeline_unified_latents.py` (if productized)
- smoke tests for stage-1/2 scripts with tiny random tensors

Minimum CI bar:
- UL math tests + at least one stage-1 smoke test

## Phase 7: Documentation

Add `examples/research_projects/unified_latents/README.md` covering:

- method summary and equations used
- stage-1 and stage-2 command lines
- expected checkpoints and how to sample
- recommended hyperparameter ranges:
- `lambda_z(0)=5`
- `c_lf` roughly `1.3-1.7`
- sigmoid bias `b` as bitrate/reconstruction tradeoff knob

## Suggested File Layout

- `examples/research_projects/unified_latents/train_ul_stage1.py`
- `examples/research_projects/unified_latents/train_ul_stage2_base.py`
- `examples/research_projects/unified_latents/sample_ul.py`
- `examples/research_projects/unified_latents/README.md`
- `src/diffusers/training_utils.py` (or `src/diffusers/utils/unified_latents.py`) for UL math helpers
- optional: `src/diffusers/pipelines/unified_latents/pipeline_unified_latents.py`
- optional: UL model classes under `src/diffusers/models/...`

## Risks and Mitigations

- Objective mismatch (predicting `x0` vs `eps` vs `v` across branches):
  - enforce one parameterization per branch and centralize conversion helpers.
- Posterior collapse in stage-1:
  - monitor prior/decoder loss ratio; tune `c_lf` and sigmoid bias `b`.
- Training instability from schedule scaling:
  - start with stable VP schedule and validated logSNR bounds.
- Over-committing to core API too early:
  - ship research scripts first, then promote stable abstractions.

## Milestones

1. M1 (MVP math + stage-1 script): UL loss implemented and trains on small run.
2. M2 (stage-2 + sampling): end-to-end UL sample generation from checkpoints.
3. M3 (tests + docs): reproducible instructions and CI smoke coverage.
4. M4 (optional): first-class Diffusers pipeline/model integration.

## Immediate Next Step

Implement Phase 1 + Phase 2 first (UL math helpers + `train_ul_stage1.py`), since this is the fastest path to validate correctness before broader API integration.
