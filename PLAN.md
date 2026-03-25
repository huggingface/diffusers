# Plan: Adding Discrete Token Diffusion (UNI-D2-style) to Diffusers

This plan captures how to natively add support for **discrete diffusion models over tokens** (LLM-style) in `diffusers`, using UNI-D2’s modular structure as a guide while staying idiomatic to the `diffusers` ecosystem.

## What UNI-D2 Provides (Useful Structure)

UNI-D2 splits discrete diffusion into interchangeable components:

- **Algorithm / Trainer**: loss definition + orchestration (LightningModule)
- **ForwardProcess**: forward corruption kernel `q(x_t | x_0)`
- **NoiseSchedule**: continuous-time schedule providing `α(t)` and `α'(t)`
- **Backbone model**: predicts denoising distribution from `(x_t, t/σ)`
- **Sampler**: reverse process (generation) from noise → data

References:
- UNI-D2 architecture description: `3rd_party/UNI-D2/docs/architecture.md:9`
- UNI-D2 trainer base as LightningModule: `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/base.py:39`

## What Diffusers Already Has (Precedent for Discrete/Categorical Diffusion)

Diffusers already contains discrete/categorical diffusion components that can be adapted for tokens:

- `VQDiffusionScheduler` implements a categorical diffusion scheduler (masked class, categorical transitions, gumbel sampling).
  - Location: `src/diffusers/schedulers/scheduling_vq_diffusion.py:106`
- A (deprecated) pipeline shows how a discrete scheduler integrates into a `DiffusionPipeline` sampling loop:
  - Location: `src/diffusers/pipelines/deprecated/vq_diffusion/pipeline_vq_diffusion.py:52`

These are strong templates for a **text-token discrete diffusion** pipeline + scheduler.

## UNI-D2 → Diffusers Mapping (Native Design)

UNI-D2 abstractions map cleanly onto diffusers concepts:

- **NoiseSchedule** → scheduler configuration + schedule utilities (`α(t)`, optional `α'(t)`), or precomputed per-step arrays.
- **ForwardProcess** → scheduler `add_noise(...)` for token IDs (training-time forward corruption).
- **Sampler** → scheduler `step(...)` + pipeline denoising loop (inference-time reverse chain).
- **Algorithm/Trainer** → in diffusers, keep core library focused on inference primitives; provide trainers as `accelerate`-based examples + reusable loss helpers (diffusers doesn’t use Lightning internally).

## Current Status (Implemented)

- Scheduler:
  - `src/diffusers/schedulers/scheduling_token_diffusion.py` (`TokenDiffusionScheduler`)
  - `set_timesteps`, `add_noise`, `step`, `save_pretrained/from_pretrained`
  - Forward process modes: `absorbing`, `uniform` (optional `exclude_mask_from_uniform`)
  - Alpha schedules: `log_linear`, `linear`, `cosine`, `geometric`
  - Schedule-aware MDLM weighting: `get_mdlm_loss_weights(...)`
  - Final “noise removal” behavior on the last step (`alpha_prev=1`)
- Additional schedulers:
  - `src/diffusers/schedulers/scheduling_block_token_diffusion.py` (`BlockTokenDiffusionScheduler`)
  - `src/diffusers/schedulers/scheduling_hybrid_token_diffusion.py` (`HybridTokenDiffusionScheduler`)
- Pipeline:
  - `src/diffusers/pipelines/token_diffusion/pipeline_token_diffusion.py` (`TokenDiffusionPipeline`)
  - Passes `timesteps` and `return_dict=True` to the model each step (like standard diffusers pipelines pass `t` to UNet)
  - Supports both scheduler modes, optional decoding via tokenizer
  - Conditioning: `prefix_ids` and `infill_mask`
- Additional pipelines:
  - `src/diffusers/pipelines/block_token_diffusion/pipeline_block_token_diffusion.py` (`BlockTokenDiffusionPipeline`)
  - `src/diffusers/pipelines/hybrid_token_diffusion/pipeline_hybrid_token_diffusion.py` (`HybridTokenDiffusionPipeline`)
- Examples:
  - Training: `examples/discrete_diffusion/train_mdlm.py`, `examples/discrete_diffusion/train_udlm.py`, `examples/discrete_diffusion/train_hybrid_token_diffusion.py`
  - Sampling: `examples/discrete_diffusion/sample_mdlm.py` (uses `TokenDiffusionPipeline`), `examples/discrete_diffusion/sample_block_token_diffusion.py`, `examples/discrete_diffusion/sample_hybrid_token_diffusion.py`
  - All sampling scripts use their respective pipeline `__call__()` (no manual denoising loops)
  - Docs: `examples/discrete_diffusion/README.md`
- Tests:
  - Scheduler tests: `tests/schedulers/test_scheduler_token_diffusion.py`
  - Pipeline tests: `tests/pipelines/test_pipeline_token_diffusion.py`
  - Block/hybrid tests: `tests/schedulers/test_scheduler_block_token_diffusion.py`, `tests/schedulers/test_scheduler_hybrid_token_diffusion.py`
  - Block/hybrid pipeline tests: `tests/pipelines/test_pipeline_block_token_diffusion.py`, `tests/pipelines/test_pipeline_hybrid_token_diffusion.py`
- Training utilities:
  - Confidence-aware loss helper: `src/diffusers/training_utils.py` (`compute_confidence_aware_loss`)
- Block refinement (commit-by-confidence):
  - Pipeline: `src/diffusers/pipelines/block_refinement/pipeline_block_refinement.py` (`BlockRefinementPipeline`)
  - Supports `attention_mask_mode` (`auto` tries 4D additive mask then falls back to 2D padding mask)
  - Examples: `examples/discrete_diffusion/train_block_refinement_cap.py`, `examples/discrete_diffusion/sample_block_refinement.py`
  - Tests: `tests/pipelines/test_pipeline_block_refinement.py`
- LLaDA2 (block-wise iterative refinement for LLMs):
  - Pipeline: `src/diffusers/pipelines/llada2/pipeline_llada2.py` (`LLaDA2Pipeline`)
  - Diffusers-native implementation (does NOT wrap model.generate())
  - Block-wise iterative refinement with confidence-based token selection
  - Block-diagonal causal attention mask for parallel decoding within blocks
  - Gumbel-max sampling for numerical stability (from UNI-D2 patterns)
  - Flexible attention mask handling (`auto`/`4d`/`2d`/`none` modes)
  - `editing_threshold` and `max_post_steps` support for post-mask token editing (LLaDA2.1 feature)
    - Functionally equivalent to official LLaDA2.1 `generate()` for default settings
    - Backward compatible: disabled by default (`editing_threshold=None`, `max_post_steps=0`)
  - `callback_on_step_end` support for step-level callbacks
  - Examples: `examples/discrete_diffusion/sample_llada2.py`
  - Compatible with LLaDA2 models from HuggingFace (e.g., `inclusionAI/LLaDA2.0-mini`, `inclusionAI/LLaDA2.1-mini`)
- DFlash (block diffusion speculative decoding):
  - Scheduler: `src/diffusers/schedulers/scheduling_dflash_token_diffusion.py` (`DFlashTokenDiffusionScheduler`)
  - Pipeline: `src/diffusers/pipelines/dflash/pipeline_dflash.py` (`DFlashPipeline`)
  - Diffusers-native denoising loop with target-hidden feature fusion
  - `callback_on_step_end` support for step-level callbacks
  - Example: `examples/discrete_diffusion/sample_dflash.py`
- SDAR (block diffusion with remasking):
  - Scheduler: `src/diffusers/schedulers/scheduling_sdar_token_diffusion.py` (`SDARTokenDiffusionScheduler`)
  - Pipeline: `src/diffusers/pipelines/sdar/pipeline_sdar.py` (`SDARPipeline`)
  - `callback_on_step_end` support for step-level callbacks
  - Example: `examples/discrete_diffusion/sample_sdar.py`
- BD3LM (block diffusion with first-hitting updates):
  - Scheduler: `src/diffusers/schedulers/scheduling_bd3lm_token_diffusion.py` (`BD3LMTokenDiffusionScheduler`)
  - Pipeline: `src/diffusers/pipelines/bd3lm/pipeline_bd3lm.py` (`BD3LMPipeline`)
  - `callback_on_step_end` support for step-level callbacks
  - Example: `examples/discrete_diffusion/sample_bd3lm.py`
  - **Note**: `BD3LMModel` is not yet exported from diffusers `__init__.py`; sample script imports fail

## End-to-End Sampling Verification

All sampling scripts use their respective pipeline's `__call__()` — no manual denoising loops.

| Script | Model | Pipeline | Status |
|--------|-------|----------|--------|
| `sample_mdlm.py` | `kuleshov-group/mdlm-owt` | `TokenDiffusionPipeline` | Verified |
| `sample_llada2.py` | `inclusionAI/LLaDA2.1-mini` | `LLaDA2Pipeline` | Verified (with editing) |
| `sample_llada2.py` | `inclusionAI/LLaDA2.0-mini` | `LLaDA2Pipeline` | Verified (backward compat) |
| `sample_dflash.py` | `z-lab/Qwen3-4B-DFlash-b16` + `Qwen/Qwen3-4B` | `DFlashPipeline` | Verified |
| `sample_sdar.py` | `JetLM/SDAR-1.7B-Chat` (revision `refs/pr/1`) | `SDARPipeline` | Verified |
| `sample_bd3lm.py` | `kuleshov-group/bd3lm-owt-block_size4` | `BD3LMPipeline` | Blocked (`BD3LMModel` not exported) |
| `sample_block_refinement.py` | (local checkpoint only) | `BlockRefinementPipeline` | No public model |
| `sample_block_token_diffusion.py` | (local checkpoint only) | `BlockTokenDiffusionPipeline` | No public model |
| `sample_hybrid_token_diffusion.py` | (local checkpoint only) | `HybridTokenDiffusionPipeline` | No public model |

### Official Code Parity Checks

- **LLaDA2 vs official `generate()`**: Functionally identical for default settings (`steps=block_length=32`). Editing logic, post-steps counting, and confidence thresholds all match. Our design adds bounded loop + transfer_schedule (converges identically).
- **DFlash vs official `spec_generate()`**: Functionally identical. Prefill, draft generation, target verification, acceptance via cumprod, KV cache cropping, and stop token handling all match exactly.
- **MDLM**: `TokenDiffusionPipeline` passes `timesteps` per step (like standard diffusers UNet pipelines pass `t`), matching official MDLM sampling.

### Design Note: Model Layer

We rely entirely on `transformers` for the model layer (`AutoModelForCausalLM`, `AutoModelForMaskedLM`, `AutoModel`). All model-specific features (architectures, attention, RoPE, etc.) are handled by `transformers` via `trust_remote_code=True`. Our pipelines are model-agnostic orchestrators that call `model(input_ids, ...).logits`.

## Design Insights from Reference Codebases (dllm-dev & UNI-D2)

### Architecture Comparison

| Concern | **dllm-dev** | **UNI-D2** | **diffusers (ours)** |
|---|---|---|---|
| Noise schedule | `Noise` → `(α_t, α'_t)` pair | `NoiseSchedule` module | `TokenDiffusionScheduler._alpha_t()` / `._alpha_prime_t()` |
| Forward process | Inline in `Denoiser._sample_q_xt()` | Separate `ForwardProcess` class | `TokenDiffusionScheduler.add_noise()` with `forward_process` config |
| Reverse step | Inline in denoiser generation | Separate `Sampler` class | `scheduler.step()` |
| Prior sampling | Inline in generation loop | `model.prior_sample()` | `scheduler.sample_prior()` |
| Training loss | `Denoiser._compute_loss()` | `Algorithm.nll_per_token()` | `scheduler.get_mdlm_loss_weights()` (weights only) |
| Pipeline | Monolithic `generate()` | Lightning `generate_samples()` | `DiffusionPipeline.__call__()` |
| Config system | Hydra YAML | Hydra YAML | `ConfigMixin` + `register_to_config` |

References:
- dllm-dev noise schedules: `3rd_party/dllm-dev/src/noise_schedule/noise_schedules.py`
- dllm-dev denoiser hierarchy: `3rd_party/dllm-dev/src/denoiser/base.py` (abstract), `3rd_party/dllm-dev/src/denoiser/diffusion.py` (MDLM/BD3LM/E2D2)
- UNI-D2 noise schedules: `3rd_party/UNI-D2/src/discrete_diffusion/noise_schedules/`
- UNI-D2 forward processes: `3rd_party/UNI-D2/src/discrete_diffusion/forward_process/`
- UNI-D2 samplers: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/`
- UNI-D2 algorithms: `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/`
- Our scheduler: `src/diffusers/schedulers/scheduling_token_diffusion.py`
- Our mixin: `src/diffusers/pipelines/pipeline_utils.py` (class `DiscreteDiffusionPipelineMixin`, line ~2340)

### Noise Schedules — Fully Aligned

All three codebases implement essentially the same alpha schedules. Our four (`log_linear`, `linear`, `cosine`, `geometric`) cover the standard set. dllm-dev adds `ExponentialNoise` (α = 1 - t^exp) which is a minor variant.

Key design difference: dllm-dev and UNI-D2 return `(α_t, α'_t)` as a pair. Ours computes them separately. Both are fine.

### Forward Processes — Sufficient But Less Modular

UNI-D2 has separate `ForwardProcess` classes (`AbsorbingForwardProcess`, `UniformForwardProcess`, `BlockAbsorbingForwardProcess`), making it trivial to compose any schedule with any corruption kernel. We bundle this into `TokenDiffusionScheduler.add_noise()` with `forward_process` as a config string.

**Trade-off**: Our approach is simpler for users. Separate classes would only matter if we added exotic forward processes (SEDD, FlexMDM, CANDI hybrid). For absorbing + uniform, our design is sufficient.

References:
- UNI-D2 absorbing forward: `3rd_party/UNI-D2/src/discrete_diffusion/forward_process/absorbing.py`
- UNI-D2 uniform forward: `3rd_party/UNI-D2/src/discrete_diffusion/forward_process/uniform.py`
- Our add_noise: `src/diffusers/schedulers/scheduling_token_diffusion.py` (method `add_noise`, line ~336)

### Reverse Sampling Strategies — Key Differences

| Strategy | dllm-dev | UNI-D2 | diffusers |
|---|---|---|---|
| **Posterior** q(x_s\|x_t, x̂₀) | Default | `AbsorbingSampler` | `TokenDiffusionScheduler.step()` |
| **Predict-then-noise** | Yes (confidence + margin) | — | `BlockRefinementPipeline` (pipeline-level) |
| **Confidence-based remasking** | In predict-then-noise | — | `SDARTokenDiffusionScheduler.step()` (4 strategies) |
| **Speculative decoding** | — | — | `DFlashTokenDiffusionScheduler.step()` |
| **First-hitting time** | Yes | — | — |
| **Score-entropy (SEDD)** | — | `SEDDSampler` | — |
| **GIDD interpolation** | — | `GIDDSampler` | — |
| **Entropy-bounded** | — | `EBSampler` | `SDARTokenDiffusionScheduler` (entropy_bounded) |

References:
- dllm-dev posterior sampling: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (method `_compute_posterior`)
- dllm-dev predict-then-noise: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (method `_generate_unconditional`, search `predict_then_noise`)
- dllm-dev first-hitting: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (method `_sample_generation_timesteps`)
- UNI-D2 absorbing sampler: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/absorbing.py`
- UNI-D2 uniform sampler: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/uniform.py`
- UNI-D2 SEDD sampler: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/sedd.py`
- UNI-D2 GIDD sampler: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/gidd.py`
- Our posterior step: `src/diffusers/schedulers/scheduling_token_diffusion.py` (method `step`, line ~398)
- Our SDAR remasking: `src/diffusers/schedulers/scheduling_sdar_token_diffusion.py` (method `step`, line ~141)
- Our block refinement: `src/diffusers/pipelines/block_refinement/pipeline_block_refinement.py` (confidence/threshold logic in `__call__`)
- Our DFlash speculative: `src/diffusers/schedulers/scheduling_dflash_token_diffusion.py` (method `step`, line ~77)

### Model Output Processing Pattern

Both reference codebases explicitly post-process model output before sampling:

```python
# dllm-dev & UNI-D2 pattern:
model_output[:, :, mask_id] = -inf        # Never predict mask token
model_output[xt != mask_id] = -inf         # Unmasked positions → keep current token
model_output[xt != mask_id, xt] = 0.0      # Set identity logit to 0
```

Our `TokenDiffusionScheduler.step()` achieves the same via:
```python
logits[..., self.mask_token_id] = -inf     # Never predict mask token
is_masked = sample == self.mask_token_id
x_prev = torch.where(is_masked & should_denoise, sampled_x0, sample)  # Keep unmasked
```

Both approaches are correct. The reference approach of zeroing non-masked logits is more numerically explicit but ours is equivalent because we only apply updates to masked positions.

### Insights to Adopt

#### 1. Training Timestep Sampling Utility (from both)

Both codebases provide structured timestep sampling for training with optional antithetic (stratified) sampling for variance reduction. We could add:

```python
# On TokenDiffusionScheduler:
def sample_training_timesteps(self, batch_size, generator=None, antithetic=False):
    """Sample random training timesteps in [0, num_train_timesteps-1]."""
```

References:
- dllm-dev collator: `3rd_party/dllm-dev/src/datasets/collator.py` (class `DenoisingCollator`)
- UNI-D2 time sampling: `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/base.py` (method `_sample_t`)

#### 2. Confidence-Margin Remasking (from dllm-dev)

dllm-dev's predict-then-noise mode uses top-2 probability margin for remasking decisions: tokens where `|top1_prob - top2_prob|` is small are re-masked. This complements our existing SDAR strategies (`low_confidence_static`, `low_confidence_dynamic`, `sequential`, `entropy_bounded`) and could be added as a `"confidence_margin"` strategy.

Reference: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (search `confidence_margin_based_noising`)

#### 3. First-Hitting Time Schedule (from dllm-dev)

A special timestep schedule where each step targets changing exactly ~1 token (exponentially spaced). Could be an option in `set_timesteps()`:

```python
scheduler.set_timesteps(num_inference_steps, schedule="first_hitting")
```

Reference: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (method `_sample_generation_timesteps`, search `first_hitting`)

#### 4. σ(t) = -log(α(t)) Model Conditioning Convention (from both)

Both codebases condition the model on `σ(t) = -log(α(t))` rather than raw timestep index. This is a model-side convention (not scheduler), but we should document it. Our sample scripts already handle this: `sample_mdlm.py` passes `timesteps=sigma` to the model.

References:
- dllm-dev sigma: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (search `sigma` in `_prepare_inputs`)
- UNI-D2 sigma: `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/base.py` (method `_sigma_from_alphat`)

#### 5. Static Attention Mask Caching (from dllm-dev)

dllm-dev registers block attention masks as model buffers and dynamically resizes them. Our `BlockRefinementPipeline` and `SDARPipeline` rebuild masks each call. For repeated generation at the same sequence length, caching would save compute.

Reference: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (search `_create_static_mask`, `register_buffer`)

#### 6. Separate Sampler Abstraction (from UNI-D2) — Architectural Consideration

UNI-D2 separates `Sampler` from `NoiseSchedule`, making it easy to plug in `GIDDSampler`, `EBSampler`, etc. In diffusers, `scheduler.step()` is the single API. If we wanted to support multiple sampling strategies per scheduler, we could add a `sampling_strategy` config field rather than separate Sampler classes — this preserves the diffusers convention while enabling extensibility.

### What We Already Do Well

- **diffusers conventions**: `set_timesteps()`, `step()`, `sample_prior()`, `add_noise()`, `prepare_latents()`, `check_inputs()`, callbacks
- **DiscreteDiffusionPipelineMixin**: Shared SAR utilities (top-k, top-p, temperature sampling) — neither reference codebase has a clean shared mixin like this
- **Pre-computed alphas in `set_timesteps()`**: Matches dllm-dev's approach, avoids per-step recomputation
- **Block mask support via `block_mask` parameter**: Clean opt-in for block-wise diffusion without separate scheduler
- **Richest remasking strategy set**: Our `SDARTokenDiffusionScheduler` has 4 strategies; dllm-dev has 2, UNI-D2 has 1
- **Speculative decoding**: `DFlashPipeline` — neither reference codebase supports this

## Coverage Gaps vs Reference Codebases (Not Yet in Diffusers)

Combined gaps from both dllm-dev and UNI-D2:

### Sampling Strategies
- **First-hitting time schedule** (dllm-dev): Exponentially spaced timesteps targeting ~1 token change per step. Useful for token-by-token generation.
  - Ref: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (search `first_hitting`)
- **SEDD (Score Entropy Discrete Diffusion)** (UNI-D2): Score-based parameterization, different from substitution-based MDLM. Requires different `step()` logic.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/sedd.py`, `3rd_party/UNI-D2/src/discrete_diffusion/sampling/sedd.py`
- **GIDD (Generalized Interpolating Discrete Diffusion)** (UNI-D2): Interpolating sampler between different discrete processes.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/gidd.py`
- **Confidence-margin remasking** (dllm-dev): Uses `|top1_prob - top2_prob|` for remasking decisions.
  - Ref: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (search `confidence_margin_based_noising`)

### Algorithms / Methods
- **FlexMDM** (UNI-D2): Any-order generation with flexible masking.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/flexmdm.py`
- **Partition MDLM** (UNI-D2): Partition-based generation.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/partition.py`
- **CANDI** (UNI-D2): Hybrid discrete-continuous noise process.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/forward_process/candi.py`
- **E2D2** (dllm-dev): Encoder-decoder architecture for block diffusion (separate encoder/decoder LLMs with cross-attention).
  - Ref: `3rd_party/dllm-dev/src/denoiser/diffusion.py` (class `E2D2`, line ~1090), `3rd_party/dllm-dev/src/backbone/encoder_decoder.py`

### Training Utilities
- **Antithetic/stratified time sampling** (both): Variance reduction during training.
  - Ref: `3rd_party/dllm-dev/src/datasets/collator.py`, `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/base.py`
- **Block size / noise level annealing** (dllm-dev): Curriculum learning via Composer algorithms.
  - Ref: `3rd_party/dllm-dev/src/custom_composer/algorithms.py`

### Forward Processes
- **Block absorbing** (UNI-D2): Per-block timestamps for block diffusion training.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/forward_process/block_absorbing.py`
- **FlexMDM forward** (UNI-D2): Flexible masking patterns.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/forward_process/flexmdm.py`
- **CANDI hybrid** (UNI-D2): Continuous+discrete noise.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/forward_process/candi.py`

### Noise Schedules
- **Hybrid diffusion** (UNI-D2): Mixture of continuous/discrete schedule components.
  - Ref: `3rd_party/UNI-D2/src/discrete_diffusion/noise_schedules/hybrid.py`
- **Exponential** (dllm-dev): `α = 1 - t^exp` with configurable exponent.
  - Ref: `3rd_party/dllm-dev/src/noise_schedule/noise_schedules.py` (class `ExponentialNoise`)

## Next Steps (Immediate)

- Export `BD3LMModel` and `BD3LMPipeline` from `diffusers.__init__.py` so `sample_bd3lm.py` can run.
- Test `sample_bd3lm.py` end-to-end with `kuleshov-group/bd3lm-owt-block_size4`.
- SDAR model (`JetLM/SDAR-1.7B-Chat`): merge upstream PR #1 (remove `LossKwargs`) so `--revision refs/pr/1` is no longer needed.
- Run `make style` and `make quality` to ensure all changes pass linting.

## Phased Implementation Plan (Updated)

### Phase 1 — Define Discrete-Token Diffusion API Surface (Done)

Goal: introduce a minimal, composable API consistent with diffusers schedulers/pipelines.

- Add a first-class scheduler for categorical token diffusion:
  - `TokenDiffusionScheduler` (or `DiscreteTokenDiffusionScheduler`)
  - Required methods: `set_timesteps`, `step`, plus **`add_noise(input_ids, timesteps, ...)`** for training
- Define forward-process modes needed for initial coverage:
  - `absorbing` (mask token replacement)
  - `uniform` (random token replacement)

Deliverables:
- Scheduler interface + configuration fields (vocab size, mask token id, schedule params)
- Clear contract for model outputs (logits/log-probs, shape `[B, L, V]`)

### Phase 2 — Implement Noise Schedules + Forward Processes (Next)

Goal: mirror UNI-D2’s separability while remaining diffusers-native.

Options:
- **Option A (diffusers-native)**: implement schedules inside the scheduler (precomputed arrays or closed-form functions).
- **Option B (more UNI-D2-like)**: add lightweight helpers for `NoiseSchedule` and `ForwardProcess` under `src/diffusers/` (still driven by scheduler config).

Next concrete work:
- Extend `TokenDiffusionScheduler` beyond log-linear by adding more `alpha(t)` families (cosine/linear/geometric). (Done)
- Expose schedule choice as a scheduler config option (e.g. `alpha_schedule=...`) in diffusers style. (Done)
- Keep training examples explicit about schedule-dependent weighting to avoid silent mismatches. (Done)

### Phase 3 — Implement Additional Token Diffusion Scheduler(s) (Revised)

Observation from reviewing more methods:
- Several methods have materially different reverse dynamics and/or state (block-wise sampling, hybrid transition models, variable length), which will likely be cleaner as separate scheduler classes (and sometimes separate pipelines), instead of one “mega-scheduler”.

Next concrete work:
- Add a block-diffusion scheduler/pipeline pair for BD3LM-like behavior (block size, block updates, optional nucleus filtering).
- Add a hybrid scheduler class for GIDD-like behavior once the required transition helpers are formalized.
- Add separate families for score/rate-based methods (SEDD/FlexMDM) rather than forcing them into `TokenDiffusionScheduler`.
- Add CANDI and Partition-MDLM variants as dedicated schedulers/pipelines (reverse dynamics differ from MDLM/UDLM).

### Phase 4 — Add Token Diffusion Pipeline + Wrappers (Done for Generic)

Goal: a native pipeline that can generate token sequences via discrete diffusion.

- New pipeline: `src/diffusers/pipelines/token_diffusion/pipeline_token_diffusion.py`
- Components:
  - `tokenizer`
  - `backbone` (transformer producing logits over vocab)
  - `scheduler` (`TokenDiffusionScheduler`)
- Sampling loop:
  - Mirrors UNI-D2 absorbing sampler + the VQDiffusion pipeline structure
  - Supports BOS injection and optional caching strategies

Current scope:
- Unconditional generation, optional start-token injection
- Prefix conditioning and infill/fixed-position masking

### Phase 5 — Add Accelerate Trainer + Examples (In Progress)

Goal: deliver practical training recipes without adding Lightning as a core dependency.

Already added:
- Training loops in `examples/` for absorbing and uniform diffusion.

Next concrete work:
- Add method-specific training scripts as new schedulers land (BD3LM/GIDD/SEDD/FlexMDM).
- Optional: factor common pieces into lightweight helpers if repetition becomes significant.
- Add a block-refinement training+sampling example that uses confidence-aware loss and the block refinement pipeline.

### Phase 6 — Tests, Docs, and Serialization

Goal: ship maintainable, testable, save/load-able primitives.

- Unit tests for scheduler:
  - mask token handling and invariants
  - shape/dtype checks
  - determinism via `torch.Generator`
  - basic sampling loop sanity for absorbing
- Ensure `save_pretrained/from_pretrained` works for scheduler and pipeline components
- Add docs page: “Discrete token diffusion in diffusers”
  - Describe mapping schedule/forward/step
  - Provide minimal example usage + links to training scripts

## Clarifying Questions (To Lock the Initial Scope)

1) Should v1 target **MDLM absorbing-only** first, or include **UDLM/uniform** as well?
2) Do we require **time-conditioning** in the backbone from day one?
3) Should the first pipeline support **conditioning** (prefix/infill) or start **unconditional/BOS-only**?

## Proposed Next Steps (Priority Order)

1) ~~Add additional `alpha(t)` schedule types to `TokenDiffusionScheduler` (cosine/linear/geometric) with tests and example guidance.~~ (Done)
2) ~~Add LLaDA2-style block-wise iterative refinement pipeline.~~ (Done - `LLaDA2Pipeline`)
3) ~~Add BD3LM-style block diffusion as a separate scheduler/pipeline.~~ (Done - `BD3LMPipeline`)
4) ~~Add DFlash speculative decoding pipeline.~~ (Done - `DFlashPipeline`)
5) ~~Add SDAR block diffusion with remasking pipeline.~~ (Done - `SDARPipeline`)
6) ~~Create `DiscreteDiffusionPipelineMixin` with shared utilities.~~ (Done)
7) ~~Merge `BlockTokenDiffusionScheduler` into `TokenDiffusionScheduler`.~~ (Done - alias)
8) ~~Address all PR #12911 review comments from @dg845.~~ (Done)

### Recently Completed
9) ~~Add `editing_threshold` and `max_post_steps` to `BlockRefinementPipeline` / `LLaDA2Pipeline`.~~ (Done — matches official LLaDA2.1 behavior)
10) ~~Fix `TokenDiffusionPipeline` to pass `timesteps` and `return_dict=True` to model.~~ (Done — now works with `kuleshov-group/mdlm-owt`)
11) ~~Rewrite `sample_mdlm.py` to use `TokenDiffusionPipeline` instead of manual loop.~~ (Done)
12) ~~Verify all sampling scripts end-to-end with public HF models.~~ (Done — MDLM, LLaDA2, DFlash, SDAR all verified)

### Low-Hanging Fruit (from reference codebase analysis)
13) Export `BD3LMModel`/`BD3LMPipeline` from `__init__.py` and test `sample_bd3lm.py` with `kuleshov-group/bd3lm-owt-block_size4`.
14) Add `sample_training_timesteps(batch_size, generator, antithetic)` to `TokenDiffusionScheduler` — useful for training scripts, matches dllm-dev collator and UNI-D2 `_sample_t`.
15) Add `"confidence_margin"` remasking strategy to `SDARTokenDiffusionScheduler` — uses `|top1_prob - top2_prob|` from dllm-dev.

### Medium Effort / High Value
16) Add first-hitting time schedule option to `set_timesteps()` — exponentially spaced steps targeting ~1 token/step, from dllm-dev.
17) Add SEDD parameterization support — score-based instead of substitution, requires `parameterization` config field and branching in `step()`. Ref: `3rd_party/UNI-D2/src/discrete_diffusion/algorithms/sedd.py`.
18) Add GIDD-style hybrid sampler — interpolating between discrete processes. Ref: `3rd_party/UNI-D2/src/discrete_diffusion/sampling/gidd.py`.

### Longer Term
19) Add FlexMDM (any-order generation) as a separate scheduler/pipeline family.
20) Add CANDI hybrid discrete-continuous noise as a separate forward process.
21) Add E2D2 encoder-decoder architecture support (from dllm-dev) — separate encoder/decoder with cross-attention.
22) Add tests for `LLaDA2Pipeline`, `DFlashPipeline`, `SDARPipeline`, `BD3LMPipeline`.
23) Add training scripts for block refinement and SDAR models.
