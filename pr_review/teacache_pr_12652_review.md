# TeaCache PR #12652 Review Notes

## PR Overview

- **PR**: https://github.com/huggingface/diffusers/pull/12652
- **Title**: Implement TeaCache
- **Author**: LawJarp-A (Prajwal A)
- **Status**: Open
- **Changes**: +1335 / -22 lines across 6 files

### What is TeaCache?

[TeaCache](https://huggingface.co/papers/2411.19108) (Timestep Embedding Aware Cache) is a training-free caching technique that speeds up diffusion model inference by **1.5x-2.6x** by reusing transformer block computations when consecutive timestep embeddings are similar.

### Algorithm

1. Extract modulated input from first transformer block (after norm1 + timestep embedding)
2. Compute relative L1 distance vs previous timestep
3. Apply model-specific polynomial rescaling: `c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]`
4. Accumulate rescaled distance across timesteps
5. If accumulated < threshold → Reuse cached residual (FAST)
6. If accumulated >= threshold → Full transformer pass (SLOW, update cache)

---

## The Mid-Forward Intercept Problem

### Why TeaCache is Model-Specific

TeaCache needs to intercept **within** a model's forward method, not just at module boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│  Model Forward                                              │
│                                                             │
│  PREPROCESSING (must always run)                            │
│  ├── x_embedder(hidden_states)                              │
│  ├── time_text_embed(timestep, ...)                         │
│  └── context_embedder(encoder_hidden_states)                │
│                                                             │
│  ═══════════════════════════════════════════════════════════│
│  DECISION POINT ◄── TeaCache needs to intercept HERE        │
│  └── Extract: transformer_blocks[0].norm1(hs, temb)[0]      │
│  ═══════════════════════════════════════════════════════════│
│                                                             │
│  CACHEABLE REGION (can be skipped if cached)                │
│  ├── for block in transformer_blocks: ...                   │
│  └── for block in single_transformer_blocks: ...            │
│                                                             │
│  POSTPROCESSING (must always run)                           │
│  ├── norm_out(hidden_states, temb)                          │
│  └── proj_out(hidden_states)                                │
└─────────────────────────────────────────────────────────────┘
```

PyTorch hooks only intercept at **module boundaries** (before/after `forward()`), not within a forward method. The `for` loop over blocks is Python control flow - there's no hook point to skip it.

### Workaround: Custom Forward Replacement

The PR replaces the entire model forward with a custom implementation that has cache logic inserted at the right point. This works but requires maintaining separate forward functions for each model.

---

## Comparison of Caching Approaches

### TeaCache vs FirstBlockCache vs FasterCache

| Aspect | TeaCache | FirstBlockCache | FasterCache |
|--------|----------|-----------------|-------------|
| **Hook target** | Model forward | Transformer blocks | Attention layers |
| **Decision signal** | Modulated input (norm1 output) | Block output residual | Iteration count |
| **Where signal is** | Inside first block | Block boundary | Attention output |
| **Model-specific needs** | norm1 structure | Block output format | Attention class type |
| **Model-agnostic?** | ❌ No | ✅ Yes | ✅ Yes |

### Why FirstBlockCache is Model-Agnostic

FirstBlockCache uses the **first block's output residual** as its signal:

```python
# FirstBlockCache: hooks individual blocks
def new_forward(self, module, *args, **kwargs):
    original_hidden_states = args[0]
    output = self.fn_ref.original_forward(*args, **kwargs)  # Run block fully
    residual = output - original_hidden_states  # Signal from OUTPUT
    should_compute = self._compare_residual(residual)
    ...
```

It doesn't need to understand block internals - just input and output.

### Why FasterCache is Model-Agnostic

FasterCache hooks **attention layers** (not blocks) using class type checking:

```python
_ATTENTION_CLASSES = (Attention, MochiAttention, AttentionModuleMixin)

for name, submodule in module.named_modules():
    if isinstance(submodule, _ATTENTION_CLASSES):
        # Hook this attention module
```

All transformer models use standardized attention classes.

---

## Model Architecture Analysis

### Models That Fit TeaCache Pattern

Models with `norm1(hidden_states, temb)` returning modulated input:

| Model | norm1 Signature | Modulation Location | Single Residual |
|-------|----------------|---------------------|-----------------|
| FLUX 1 | `norm1(hs, emb=temb) → (tensor, gate)` | Inside norm1 | ✅ |
| FLUX Kontext | `norm1(hs, emb=temb) → (tensor, gate)` | Inside norm1 | ✅ |
| Mochi | `norm1(hs, temb) → (tensor, g, s, g)` | Inside norm1 | ✅ |
| Lumina2 | `norm1(hs, temb) → (tensor, gate)` | Inside norm1 | ✅ |

### Models That DON'T Fit Pattern

| Model | norm1 Signature | Modulation Location | Issue |
|-------|----------------|---------------------|-------|
| **FLUX 2** | `norm1(hs) → tensor` | Outside norm1 | Plain LayerNorm |
| **Wan** | `norm1(hs) → tensor` | Outside norm1 | Plain LayerNorm |
| **ZImage** | `attention_norm1(x) → tensor` | Outside norm1 | Plain LayerNorm |
| **CogVideoX** | N/A (uses `emb` directly) | N/A | Dual residual needed |

### FLUX 1 vs FLUX 2 Architecture Difference

**FLUX 1** (AdaLayerNorm - modulation inside):
```python
class FluxTransformerBlock:
    self.norm1 = AdaLayerNormZero(dim)  # Takes temb!

    def forward(self, hidden_states, temb, ...):
        norm_hs, gate = self.norm1(hidden_states, emb=temb)  # Modulation inside
```

**FLUX 2** (Plain LayerNorm - modulation outside):
```python
class Flux2TransformerBlock:
    self.norm1 = nn.LayerNorm(dim)  # NO temb!

    def forward(self, hidden_states, temb_mod_params_img, ...):
        (shift_msa, scale_msa, gate_msa), ... = temb_mod_params_img
        norm_hs = self.norm1(hidden_states)  # Plain norm
        norm_hs = (1 + scale_msa) * norm_hs + shift_msa  # Modulation outside
```

FLUX 2 follows the Wan/ZImage pattern and would need a separate custom forward.

---

## CogVideoX: The Architectural Outlier

CogVideoX has two unique requirements that don't fit the pattern:

### 1. Different Modulated Input Source

```python
# Other models: extract from norm1
modulated_inp = block.norm1(hidden_states, temb)[0]

# CogVideoX: uses timestep embedding directly
modulated_inp = emb  # Just the embedding, computed before blocks!
```

### 2. Dual Residual Caching

CogVideoX blocks return and modify TWO tensors:
```python
def forward(self, hidden_states, encoder_hidden_states, temb, ...):
    # Both are modified!
    return hidden_states, encoder_hidden_states
```

Requires caching two residuals:
```python
state.previous_residual = hs_output - hs_input
state.previous_residual_encoder = enc_output - enc_input  # Extra!
```

---

## Recommendations

### Simplification: FLUX-Only Support

Given the architectural diversity, recommend supporting only FLUX 1 and FLUX Kontext initially:

```python
_MODEL_CONFIG = {
    "FluxKontext": {
        "forward_func": _flux_teacache_forward,
        "coefficients": [-1.04655119e03, 3.12563399e02, -1.69500694e01, 4.10995971e-01, 3.74537863e-02],
    },
    "Flux": {
        "forward_func": _flux_teacache_forward,
        "coefficients": [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01],
    },
}
```

### What to Remove from PR

1. **CogVideoX support** - Dual residual architecture doesn't fit
2. **Mochi support** - Can be added later if needed
3. **Lumina2 support** - Can be added later if needed
4. **FLUX 2 support** - Different architecture (plain LayerNorm)

### Estimated Code Reduction

| Component | Original (PR) | FLUX-Only |
|-----------|---------------|-----------|
| Forward functions | 4 (~400 lines) | 1 (~100 lines) |
| Model configs | 10 entries | 2 entries |
| State fields | 8 | 5 |
| Utility functions | 6 | 3 |
| **Total teacache.py** | ~900 lines | ~350 lines |

### Simplified State

```python
class TeaCacheState(BaseState):
    def __init__(self):
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        # Removed: previous_residual_encoder (CogVideoX)
        # Removed: cache_dict (Lumina2)
        # Removed: uncond_seq_len (Lumina2)
```

---

## Why Custom Forwards Are Necessary

Despite the maintenance burden, custom forwards are the pragmatic approach for TeaCache because:

1. **Mid-forward intercept required** - Need to access `norm1` output before blocks run
2. **Architectural diversity** - Models differ in where/how modulation happens
3. **Block-level hooks insufficient** - Can't extract modulated input from block hooks
4. **Algorithm requirements** - TeaCache paper specifically uses modulated input as signal

### Alternative Approaches Considered

| Approach | Works? | Issue |
|----------|--------|-------|
| Block-level hooks (like FirstBlockCache) | ❌ | Can't access modulated input inside block |
| Attention-level hooks (like FasterCache) | ❌ | Different algorithm, not TeaCache |
| Hook norm1 directly | ⚠️ | norm1 interface varies per model |
| Hybrid (FirstBlockCache signal + TeaCache algorithm) | ⚠️ | Loses "optimal" signal per paper |

---

## PR Code Quality Issues (From Review)

1. **torch.compile incompatibility** - `.item()` calls in `_compute_rel_l1_distance` create graph breaks
2. **Boundary check bug** - `state.cnt == state.num_steps - 1` when `num_steps=0` evaluates to `-1`
3. **Incomplete Lumina2 state reset** - `cache_dict` and `uncond_seq_len` not reset
4. **Model auto-detection fragility** - Substring matching relies on iteration order

---

## Extension Path

If support for additional models is needed later:

1. **Mochi** - Same pattern as FLUX, just add coefficients and reuse `_flux_teacache_forward` or create similar
2. **Lumina2** - Same pattern but needs per-sequence-length caching for CFG
3. **FLUX 2 / Wan / ZImage** - Need separate forwards that extract modulated input differently
4. **CogVideoX** - Needs dual residual support, significant additional complexity

---

## Summary

- **TeaCache requires custom forwards** due to mid-forward intercept requirement
- **FLUX 1 + FLUX Kontext only** is the recommended scope for initial implementation
- **~60% code reduction** possible by removing unsupported models
- **Clear extension path** for adding models later as needed
- **Maintenance burden** is acceptable given the architectural constraints
