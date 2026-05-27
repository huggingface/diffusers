# Model conventions and rules

Shared reference for model-related conventions, patterns, and gotchas.
Linked from `AGENTS.md`, `skills/model-integration/SKILL.md`, and `review-rules.md`.

## Coding style

- All layer calls should be visible directly in `forward` — avoid helper functions that hide `nn.Module` calls.
- Avoid graph breaks for `torch.compile` compatibility — do not insert NumPy operations in forward implementations and any other patterns that can break `torch.compile` compatibility with `fullgraph=True`.
- No new mandatory dependency without discussion (e.g. `einops`). Optional deps guarded with `is_X_available()` and a dummy in `utils/dummy_*.py`.

## Common model conventions

* Models use `ModelMixin` with `register_to_config` for config serialization. 
* When adding a new transformer (or reviewing one), skim `src/diffusers/models/transformers/transformer_flux.py`, `src/diffusers/models/transformers/transformer_flux2.py`, `src/diffusers/models/transformers/transformer_qwenimage.py`, and `src/diffusers/models/transformers/transformer_wan.py` first to establish the pattern. Most conventions (mixin set, file structure, naming, gradient-checkpointing implementation, `_no_split_modules` settings, etc.) are easiest to internalize by comparison rather than from a fixed list.

## Attention pattern

Attention must follow the diffusers pattern: both the `Attention` class and its processor are defined in the model file. The processor's `__call__` handles the actual compute and must use `dispatch_attention_fn` rather than calling `F.scaled_dot_product_attention` directly. The attention class inherits `AttentionModuleMixin` and declares `_default_processor_cls` and `_available_processors`.

```python
# transformer_mymodel.py

class MyModelAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, attention_mask=None, ...):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        # reshape, apply rope, etc.
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


class MyModelAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MyModelAttnProcessor
    _available_processors = [MyModelAttnProcessor]

    def __init__(self, query_dim, heads=8, dim_head=64, ...):
        super().__init__()
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim), nn.Dropout(0.0)])
        self.set_processor(MyModelAttnProcessor())

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return self.processor(self, hidden_states, attention_mask, **kwargs)
```

### Attention masks

What you pass as `attn_mask=` to `dispatch_attention_fn` determines which backends work:

- **No mask needed → pass `None`, not an all-zero tensor.** A dense 4D additive float mask of all `0.0` does no math but still hard-raises on `flash` / `_flash_3` / `_sage` (see `attention_dispatch.py:2328, 2544, 3266`). Only materialize a mask when it carries information. This is the Flux / Flux2 / Wan pattern: no mask, works on every backend, relies on the model having been trained tolerating consistent padding.
- **Padding mask → bool `(B, L)` or `(B, 1, 1, L)`.** Only pass when the batch actually contains different-length sequences (i.e. there is real padding). If all sequences are the same length, set the mask to `None` — many backends (flash, sage, aiter) raise `ValueError` on any non-None mask, and even SDPA-based backends pay unnecessary overhead processing a no-op mask. See `pipeline_qwenimage.py` `encode_prompt` for the pattern: `if mask.all(): mask = None`. When a mask is needed, use bool format — it stays compatible with the `*_varlen` kernels via `_normalize_attn_mask` (`attention_dispatch.py:639`), which reduces bool masks to `cu_seqlens`. Dense additive-float masks *cannot* be reduced this way and so lose the varlen path.
- **Other mask types (structural, BlockMask, etc.)** — if the model requires a different mask pattern, figure out how to support as many backends as possible (e.g. use `window_size` kwarg for sliding window on flash, `BlockMask` for Flex) and document which backends are supported for that model.
- **Don't declare `attention_mask` (or `encoder_hidden_states_mask`) in the forward signature if you ignore it.** "For API stability with other transformers" is not a reason; readers assume a declared param is honored, and downstream pipelines will pass padding masks that silently get dropped. Some existing models in the repo carry unused mask params for historical reasons — e.g. `QwenDoubleStreamAttnProcessor2_0.__call__` declares `encoder_hidden_states_mask` but never reads it (the joint mask is routed through `attention_mask` instead), and the block-level forward in `transformer_qwenimage.py` declares it but always receives `None`. This is a legacy behavior and should not be replicated in new models.

## Model class attributes

Each `ModelMixin` subclass can declare class-level attributes that configure optimization features. Each attribute corresponds to a user-facing API — the attribute controls how that feature behaves for the model. When adding a new transformer, set all that apply — skim `transformer_flux.py`, `transformer_wan.py`, `transformer_qwenimage.py` for examples.

### `_no_split_modules`

**API:** `Model.from_pretrained(..., device_map="auto")` — called in `model_loading_utils.py:87` via `model._get_no_split_modules()`, which feeds the list to `accelerate`'s `infer_auto_device_map(no_split_module_classes=...)`.

Lists which `nn.Module` subclasses must stay on a single device (i.e. never have their children placed on different devices).

- **`None` (default)** — `from_pretrained(..., device_map="auto")` raises `ValueError` (`modeling_utils.py:1863`).
- **`[]`** — split anywhere you like.
- **`["MyBlock"]`** — keep all `MyBlock` instances intact on one device.

**Why it's needed.** When `accelerate` splits a model across devices, it installs hooks on leaf modules that move inputs to the module's device before `forward` runs. Any inline operation (`+`, `*`, `torch.cat`) that combines tensors from different submodules has no hook — if those submodules landed on different devices, it crashes with "tensors on different devices". The fix is either: (a) list the parent module in `_no_split_modules` so all its children stay co-located, or (b) pack the operation into its own `nn.Module`. Inline ops on outputs from the **same** submodule call are fine since they're already on the same device.
When deciding which modules to list, inspect `forward` methods at every level of the module tree — not just the top-level model, but also its submodules recursively. Any module with inline ops combining tensors from different children or stored parameters needs to be listed.

Every transformer in the repo declares it — new transformers should too. It's cheap and prevents a confusing error when users try `device_map="auto"`.

```python
_no_split_modules = ["MyModelTransformerBlock"]
```

### `_repeated_blocks`

**API:** `model.compile_repeated_blocks(*args, **kwargs)` — walks all submodules, compiles each one whose `__class__.__name__` matches an entry in this list (`modeling_utils.py:1552`). Arguments are forwarded to `torch.compile`.

Lists the class names of the repeated sub-modules (e.g. transformer blocks) for regional compilation instead of compiling the entire model. Must match the class `__name__` exactly.

```python
# Flux: two block types
_repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
# Wan: one block type
_repeated_blocks = ["WanTransformerBlock"]
```

Typically these are the layers that run many times (e.g. the transformer blocks in the denoising loop), since those benefit most from compilation. If empty or not set, `compile_repeated_blocks()` raises `ValueError`.

### `_skip_layerwise_casting_patterns`

**API:** `model.enable_layerwise_casting(storage_dtype=..., compute_dtype=...)` — applies hooks that store weights in a low-precision dtype and cast to compute dtype on each forward. Modules matching these patterns are skipped (`modeling_utils.py:435`).

List of regex/substring patterns matching module names that should **stay in full precision**. Typically precision-sensitive layers: patch embeddings, positional embeddings, normalization layers.

```python
# Common pattern — skip embeddings and norms:
_skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
# Flux pattern:
_skip_layerwise_casting_patterns = ["pos_embed", "norm"]
```

If `None`, no modules are skipped (everything gets cast). Modules in `_keep_in_fp32_modules` are also skipped automatically.

### `_keep_in_fp32_modules`

**API:** `Model.from_pretrained(..., torch_dtype=torch.bfloat16)` — during loading, modules matching these patterns are kept in `float32` even when the rest of the model is cast to the requested dtype (`modeling_utils.py:1160`). Also respected by `enable_layerwise_casting()`.

List of module name patterns for modules that are numerically unstable in lower precision — timestep embeddings, scale/shift tables, normalization parameters.

```python
# Wan pattern:
_keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
```

If `None` (default), all modules follow the requested `torch_dtype`.

### `_cp_plan`

**API:** `model.enable_parallelism(config=parallel_config)` — when the config includes `context_parallel_config`, this plan is used by `apply_context_parallel()` to shard tensors across GPUs for sequence parallelism (`modeling_utils.py:1665`).

Dict describing how to partition the model's tensors for context parallelism. Maps parameter/activation names to their sharding strategy.

```python
# Minimal example (see transformer_flux.py, transformer_wan.py for full plans):
_cp_plan = {
    "": { ... },        # default sharding for unnamed tensors
    "rope": { ... },    # RoPE-specific sharding
}
```

If `None` (default), `enable_parallelism()` with `context_parallel_config` raises `ValueError` unless a `cp_plan` is passed explicitly as an argument. To derive a plan for a new model, study the mechanism in `hooks/context_parallel.py` and `_modeling_parallel.py`, compare existing plans in `transformer_flux.py` and `transformer_wan.py`, then test and adjust — correct plans depend on the model's data flow and require validation.

### `_supports_gradient_checkpointing`

**API:** `model.enable_gradient_checkpointing()` — walks submodules for a `gradient_checkpointing` attribute, flips it to `True`, and sets `_gradient_checkpointing_func` (`modeling_utils.py:285`).

Boolean gate. If `False` (default), calling that method raises `ValueError`. All transformers in the repo support this. To add support, just: (1) set the class attribute to `True`, (2) add `self.gradient_checkpointing = False` in `__init__`, (3) add `if torch.is_grad_enabled() and self.gradient_checkpointing:` branches in `forward` that call `self._gradient_checkpointing_func`. See gotcha #4.

## Gotchas

1. **Forgetting to register imports.** Every new class must be registered in the appropriate `__init__.py` with lazy imports — both the sub-package `__init__.py` and the top-level `src/diffusers/__init__.py` (which has `_import_structure` and `_lazy_modules`). Missing either causes `ImportError` that only shows up when users try `from diffusers import YourNewClass`.

2. **Using `einops` or other non-PyTorch deps.** Reference implementations often use `einops.rearrange`. Always rewrite with native PyTorch (`reshape`, `permute`, `unflatten`). Don't add the dependency. If a dependency is truly unavoidable, guard its import: `if is_my_dependency_available(): import my_dependency`.


3. **Capability flags without matching implementation.** for example, `_supports_gradient_checkpointing = True` only takes effect if `forward` actually has `if self.gradient_checkpointing:` branches calling `self._gradient_checkpointing_func` on each block. Setting the flag without those branches means training code silently no-ops the checkpoint and runs a normal forward.
4. **Hardcoded dtype in model forward.** Don't hardcode `torch.float32` or `torch.bfloat16`, and don't cast activations by reading a weight's dtype (`self.linear.weight.dtype`) — the stored weight dtype isn't the compute dtype under gguf / quantized loading. Always derive the cast target from the input tensor's dtype or `self.dtype`.

5. **`torch.float64` anywhere in the model.** MPS and several NPU backends don't support float64 -- ops will either error out or silently fall back. Reference repos commonly reach for float64 in RoPE frequency bases, timestep embeddings, sinusoidal position encodings, and similar "precision-sensitive" precompute code (`torch.arange(..., dtype=torch.float64)`, `.double()`, `torch.float64` literals). When porting a model, grep for `float64` / `double()` up front and resolve as follows:
    - **Default: just use `torch.float32`.** For inference it is almost always sufficient -- the precision difference in RoPE angles, timestep embeddings, etc. is immaterial to image/video quality. Flip it and move on.
    - **Only if float32 visibly degrades output, fall back to the device-gated pattern** we use in the repo:
      ```python
      is_mps = hidden_states.device.type == "mps"
      is_npu = hidden_states.device.type == "npu"
      freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
      ```
      See `transformer_flux.py`, `transformer_flux2.py`, `transformer_wan.py`, `unet_2d_condition.py` for reference usages. Never leave an unconditional `torch.float64` in the model.
