# Model conventions and rules

Shared reference for model-related conventions, patterns, and gotchas.
Linked from `AGENTS.md`, `skills/model-integration/SKILL.md`, and `review-rules.md`.

## Coding style

- All layer calls should be visible directly in `forward` — avoid helper functions that hide `nn.Module` calls.
- Avoid graph breaks for `torch.compile` compatibility — do not insert NumPy operations in forward implementations and any other patterns that can break `torch.compile` compatibility with `fullgraph=True`.
- No new mandatory dependency without discussion (e.g. `einops`). Optional deps guarded with `is_X_available()` and a dummy in `utils/dummy_*.py`.

## Common model conventions

- Models use `ModelMixin` with `register_to_config` for config serialization

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

Consult the implementations in `src/diffusers/models/transformers/` if you need further references.

## Gotchas

1. **Forgetting `__init__.py` lazy imports.** Every new class must be registered in the appropriate `__init__.py` with lazy imports. Missing this causes `ImportError` that only shows up when users try `from diffusers import YourNewClass`.

2. **Using `einops` or other non-PyTorch deps.** Reference implementations often use `einops.rearrange`. Always rewrite with native PyTorch (`reshape`, `permute`, `unflatten`). Don't add the dependency. If a dependency is truly unavoidable, guard its import: `if is_my_dependency_available(): import my_dependency`.

3. **Missing `make fix-copies` after `# Copied from`.** If you add `# Copied from` annotations, you must run `make fix-copies` to propagate them. CI will fail otherwise.

4. **Wrong `_supports_cache_class` / `_no_split_modules`.** These class attributes control KV cache and device placement. Copy from a similar model and verify -- wrong values cause silent correctness bugs or OOM errors.

5. **Missing `@torch.no_grad()` on pipeline `__call__`.** Forgetting this causes GPU OOM from gradient accumulation during inference.

6. **Config serialization gaps.** Every `__init__` parameter in a `ModelMixin` subclass must be captured by `register_to_config`. If you add a new param but forget to register it, `from_pretrained` will silently use the default instead of the saved value.

7. **Forgetting to update `_import_structure` and `_lazy_modules`.** The top-level `src/diffusers/__init__.py` has both -- missing either one causes partial import failures.

8. **Hardcoded dtype in model forward.** Don't hardcode `torch.float32` or `torch.bfloat16`, and don't cast activations by reading a weight's dtype (`self.linear.weight.dtype`) — the stored weight dtype isn't the compute dtype under gguf / quantized loading. Always derive the cast target from the input tensor's dtype or `self.dtype`.

9. **`torch.float64` anywhere in the model.** MPS and several NPU backends don't support float64 -- ops will either error out or silently fall back. Reference repos commonly reach for float64 in RoPE frequency bases, timestep embeddings, sinusoidal position encodings, and similar "precision-sensitive" precompute code (`torch.arange(..., dtype=torch.float64)`, `.double()`, `torch.float64` literals). When porting a model, grep for `float64` / `double()` up front and resolve as follows:
    - **Default: just use `torch.float32`.** For inference it is almost always sufficient -- the precision difference in RoPE angles, timestep embeddings, etc. is immaterial to image/video quality. Flip it and move on.
    - **Only if float32 visibly degrades output, fall back to the device-gated pattern** we use in the repo:
      ```python
      is_mps = hidden_states.device.type == "mps"
      is_npu = hidden_states.device.type == "npu"
      freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
      ```
      See `transformer_flux.py`, `transformer_flux2.py`, `transformer_wan.py`, `unet_2d_condition.py` for reference usages. Never leave an unconditional `torch.float64` in the model.
