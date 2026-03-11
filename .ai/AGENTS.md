# Diffusers — Agent Guide

### Philosophy

Write code as simple and explicit as possible.

- Minimize small helper/utility functions — inline the logic instead. A reader should be able to follow the full flow without jumping between functions.
- No defensive code or unused code paths — do not add fallback paths, safety checks, or configuration options "just in case". When porting from a research repo, delete training-time code paths, experimental flags, and ablation branches entirely — only keep the inference path you are actually integrating.
- Do not guess user intent and silently correct behavior. Make the expected inputs clear in the docstring, and raise a concise error for unsupported cases rather than adding complex fallback logic.

---

### Dependencies
- No new mandatory dependency without discussion (e.g. `einops`)
- Optional deps guarded with `is_X_available()` and a dummy in `utils/dummy_*.py`

### Code Style
- `make style` and `make fix-copies` should be run as the final step before opening a PR

### Copied Code
- Many classes are kept in sync with a source via a `# Copied from ...` header comment
- Do not edit a `# Copied from` block directly — run `make fix-copies` to propagate changes from the source
- Remove the header to intentionally break the link

### Models
- All layer calls should be visible directly in `forward` — avoid helper functions that hide `nn.Module` calls.
- Attention must follow the diffusers pattern: both the `Attention` class and its processor are defined in the model file. The processor's `__call__` handles the actual compute and must use `dispatch_attention_fn` rather than calling `F.scaled_dot_product_attention` directly. The attention class inherits `AttentionModuleMixin` and declares `_default_processor_cls` and `_available_processors`.

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

### Pipeline
- All pipelines must inherit from `DiffusionPipeline`

### Tests
- Slow tests gated with `@slow` and `RUN_SLOW=1`