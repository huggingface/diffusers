---
name: integrating-models
description: >
  Use when adding a new model or pipeline to diffusers, setting up file
  structure for a new model, converting a pipeline to modular format, or
  converting weights for a new version of an already-supported model.
---

## Goal

Integrate a new model into diffusers end-to-end. The overall flow:

1. **Gather info** — ask the user for the reference repo, setup guide, a runnable inference script, and other objectives such as standard vs modular.
2. **Confirm the plan** — once you have everything, tell the user exactly what you'll do: e.g. "I'll integrate model X with pipeline Y into diffusers based on your script. I'll run parity tests (model-level and pipeline-level) using the `parity-testing` skill to verify numerical correctness against the reference."
3. **Implement** — write the diffusers code (model, pipeline, scheduler if needed), convert weights, register in `__init__.py`.
4. **Parity test** — use the `parity-testing` skill to verify component and e2e parity against the reference implementation.
5. **Deliver a unit test** — provide a self-contained test script that runs the diffusers implementation, checks numerical output (np allclose), and saves an image/video for visual verification. This is what the user runs to confirm everything works.

Work one workflow at a time — get it to full parity before moving on.

## Setup — gather before starting

Before writing any code, gather info in this order:

1. **Reference repo** — ask for the github link. If they've already set it up locally, ask for the path. Otherwise, ask what setup steps are needed (install deps, download checkpoints, set env vars, etc.) and run through them before proceeding.
2. **Inference script** — ask for a runnable end-to-end script for a basic workflow first (e.g. T2V). Then ask what other workflows they want to support (I2V, V2V, etc.) and agree on the full implementation order together.
3. **Standard vs modular** — standard pipelines, modular, or both?

Use `AskUserQuestion` with structured choices for step 3 when the options are known.

## Standard Pipeline Integration

### File structure for a new model

```
src/diffusers/
  models/transformers/transformer_<model>.py     # The core model
  schedulers/scheduling_<model>.py               # If model needs a custom scheduler
  pipelines/<model>/
    __init__.py
    pipeline_<model>.py                          # Main pipeline
    pipeline_<model>_<variant>.py                # Variant pipelines (e.g. pyramid, distilled)
    pipeline_output.py                           # Output dataclass
  loaders/lora_pipeline.py                       # LoRA mixin (add to existing file)

tests/
  models/transformers/test_models_transformer_<model>.py
  pipelines/<model>/test_<model>.py
  lora/test_lora_layers_<model>.py

docs/source/en/api/
  pipelines/<model>.md
  models/<model>_transformer3d.md                # or appropriate name
```

### Integration checklist

- [ ] Implement transformer model with `from_pretrained` support
- [ ] Implement or reuse scheduler
- [ ] Implement pipeline(s) with `__call__` method
- [ ] Add LoRA support if applicable
- [ ] Register all classes in `__init__.py` files (lazy imports)
- [ ] Write unit tests (model, pipeline, LoRA)
- [ ] Write docs
- [ ] Run `make style` and `make quality`
- [ ] Test parity with reference implementation (see `parity-testing` skill)

### Attention pattern

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

### Implementation rules

1. **Don't combine structural changes with behavioral changes.** Restructuring code to fit diffusers APIs (ModelMixin, ConfigMixin, etc.) is unavoidable. But don't also "improve" the algorithm, refactor computation order, or rename internal variables for aesthetics. Keep numerical logic as close to the reference as possible, even if it looks unclean. For standard → modular, this is stricter: copy loop logic verbatim and only restructure into blocks. Clean up in a separate commit after parity is confirmed.
2. **Pipelines must inherit from `DiffusionPipeline`.** Consult implementations in `src/diffusers/pipelines` in case you need references.
3. **Don't subclass an existing pipeline for a variant.** DO NOT use an existing pipeline class (e.g., `FluxPipeline`) to override another pipeline (e.g., `FluxImg2ImgPipeline`) which will be a part of the core codebase (`src`).

### Test setup

- Slow tests gated with `@slow` and `RUN_SLOW=1`
- All model-level tests must use the `BaseModelTesterConfig`, `ModelTesterMixin`, `MemoryTesterMixin`, `AttentionTesterMixin`, `LoraTesterMixin`, and `TrainingTesterMixin` classes initially to write the tests. Any additional tests should be added after discussions with the maintainers. Use `tests/models/transformers/test_models_transformer_flux.py` as a reference.

### Common diffusers conventions

- Pipelines inherit from `DiffusionPipeline`
- Models use `ModelMixin` with `register_to_config` for config serialization
- Schedulers use `SchedulerMixin` with `ConfigMixin`
- Use `@torch.no_grad()` on pipeline `__call__`
- Support `output_type="latent"` for skipping VAE decode
- Support `generator` parameter for reproducibility
- Use `self.progress_bar(timesteps)` for progress tracking

## Gotchas

1. **Forgetting `__init__.py` lazy imports.** Every new class must be registered in the appropriate `__init__.py` with lazy imports. Missing this causes `ImportError` that only shows up when users try `from diffusers import YourNewClass`.

2. **Using `einops` or other non-PyTorch deps.** Reference implementations often use `einops.rearrange`. Always rewrite with native PyTorch (`reshape`, `permute`, `unflatten`). Don't add the dependency. If a dependency is truly unavoidable, guard its import: `if is_my_dependency_available(): import my_dependency`.

3. **Missing `make fix-copies` after `# Copied from`.** If you add `# Copied from` annotations, you must run `make fix-copies` to propagate them. CI will fail otherwise.

4. **Wrong `_supports_cache_class` / `_no_split_modules`.** These class attributes control KV cache and device placement. Copy from a similar model and verify -- wrong values cause silent correctness bugs or OOM errors.

5. **Missing `@torch.no_grad()` on pipeline `__call__`.** Forgetting this causes GPU OOM from gradient accumulation during inference.

6. **Config serialization gaps.** Every `__init__` parameter in a `ModelMixin` subclass must be captured by `register_to_config`. If you add a new param but forget to register it, `from_pretrained` will silently use the default instead of the saved value.

7. **Forgetting to update `_import_structure` and `_lazy_modules`.** The top-level `src/diffusers/__init__.py` has both -- missing either one causes partial import failures.

8. **Hardcoded dtype in model forward.** Don't hardcode `torch.float32` or `torch.bfloat16` in the model's forward pass. Use the dtype of the input tensors or `self.dtype` so the model works with any precision.

---

## Modular Pipeline Conversion

See [modular-conversion.md](modular-conversion.md) for the full guide on converting standard pipelines to modular format, including block types, build order, guider abstraction, and conversion checklist.

---

## Weight Conversion Tips

<!-- TODO: Add concrete examples as we encounter them. Common patterns to watch for:
  - Fused QKV weights that need splitting into separate Q, K, V
  - Scale/shift ordering differences (reference stores [shift, scale], diffusers expects [scale, shift])
  - Weight transpositions (linear stored as transposed conv, or vice versa)
  - Interleaved head dimensions that need reshaping
  - Bias terms absorbed into different layers
  Add each with a before/after code snippet showing the conversion. -->
